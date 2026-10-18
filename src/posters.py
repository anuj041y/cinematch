"""Poster lookup and fallback helpers."""

from __future__ import annotations

import base64
import html
import math
import mimetypes
import re
import threading
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests

from config.settings import ASSETS_DIR, TMDB_API_KEY, WATCHMODE_API_KEY

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_TMDB_PATH = re.compile(r"^/[A-Za-z0-9._-]+\.(?:jpg|jpeg|png|webp)$", re.IGNORECASE)
_LOOKUP_CACHE: dict[tuple[str, str], tuple[float, str | None]] = {}
_WATCHMODE_CACHE: dict[tuple[str, str, str], tuple[float, str | None]] = {}
_REMOTE_IMAGE_CACHE: dict[str, tuple[float, bool | None]] = {}
_LOOKUP_LOCK = threading.Lock()
_HTTP_LOCAL = threading.local()
_WATCHMODE_RATE_LOCK = threading.Lock()
_WATCHMODE_NEXT_REQUEST = 0.0


class PosterLookupRateLimit(RuntimeError):
    """Raised when a poster provider asks the repair script to slow down."""


def _wait_for_watchmode_slot() -> None:
    global _WATCHMODE_NEXT_REQUEST
    with _WATCHMODE_RATE_LOCK:
        now = time.monotonic()
        delay = max(0.0, _WATCHMODE_NEXT_REQUEST - now)
        _WATCHMODE_NEXT_REQUEST = max(now, _WATCHMODE_NEXT_REQUEST) + 0.55
    if delay:
        time.sleep(delay)


def _value(value: Any) -> str:
    if value is None:
        return ""
    try:
        missing = value != value
        if isinstance(missing, bool) and missing:
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"} else text


def _field(movie: Any, *names: str) -> str:
    for name in names:
        try:
            value = _value(movie.get(name))
        except (AttributeError, TypeError):
            value = ""
        if value:
            return value
    return ""


def _is_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
        return parsed.scheme.lower() in {"http", "https"} and bool(parsed.hostname) and not parsed.username and not parsed.password
    except ValueError:
        return False


def _local_image(value: str, project_root: Path) -> Path | None:
    if not value or _is_http_url(value):
        return None
    candidate = Path(value).expanduser()
    options = [candidate] if candidate.is_absolute() else [project_root / candidate, project_root / "data" / candidate]
    for option in options:
        try:
            resolved = option.resolve()
            if resolved.is_file() and resolved.suffix.lower() in _IMAGE_SUFFIXES:
                from PIL import Image
                with Image.open(resolved) as image:
                    image.verify()
                return resolved
        except (OSError, ValueError, SyntaxError):
            continue
    return None


def _fallback_path(project_root: Path) -> Path:
    for candidate in (project_root / "assets" / "default_poster.png", ASSETS_DIR / "default_poster.png",
                      project_root / "assets" / "default_poster.svg", ASSETS_DIR / "default_poster.svg"):
        if candidate.is_file():
            return candidate.resolve()
    return (project_root / "assets" / "default_poster.png").resolve()


@lru_cache(maxsize=32)
def _local_data_uri(path: str, modified_ns: int) -> str:
    del modified_ns
    image_path = Path(path)
    mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    content = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{content}"


def _tmdb_path_url(value: str) -> str | None:
    path = value.strip()
    if not _TMDB_PATH.fullmatch(path) or ".." in path:
        return None
    return f"{TMDB_IMAGE_BASE}{path}"


def _remote_image_status(url: str) -> bool | None:
    """Check a remote image briefly; None means the server could not verify it."""
    now = time.monotonic()
    with _LOOKUP_LOCK:
        cached = _REMOTE_IMAGE_CACHE.get(url)
        if cached and cached[0] > now:
            return cached[1]

    response = None
    available: bool | None = None
    try:
        session = getattr(_HTTP_LOCAL, "session", None)
        if session is None:
            session = requests.Session()
            _HTTP_LOCAL.session = session
        response = session.head(url, allow_redirects=True, timeout=(1, 2))
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if response.status_code in {408, 425, 429} or response.status_code >= 500:
            available = None
        elif response.status_code >= 400:
            available = False
        else:
            available = content_type.startswith("image/") or content_type in {"", "application/octet-stream"}
    except requests.RequestException:
        available = None
    finally:
        if response is not None:
            response.close()

    lifetime = 6 * 60 * 60 if available is True else 5 * 60 if available is False else 30
    with _LOOKUP_LOCK:
        if len(_REMOTE_IMAGE_CACHE) >= 1024:
            expired = [key for key, (expires, _) in _REMOTE_IMAGE_CACHE.items() if expires <= now]
            for key in expired:
                _REMOTE_IMAGE_CACHE.pop(key, None)
            if len(_REMOTE_IMAGE_CACHE) >= 1024:
                _REMOTE_IMAGE_CACHE.pop(next(iter(_REMOTE_IMAGE_CACHE)))
        _REMOTE_IMAGE_CACHE[url] = (now + lifetime, available)
    return available


def remote_image_status(url: str) -> bool | None:
    """Return True for an image response, False for a broken link, or None if unverified."""
    if not _is_http_url(url):
        return False
    return _remote_image_status(url)


def _tmdb_id(movie: Any) -> str:
    explicit = _field(movie, "tmdb_id", "tmdbId", "tmdbID")
    canonical = _field(movie, "movie_id", "movieId")
    if not explicit and canonical.startswith("tmdb:"):
        explicit = canonical.split(":", 1)[1]
    # A MovieLens movieId must never be treated as a TMDB ID.
    if not explicit and canonical.startswith("tmdb:"):
        explicit = _field(movie, "id")
    elif not explicit and not _field(movie, "movieId") and not canonical:
        explicit = _field(movie, "id")
    raw = explicit.strip()
    try:
        number = float(raw)
        if math.isfinite(number) and number > 0 and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError, OverflowError):
        pass
    return ""


def _tmdb_lookup(title: str, api_key: str, tmdb_id: str = "") -> str | None:
    """Look up one poster with bounded timeouts and temporary success/failure caching."""
    clean_title = title.strip()
    if not api_key or (not clean_title and not tmdb_id):
        return None
    cache_key = (tmdb_id or clean_title.casefold(), api_key)
    now = time.monotonic()
    with _LOOKUP_LOCK:
        cached = _LOOKUP_CACHE.get(cache_key)
        if cached and cached[0] > now:
            return cached[1]
    poster_url = None
    try:
        if tmdb_id:
            endpoint = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
            response = requests.get(endpoint, params={"api_key": api_key}, timeout=(2, 4))
            response.raise_for_status()
            payload = response.json()
        else:
            response = requests.get("https://api.themoviedb.org/3/search/movie",
                                    params={"api_key": api_key, "query": clean_title}, timeout=(2, 4))
            response.raise_for_status()
            results = response.json().get("results", [])
            payload = results[0] if isinstance(results, list) and results and isinstance(results[0], dict) else {}
        path = _value(payload.get("poster_path")) if isinstance(payload, dict) else ""
        poster_url = _tmdb_path_url(path)
    except Exception:
        poster_url = None
    # Keep successful lookups for six hours and failures for two minutes.
    lifetime = 6 * 60 * 60 if poster_url else 2 * 60
    with _LOOKUP_LOCK:
        if len(_LOOKUP_CACHE) >= 512:
            expired = [key for key, (expires, _) in _LOOKUP_CACHE.items() if expires <= now]
            for key in expired:
                _LOOKUP_CACHE.pop(key, None)
            if len(_LOOKUP_CACHE) >= 512:
                _LOOKUP_CACHE.pop(next(iter(_LOOKUP_CACHE)))
        _LOOKUP_CACHE[cache_key] = (now + lifetime, poster_url)
    return poster_url


def _normalized_title(value: str) -> str:
    without_year = re.sub(r"\s*\(\d{4}\)\s*$", "", value.strip())
    folded = unicodedata.normalize("NFKD", without_year).encode("ascii", "ignore").decode("ascii")
    folded = re.sub(r"\bcolour\b", "color", folded.casefold())
    return re.sub(r"[^a-z0-9]+", "", folded)


def _watchmode_title_matches(candidate: str, title: str) -> bool:
    title_key = _normalized_title(title)
    if _normalized_title(candidate) == title_key:
        return True
    main_title = re.split(r"\s*[:–—-]\s*", candidate.strip(), maxsplit=1)[0]
    return _normalized_title(main_title) == title_key


def _watchmode_lookup(title: str, year: str, api_key: str, *, request: bool) -> str | None:
    """Search Watchmode once for a catalog movie with no poster."""
    title_key = _normalized_title(title)
    if not api_key or not title_key:
        return None
    cache_key = (title_key, year, api_key)
    now = time.monotonic()
    with _LOOKUP_LOCK:
        cached = _WATCHMODE_CACHE.get(cache_key)
        if cached and cached[0] > now:
            return cached[1]
    if not request:
        return None

    poster_url = None
    try:
        _wait_for_watchmode_slot()
        response = requests.get(
            "https://api.watchmode.com/v1/autocomplete-search/",
            params={"search_value": title, "search_type": 3},
            headers={"X-API-Key": api_key},
            timeout=(2, 3),
        )
        if response.status_code == 429:
            raise PosterLookupRateLimit("Watchmode rate limit reached.")
        response.raise_for_status()
        payload = response.json()
        results = payload.get("results", []) if isinstance(payload, dict) else []
        exact_matches = [item for item in results if isinstance(item, dict)
                         and _watchmode_title_matches(_value(item.get("name", "")), title)]
        if year:
            target_year = int(year)
            result_years = []
            for item in exact_matches:
                match = re.search(r"\b(18|19|20|21)\d{2}\b", str(item.get("year", "")))
                result_years.append((item, int(match.group(0)) if match else None))
            same_year = [item for item, result_year in result_years if result_year == target_year]
            nearby_year = [item for item, result_year in result_years
                           if result_year is not None and abs(result_year - target_year) == 1]
            if same_year:
                exact_matches = same_year
            elif nearby_year:
                exact_matches = nearby_year
            elif any(result_year is not None for _, result_year in result_years):
                exact_matches = []
        def relevance(item: dict[str, Any]) -> float:
            try:
                return float(item.get("relevance", 0) or 0)
            except (TypeError, ValueError, OverflowError):
                return 0.0

        exact_matches.sort(key=relevance, reverse=True)
        for item in exact_matches:
            candidate = next((_value(item.get(field)) for field in
                              ("poster", "posterMedium", "posterLarge", "image_url")
                              if _value(item.get(field))), "")
            if _is_http_url(candidate):
                parsed = urlparse(candidate)
                poster_url = parsed._replace(scheme="https").geturl()
                break
    except PosterLookupRateLimit:
        raise
    except Exception:
        poster_url = None

    lifetime = 6 * 60 * 60 if poster_url else 2 * 60
    with _LOOKUP_LOCK:
        if len(_WATCHMODE_CACHE) >= 512:
            expired = [key for key, (expires, _) in _WATCHMODE_CACHE.items() if expires <= now]
            for key in expired:
                _WATCHMODE_CACHE.pop(key, None)
            if len(_WATCHMODE_CACHE) >= 512:
                _WATCHMODE_CACHE.pop(next(iter(_WATCHMODE_CACHE)))
        _WATCHMODE_CACHE[cache_key] = (now + lifetime, poster_url)
    return poster_url


def poster_source(movie: Any, project_root: str | Path, api_key: str | None = None,
                  *, allow_lookup: bool = False, watchmode_api_key: str | None = None) -> str | Path:
    """Find a poster from local data, catalog URLs, or an optional API lookup."""
    root = Path(project_root).resolve()
    raw_values = [_field(movie, name) for name in ("poster_path", "poster_url", "poster", "posterUrl")]

    # Prefer a local image over a remote source.
    for value in raw_values:
        local = _local_image(value, root)
        if local:
            return local

    # Saved replacements must take priority over the original catalog path.
    remote_candidates = []
    for name in ("poster_url", "posterUrl", "poster", "poster_path"):
        value = _field(movie, name)
        if _is_http_url(value):
            remote_candidates.append(value)
        elif url := _tmdb_path_url(value):
            remote_candidates.append(url)
    remote_candidates = list(dict.fromkeys(remote_candidates))
    if not allow_lookup and remote_candidates:
        return remote_candidates[0]

    unverified = None
    for url in remote_candidates:
        status = _remote_image_status(url)
        if status is True:
            return url
        if status is None and unverified is None:
            unverified = url
    if unverified:
        return unverified

    key = (TMDB_API_KEY if api_key is None else api_key).strip()
    if allow_lookup and key:
        lookup = _tmdb_lookup(_field(movie, "title"), key, _tmdb_id(movie))
        if lookup and _remote_image_status(lookup) is not False:
            return lookup

    watchmode_key = (WATCHMODE_API_KEY if watchmode_api_key is None else watchmode_api_key).strip()
    title = _field(movie, "title")
    year_value = _field(movie, "release_year", "year")
    year_match = re.search(r"\b(18|19|20|21)\d{2}\b", year_value)
    year = year_match.group(0) if year_match else ""
    lookup = _watchmode_lookup(title, year, watchmode_key, request=allow_lookup)
    if lookup:
        return lookup
    return _fallback_path(root)


def poster_html(source: str | Path, project_root: str | Path, alt: str = "Movie poster") -> str:
    """Build a poster block with the bundled image as a fallback."""
    root = Path(project_root).resolve()
    fallback = root / "assets" / "default_poster.svg"
    if not fallback.is_file():
        fallback = _fallback_path(root)
    fallback_uri = _local_data_uri(str(fallback), fallback.stat().st_mtime_ns)
    if isinstance(source, Path):
        image = source.resolve()
        image_uri = _local_data_uri(str(image), image.stat().st_mtime_ns)
    else:
        url = str(source)
        if not _is_http_url(url):
            raise ValueError("Poster HTML needs a local image path or a complete HTTP(S) URL.")
        css_url = url.replace("\\", "%5C").replace("'", "%27").replace('"', "%22")
        image_uri = html.escape(css_url, quote=True)
    accessible_alt = html.escape(alt, quote=True)
    return (
        f"<div role='img' aria-label='{accessible_alt}' "
        f"style=\"width:100%;aspect-ratio:2/3;border-radius:8px;"
        f"background-image:url('{image_uri}'),url('{fallback_uri}');"
        "background-size:cover;background-position:center;background-color:#24242b;\"></div>"
    )
