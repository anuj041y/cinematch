"""Find missing or broken movie posters and save verified replacements."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import PROCESSED_DATA_DIR, TMDB_API_KEY, WATCHMODE_API_KEY
from src.posters import PosterLookupRateLimit, poster_source, remote_image_status

STATUS_TTL = 30 * 24 * 60 * 60
LOOKUP_TTL = 7 * 24 * 60 * 60


def _clean_field(value: object) -> str:
    text = "" if value is None else str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"} else text


def _load_cache(path: Path) -> dict[str, dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {"links": payload.get("links", {}), "lookups": payload.get("lookups", {})}
    except (OSError, json.JSONDecodeError, AttributeError):
        return {"links": {}, "lookups": {}}


def _save_cache(cache: dict[str, dict], path: Path) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    try:
        temp_path.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _fresh(entry: object, ttl: int, now: float) -> bool:
    if not isinstance(entry, dict):
        return False
    try:
        return now - float(entry.get("checked_at", 0)) < ttl
    except (TypeError, ValueError, OverflowError):
        return False


def _lookup_key(row: pd.Series) -> str:
    title = _clean_field(row.get("title")).casefold()
    year = _clean_field(row.get("release_year", row.get("year")))
    return f"{title}\x1f{year}"


def _lookup_one(index: int, key: str, row: pd.Series) -> tuple[int, str, str | Path | None, bool | None, bool]:
    try:
        replacement = poster_source(
            row, ROOT, api_key=TMDB_API_KEY, allow_lookup=True,
            watchmode_api_key=WATCHMODE_API_KEY,
        )
    except PosterLookupRateLimit:
        return index, key, None, None, True
    if isinstance(replacement, str) and replacement.startswith(("https://", "http://")):
        return index, key, replacement, remote_image_status(replacement), False
    return index, key, replacement, False, False


def _save_catalog(catalog: pd.DataFrame, path: Path) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    try:
        catalog.to_csv(temp_path, index=False)
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50,
                        help="number of movies to process (default: 50)")
    parser.add_argument("--all", action="store_true",
                        help="process every missing movie, or the full catalog with --repair-broken")
    parser.add_argument("--repair-broken", action="store_true",
                        help="also check existing catalog poster links for 404s")
    parser.add_argument("--title", action="append", default=[],
                        help="process this exact title; repeat to add titles")
    parser.add_argument("--workers", type=int, default=32,
                        help="parallel image-link checks (default: 32)")
    parser.add_argument("--lookup-workers", type=int, default=4,
                        help="parallel, rate-limited API lookups (default: 4)")
    parser.add_argument("--max-api-lookups", type=int, default=100,
                        help="maximum new API searches this run (default: 100)")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if not 1 <= args.workers <= 64:
        parser.error("--workers must be between 1 and 64")
    if not 1 <= args.lookup_workers <= 8:
        parser.error("--lookup-workers must be between 1 and 8")
    if args.max_api_lookups < 0:
        parser.error("--max-api-lookups must be zero or greater")
    if not (TMDB_API_KEY or WATCHMODE_API_KEY):
        print("No poster API key found. Set WATCHMODE_API_KEY or TMDB_API_KEY in .env, then retry.")
        return 2

    catalog_path = PROCESSED_DATA_DIR / "movies_clean.csv"
    if not catalog_path.is_file():
        print(f"Prepared catalog not found: {catalog_path}")
        print("Run `python scripts/prepare_data.py` first.")
        return 2

    catalog = pd.read_csv(catalog_path, low_memory=False)
    for column in ("poster_path", "poster_url"):
        if column not in catalog:
            catalog[column] = ""
        catalog[column] = catalog[column].map(_clean_field)

    has_source = catalog.poster_path.ne("") | catalog.poster_url.ne("")
    missing = catalog.index[~has_source].tolist()
    candidates = catalog.index.tolist() if args.repair_broken else missing
    if args.title:
        requested_titles = {title.strip().casefold() for title in args.title}
        candidates = [index for index in candidates
                      if _clean_field(catalog.at[index, "title"]).casefold() in requested_titles]
    elif candidates:
        popularity = pd.to_numeric(catalog.get("popularity", pd.Series(0, index=catalog.index)),
                                   errors="coerce").fillna(0)
        rating_count = pd.to_numeric(catalog.get("rating_count", pd.Series(0, index=catalog.index)),
                                     errors="coerce").fillna(0)
        candidates = catalog.loc[candidates].assign(
            _popularity=popularity.loc[candidates], _rating_count=rating_count.loc[candidates]
        ).sort_values(["_popularity", "_rating_count"], ascending=False).index.tolist()

    selected = candidates if args.all else candidates[:args.limit]
    if not selected:
        print("No matching movies need poster processing.")
        return 0

    fallback_paths = {
        (ROOT / "assets" / "default_poster.png").resolve(),
        (ROOT / "assets" / "default_poster.svg").resolve(),
    }
    sources: dict[int, str | Path] = {}
    check_urls: set[str] = set()
    needs_lookup: set[int] = set()
    cache_path = PROCESSED_DATA_DIR / ".poster_repair_cache.json"
    cache = _load_cache(cache_path)
    now = time.time()
    for index in selected:
        source = poster_source(catalog.loc[index], ROOT)
        sources[index] = source
        if isinstance(source, Path):
            if source.resolve() in fallback_paths:
                needs_lookup.add(index)
        else:
            check_urls.add(source)

    statuses: dict[str, bool | None] = {}
    if args.repair_broken and check_urls:
        urls = sorted(check_urls)
        pending_urls = []
        for url in urls:
            entry = cache["links"].get(url)
            if _fresh(entry, STATUS_TTL, now) and entry.get("status") in (True, False):
                statuses[url] = entry["status"]
            else:
                pending_urls.append(url)
        print(f"Checking {len(pending_urls):,} new links; using {len(statuses):,} cached results.")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0, len(pending_urls), 500):
                batch = pending_urls[start:start + 500]
                results = list(pool.map(remote_image_status, batch))
                statuses.update(zip(batch, results))
                for url, status in zip(batch, results):
                    if status is not None:
                        cache["links"][url] = {"status": status, "checked_at": time.time()}
                checked = min(start + len(batch), len(pending_urls))
                if checked % 5000 < len(batch) or checked == len(pending_urls):
                    _save_cache(cache, cache_path)
                    print(f"  Checked {checked:,}/{len(pending_urls):,} new links")

        for index, source in sources.items():
            if not isinstance(source, str):
                continue
            status = statuses.get(source)
            if status is False:
                needs_lookup.add(index)

    broken_count = sum(status is False for status in statuses.values())
    unknown_count = sum(status is None for status in statuses.values())
    valid_count = sum(status is True for status in statuses.values())
    print(f"Poster links: {valid_count:,} working, {broken_count:,} broken, {unknown_count:,} unverified.")
    print(f"Looking up replacements for {len(needs_lookup):,} movies...")

    updated = no_match = unverified_replacements = 0
    api_lookups = 0
    cached_lookups = 0
    rate_limited = False
    pending_queries: list[tuple[int, str, pd.Series]] = []
    for index in selected:
        if index not in needs_lookup:
            continue
        row = catalog.loc[index]
        key = _lookup_key(row)
        cached = cache["lookups"].get(key)
        if _fresh(cached, LOOKUP_TTL, time.time()):
            cached_lookups += 1
            cached_url = _clean_field(cached.get("url"))
            if not cached_url:
                no_match += 1
                continue
            cached_status = cache["links"].get(cached_url)
            if _fresh(cached_status, STATUS_TTL, time.time()) and cached_status.get("status") is True:
                catalog.at[index, "poster_url"] = cached_url
                updated += 1
                continue
            cache["lookups"].pop(key, None)
        pending_queries.append((index, key, row))

    to_query = pending_queries[:args.max_api_lookups]
    deferred = max(0, len(pending_queries) - len(to_query))
    if to_query:
        print(f"Searching {len(to_query):,} titles with {args.lookup_workers} workers; {deferred:,} will wait for the next run.")
    with ThreadPoolExecutor(max_workers=args.lookup_workers) as pool:
        for start in range(0, len(to_query), args.lookup_workers):
            batch = to_query[start:start + args.lookup_workers]
            results = list(pool.map(lambda item: _lookup_one(*item), batch))
            for index, key, replacement, status, limited in results:
                api_lookups += 1
                if limited:
                    rate_limited = True
                    continue
                if isinstance(replacement, str) and replacement.startswith(("https://", "http://")):
                    if status is True:
                        catalog.at[index, "poster_url"] = replacement
                        cache["links"][replacement] = {"status": True, "checked_at": time.time()}
                        cache["lookups"][key] = {"url": replacement, "checked_at": time.time()}
                        updated += 1
                    elif status is None:
                        unverified_replacements += 1
                    else:
                        cache["lookups"][key] = {"url": "", "checked_at": time.time()}
                        no_match += 1
                else:
                    cache["lookups"][key] = {"url": "", "checked_at": time.time()}
                    no_match += 1

            if api_lookups and api_lookups % 25 == 0:
                _save_catalog(catalog, catalog_path)
                _save_cache(cache, cache_path)
                print(f"  Searched {api_lookups:,}/{len(to_query):,} titles; saved {updated:,} posters.")
            if rate_limited:
                print("Poster API rate limit reached; saving progress and stopping.")
                deferred += len(to_query) - start - len(batch)
                break

    _save_catalog(catalog, catalog_path)
    _save_cache(cache, cache_path)
    print(f"Saved {updated:,} verified poster replacements to {catalog_path}.")
    print(f"API searches: {api_lookups:,}; cached results: {cached_lookups:,}; no match: {no_match:,}; image check timed out: {unknown_count:,}; API match unverified: {unverified_replacements:,}.")
    if rate_limited:
        print("Run the same command again after the API quota resets to continue.")
    elif deferred:
        print(f"{deferred:,} replacements were deferred by --max-api-lookups. Run the same command again to continue from the saved cache.")
    elif args.repair_broken and not args.all and len(candidates) > len(selected):
        print(f"{len(candidates) - len(selected):,} movies were not scanned. Add --all to scan the full catalog.")
    elif not args.all and len(candidates) > len(selected):
        print(f"{len(candidates) - len(selected):,} movies still need a lookup. Run again or use --all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
