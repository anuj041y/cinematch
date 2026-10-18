"""Load and prepare movie and rating data."""

from __future__ import annotations

import ast
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


MOVIE_COLUMNS = [
    "movie_id", "title", "release_year", "genres", "overview", "keywords",
    "cast", "director", "avg_rating", "rating_count", "runtime", "popularity",
    "poster_path", "poster_url", "tmdb_id", "imdb_id",
]
RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]


class DatasetError(RuntimeError):
    """Raised when supplied dataset files cannot be used."""


def _column(frame: pd.DataFrame, *names: str) -> str | None:
    lowered = {str(c).strip().lower(): c for c in frame.columns}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def normalize_external_id(value: Any) -> str | None:
    """Return a safe integer-like external ID without truncating arbitrary values."""
    if value is None or pd.isna(value):
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return None
    if re.fullmatch(r"[+-]?\d+", raw):
        try:
            return str(int(raw))
        except ValueError:
            return None
    if re.fullmatch(r"[+-]?\d+\.0+", raw):
        try:
            return str(int(raw.split(".", 1)[0]))
        except ValueError:
            return None
    return None


def clean_text(value: Any, default: str = "") -> str:
    """Convert scalar metadata to clean text without turning nulls into 'nan'."""
    if value is None:
        return default
    if not isinstance(value, (list, tuple, dict, set)):
        try:
            if pd.isna(value):
                return default
        except (TypeError, ValueError):
            pass
    text = str(value).strip()
    return default if text.casefold() in {"", "nan", "none", "null", "<na>", "nat"} else text


def parse_names(value: Any, key: str = "name") -> list[str]:
    """Parse list-like genre/credit/keyword fields from JSON or MovieLens text."""
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        text = clean_text(value)
        if not text:
            return []
        try:
            raw = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            try:
                raw = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                # MovieLens genres are pipe-separated; ordinary text remains one feature.
                raw = text.split("|") if "|" in text else [text]
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    output: list[str] = []
    for item in raw:
        candidate = item.get(key) if isinstance(item, dict) else item
        candidate = clean_text(candidate)
        if candidate and candidate.casefold() != "(no genres listed)" and candidate not in output:
            output.append(candidate)
    return output


def format_title(value: Any) -> str:
    return clean_text(value, "Untitled movie")


def format_year(value: Any) -> str | None:
    try:
        year = float(value)
        if not math.isfinite(year) or not year.is_integer() or not 1000 <= year <= 9999:
            return None
        return str(int(year))
    except (TypeError, ValueError, OverflowError):
        return None


def format_genres(value: Any, separator: str = " · ") -> str:
    return separator.join(parse_names(value))


def format_overview(value: Any) -> str:
    return clean_text(value)


def format_runtime(value: Any) -> str | None:
    try:
        minutes = float(value)
        if not math.isfinite(minutes) or minutes <= 0:
            return None
        label = str(int(minutes)) if minutes.is_integer() else f"{minutes:g}"
        return f"{label} min"
    except (TypeError, ValueError, OverflowError):
        return None


def format_people(value: Any, separator: str = ", ") -> str:
    names = parse_names(value)
    return separator.join(names)


def format_rating_and_votes(rating: Any, vote_count: Any) -> str | None:
    try:
        score = float(rating)
        if not math.isfinite(score) or score <= 0:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    label = f"⭐ {score:.1f}/5"
    try:
        count = float(vote_count)
        if math.isfinite(count) and count > 0:
            label += f" · {int(count):,} ratings"
    except (TypeError, ValueError, OverflowError):
        pass
    return label


def normalize_catalog(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize catalog fields and numeric metadata."""
    result = frame.copy()
    for column in ("movie_id", "poster_path", "poster_url", "tmdb_id", "imdb_id"):
        if column in result:
            result[column] = result[column].map(clean_text)
    if "overview" in result:
        result["overview"] = result["overview"].map(format_overview)
    if "title" in result:
        result["title"] = result["title"].map(format_title)
    for column in ("genres", "keywords", "cast", "director"):
        if column in result:
            result[column] = result[column].map(lambda value: "|".join(parse_names(value)))
    for column in ("release_year", "avg_rating", "rating_count", "runtime", "popularity"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if "release_year" in result:
        years = result["release_year"]
        valid_years = years.notna() & years.between(1000, 9999) & (years.mod(1).abs() < 1e-9)
        result.loc[~valid_years, "release_year"] = float("nan")
    if "avg_rating" in result:
        valid_ratings = result["avg_rating"].between(0, 5)
        result.loc[~valid_ratings, "avg_rating"] = float("nan")
    if "rating_count" in result:
        result.loc[result["rating_count"] < 0, "rating_count"] = float("nan")
    if "runtime" in result:
        result.loc[result["runtime"] <= 0, "runtime"] = float("nan")
    return result


def _read_csv(path: Path, *, usecols: list[str] | None = None) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, usecols=usecols)
    except ValueError:
        # Some Kaggle exports have duplicate/extra columns; allow schema inspection.
        return pd.read_csv(path, low_memory=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError(f"Could not read {path.name}: {exc}") from exc


def _read_optional(directory: Path, filename: str) -> pd.DataFrame | None:
    path = directory / filename
    return _read_csv(path) if path.is_file() else None


def _json_feature_file(frame: pd.DataFrame | None, id_col_candidates: tuple[str, ...],
                       value_col_candidates: tuple[str, ...], output_key: str) -> dict[str, list[str]]:
    if frame is None:
        return {}
    id_col = _column(frame, *id_col_candidates)
    value_col = _column(frame, *value_col_candidates)
    if not id_col or not value_col:
        return {}
    output: dict[str, list[str]] = {}
    for _, row in frame[[id_col, value_col]].iterrows():
        item_id = normalize_external_id(row[id_col])
        if item_id:
            output[item_id] = parse_names(row[value_col], output_key)
    return output


def _directors_by_id(frame: pd.DataFrame | None) -> dict[str, str]:
    if frame is None:
        return {}
    id_col, crew_col = _column(frame, "id", "tmdbId"), _column(frame, "crew")
    if not id_col or not crew_col:
        return {}
    directors: dict[str, str] = {}
    for _, row in frame[[id_col, crew_col]].iterrows():
        ext_id = normalize_external_id(row[id_col])
        if not ext_id:
            continue
        try:
            crew = json.loads(row[crew_col]) if isinstance(row[crew_col], str) else row[crew_col]
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(crew, list):
            director = next((person.get("name", "") for person in crew
                             if isinstance(person, dict) and person.get("job") == "Director"), "")
            if director:
                directors[ext_id] = str(director)
    return directors


def prepare_data(raw_dir: str | Path, output_dir: str | Path) -> dict[str, int]:
    """Clean supported source files and write the processed catalog and ratings."""
    raw = Path(raw_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = raw / "movies_metadata.csv"
    movielens_path = raw / "movies.csv"
    if metadata_path.is_file():
        source = _read_csv(metadata_path)
        id_col = _column(source, "id", "tmdbId", "movieId")
        title_col = _column(source, "title", "original_title", "name")
        if not id_col or not title_col:
            raise DatasetError("movies_metadata.csv needs an id and title column.")
        source_id_type = "tmdb"
        raw_id_col = id_col
        title_name = title_col
    elif movielens_path.is_file():
        source = _read_csv(movielens_path)
        id_col = _column(source, "movieId", "movie_id", "id")
        title_col = _column(source, "title", "name")
        if not id_col or not title_col:
            raise DatasetError("movies.csv needs a movieId and title column.")
        source_id_type = "ml"
        raw_id_col = id_col
        title_name = title_col
    else:
        raise DatasetError(
            f"No movie metadata found in {raw}. Add movies_metadata.csv (The Movies Dataset) "
            "or movies.csv (MovieLens). See README.md for download instructions."
        )

    credits_frame = _read_optional(raw, "credits.csv")
    credits_by_id = _json_feature_file(credits_frame, ("id", "tmdbId"), ("cast",), "name")
    directors_by_id = _directors_by_id(credits_frame)
    keywords_by_id = _json_feature_file(_read_optional(raw, "keywords.csv"), ("id", "tmdbId"), ("keywords", "keyword"), "name")
    links = _read_optional(raw, "links.csv")
    link_to_tmdb: dict[str, str] = {}
    if links is not None:
        ml_col, tmdb_col = _column(links, "movieId", "movie_id"), _column(links, "tmdbId", "tmdb_id")
        if ml_col and tmdb_col:
            for _, row in links[[ml_col, tmdb_col]].iterrows():
                ml_id, tmdb_id = normalize_external_id(row[ml_col]), normalize_external_id(row[tmdb_col])
                if ml_id and tmdb_id:
                    link_to_tmdb[ml_id] = tmdb_id

    movies: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for _, row in source.iterrows():
        ext_id = normalize_external_id(row[raw_id_col])
        title = str(row[title_name]).strip() if pd.notna(row[title_name]) else ""
        if not ext_id or not title or title.lower() in {"nan", "none"}:
            continue
        tmdb_id = ext_id if source_id_type == "tmdb" else link_to_tmdb.get(ext_id)
        movie_id = f"tmdb:{tmdb_id}" if source_id_type == "tmdb" else f"ml:{ext_id}"
        if movie_id in seen_ids:
            continue
        seen_ids.add(movie_id)
        genres_col = _column(source, "genres", "genre")
        overview_col = _column(source, "overview", "plot", "description")
        year = None
        release_col = _column(source, "release_date", "releaseDate")
        if release_col and pd.notna(row[release_col]):
            parsed = pd.to_datetime(row[release_col], errors="coerce")
            if pd.notna(parsed):
                year = int(parsed.year)
        if year is None:
            year_match = re.search(r"\((\d{4})\)\s*$", title)
            if year_match:
                year = int(year_match.group(1))
        key_id = tmdb_id or ext_id
        cast_values = credits_by_id.get(key_id, [])
        director = directors_by_id.get(key_id, "")
        if not director and source_id_type == "tmdb":
            credit_col = _column(source, "credits")
            if credit_col:
                try:
                    embedded = json.loads(row[credit_col]) if isinstance(row[credit_col], str) else row[credit_col]
                    crew = embedded.get("crew", []) if isinstance(embedded, dict) else []
                    director = next((str(person.get("name", "")) for person in crew
                                     if isinstance(person, dict) and person.get("job") == "Director"), "")
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass
        poster_col = _column(source, "poster_path", "poster")
        poster_url_col = _column(source, "poster_url", "image_url")
        runtime_col = _column(source, "runtime")
        popularity_col = _column(source, "popularity")
        vote_col = _column(source, "vote_average", "avg_rating", "rating")
        count_col = _column(source, "vote_count", "rating_count")
        imdb_col = _column(source, "imdb_id", "imdbId")
        raw_vote = pd.to_numeric(row[vote_col], errors="coerce") if vote_col else None
        if raw_vote is not None and pd.notna(raw_vote):
            if str(vote_col).lower() == "vote_average" or float(raw_vote) > 5:
                raw_vote = float(raw_vote) / 2.0
        overview = str(row[overview_col]).strip() if overview_col and pd.notna(row[overview_col]) else ""
        movies.append({
            "movie_id": movie_id,
            "title": title,
            "release_year": year,
            "genres": "|".join(parse_names(row[genres_col])) if genres_col else "",
            "overview": overview if overview.lower() not in {"nan", "none"} else "",
            "keywords": "|".join(keywords_by_id.get(key_id, [])),
            "cast": "|".join(cast_values[:10]),
            "director": director,
            "avg_rating": float(raw_vote) if raw_vote is not None and pd.notna(raw_vote) else None,
            "rating_count": int(pd.to_numeric(row[count_col], errors="coerce")) if count_col and pd.notna(pd.to_numeric(row[count_col], errors="coerce")) else 0,
            "runtime": float(pd.to_numeric(row[runtime_col], errors="coerce")) if runtime_col and pd.notna(pd.to_numeric(row[runtime_col], errors="coerce")) else None,
            "popularity": float(pd.to_numeric(row[popularity_col], errors="coerce")) if popularity_col and pd.notna(pd.to_numeric(row[popularity_col], errors="coerce")) else 0.0,
            "poster_path": str(row[poster_col]).strip() if poster_col and pd.notna(row[poster_col]) else "",
            "poster_url": str(row[poster_url_col]).strip() if poster_url_col and pd.notna(row[poster_url_col]) else "",
            "tmdb_id": tmdb_id or "",
            "imdb_id": str(row[imdb_col]).strip() if imdb_col and pd.notna(row[imdb_col]) else "",
        })
    if not movies:
        raise DatasetError("No usable movie rows were found. Check that IDs and titles are populated.")
    movie_frame = pd.DataFrame(movies, columns=MOVIE_COLUMNS)
    valid_ids = set(movie_frame["movie_id"])

    ratings_path = raw / "ratings.csv"
    ratings_path_out = out / "ratings_clean.csv"
    source_rating_rows = 0
    rating_row_count = 0
    user_ids: set[str] = set()
    rated_movie_ids: set[str] = set()
    unmapped_movie_ids: set[str] = set()
    mapped_rating_rows = 0
    unmapped_rating_rows = 0
    invalid_timestamp_rows = 0
    rejected_reasons: dict[str, int] = {}
    rating_sums: dict[str, float] = {}
    rating_counts: dict[str, int] = {}
    ratings_temp_path: Path | None = None
    movie_temp_path: Path | None = None
    if ratings_path.is_file():
        try:
            rating_header = pd.read_csv(ratings_path, nrows=0)
            user_col = _column(rating_header, "userId", "user_id", "user")
            rating_movie_col = _column(rating_header, "movieId", "movie_id", "tmdbId", "tmdb_id")
            rating_col = _column(rating_header, "rating", "score")
            time_col = _column(rating_header, "timestamp", "rated_at")
            required = {"user_id": user_col, "movie_id": rating_movie_col, "rating": rating_col}
            missing = [name for name, column in required.items() if column is None]
            if missing:
                raise DatasetError(f"ratings.csv is missing required columns: {', '.join(missing)}.")

            rating_ids_are_tmdb = str(rating_movie_col).strip().casefold() in {"tmdbid", "tmdb_id"}
            fd, temp_name = tempfile.mkstemp(prefix=".ratings_clean.", suffix=".tmp", dir=out)
            os.close(fd)
            ratings_temp_path = Path(temp_name)
            first_chunk = True
            read_dtypes = {user_col: "string", rating_movie_col: "string"}
            for rating_source in pd.read_csv(ratings_path, low_memory=False, chunksize=250_000,
                                             dtype=read_dtypes):
                source_rating_rows += len(rating_source)
                ext_ids = rating_source[rating_movie_col].map(normalize_external_id)
                users = rating_source[user_col].astype("string").str.strip()
                rating_values = pd.to_numeric(rating_source[rating_col], errors="coerce")

                if rating_ids_are_tmdb:
                    mapped_movies = ext_ids.map(
                        lambda value: f"tmdb:{value}" if value is not None else pd.NA)
                else:
                    mapped_movies = ext_ids.map(
                        lambda value: f"ml:{value}" if value is not None else pd.NA)
                    if source_id_type == "tmdb" and link_to_tmdb:
                        tmdb_ids = ext_ids.map(link_to_tmdb)
                        linked_movies = tmdb_ids.map(
                            lambda value: f"tmdb:{value}" if value is not None else pd.NA)
                        mapped_movies = linked_movies.fillna(mapped_movies)

                user_valid = users.notna() & users.ne("") & ~users.str.casefold().isin(
                    {"nan", "none", "null", "<na>"})
                movie_valid = ext_ids.notna()
                rating_valid = rating_values.notna() & rating_values.between(0.5, 5.0)
                reasons = pd.Series("", index=rating_source.index, dtype="string")
                reasons.loc[~user_valid] = "invalid_user_id"
                reasons.loc[user_valid & ~movie_valid] = "invalid_movie_id"
                reasons.loc[user_valid & movie_valid & ~rating_valid] = "invalid_rating"
                valid = reasons.eq("")
                for reason, count in reasons.loc[~valid].value_counts().items():
                    rejected_reasons[str(reason)] = rejected_reasons.get(str(reason), 0) + int(count)

                if time_col:
                    raw_timestamps = rating_source[time_col]
                    timestamps = pd.to_numeric(raw_timestamps, errors="coerce")
                    timestamp_valid = (timestamps.notna() & timestamps.between(-9e18, 9e18)
                                       & timestamps.mod(1).eq(0))
                    has_timestamp = raw_timestamps.notna() & raw_timestamps.astype("string").str.strip().ne("")
                    invalid_timestamp_rows += int((has_timestamp & ~timestamp_valid).sum())
                    timestamps = timestamps.where(timestamp_valid).astype("Int64")
                else:
                    timestamps = pd.Series(pd.NA, index=rating_source.index, dtype="Int64")

                clean = pd.DataFrame({"user_id": users, "movie_id": mapped_movies,
                                      "rating": rating_values, "timestamp": timestamps},
                                     index=rating_source.index).loc[valid].copy()
                if clean.empty:
                    continue
                clean["user_id"] = clean["user_id"].astype("string")
                clean["movie_id"] = clean["movie_id"].astype("string")
                clean["rating"] = clean["rating"].astype("float32")
                rating_row_count += len(clean)
                user_ids.update(clean["user_id"].unique().tolist())
                rated_movie_ids.update(clean["movie_id"].unique().tolist())
                catalog_match = clean["movie_id"].isin(valid_ids)
                mapped_rating_rows += int(catalog_match.sum())
                unmapped_rating_rows += int((~catalog_match).sum())
                unmapped_movie_ids.update(clean.loc[~catalog_match, "movie_id"].unique().tolist())
                if catalog_match.any():
                    by_movie = clean.loc[catalog_match].groupby("movie_id")["rating"].agg(["sum", "count"])
                    for movie_id, stat in by_movie.iterrows():
                        rating_sums[movie_id] = rating_sums.get(movie_id, 0.0) + float(stat["sum"])
                        rating_counts[movie_id] = rating_counts.get(movie_id, 0) + int(stat["count"])
                clean.to_csv(ratings_temp_path, index=False,
                             mode="w" if first_chunk else "a", header=first_chunk)
                first_chunk = False

            if source_rating_rows == 0:
                raise DatasetError("ratings.csv contains no data rows; the existing cleaned ratings file was left unchanged.")
            if rating_row_count == 0:
                reason_summary = ", ".join(f"{name}: {count:,}" for name, count in sorted(rejected_reasons.items()))
                raise DatasetError(
                    f"All {source_rating_rows:,} rows in ratings.csv were rejected ({reason_summary or 'no valid rows'}); "
                    "the existing cleaned ratings file was left unchanged."
                )

            if rating_counts:
                rating_stats = pd.DataFrame({"mean": {mid: rating_sums[mid] / rating_counts[mid] for mid in rating_counts},
                                             "count": rating_counts})
                movie_frame = movie_frame.set_index("movie_id")
                movie_frame.loc[rating_stats.index, "avg_rating"] = rating_stats["mean"]
                movie_frame.loc[rating_stats.index, "rating_count"] = rating_stats["count"]
                movie_frame = movie_frame.reset_index()
        except DatasetError:
            if ratings_temp_path is not None:
                ratings_temp_path.unlink(missing_ok=True)
                ratings_temp_path = None
            raise
        except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            if ratings_temp_path is not None:
                ratings_temp_path.unlink(missing_ok=True)
                ratings_temp_path = None
            raise DatasetError(f"Could not process ratings.csv: {exc}") from exc
    else:
        if not ratings_path_out.is_file():
            fd, temp_name = tempfile.mkstemp(prefix=".ratings_clean.", suffix=".tmp", dir=out)
            os.close(fd)
            ratings_temp_path = Path(temp_name)
            pd.DataFrame(columns=RATING_COLUMNS).to_csv(ratings_temp_path, index=False)

    try:
        fd, temp_name = tempfile.mkstemp(prefix=".movies_clean.", suffix=".tmp", dir=out)
        os.close(fd)
        movie_temp_path = Path(temp_name)
        movie_frame.to_csv(movie_temp_path, index=False)
        os.replace(movie_temp_path, out / "movies_clean.csv")
        movie_temp_path = None
        if ratings_temp_path is not None:
            os.replace(ratings_temp_path, ratings_path_out)
            ratings_temp_path = None
    finally:
        if ratings_temp_path is not None:
            ratings_temp_path.unlink(missing_ok=True)
        if movie_temp_path is not None:
            movie_temp_path.unlink(missing_ok=True)

    ratings_output_size = ratings_path_out.stat().st_size if ratings_path_out.exists() else 0
    return {
        "movies": len(movie_frame), "ratings": rating_row_count,
        "source_rating_rows": source_rating_rows, "valid_ratings": rating_row_count,
        "rejected_ratings": sum(rejected_reasons.values()),
        "rejected_reasons": rejected_reasons,
        "users": len(user_ids), "unique_rated_movies": len(rated_movie_ids),
        "mapped_ratings": mapped_rating_rows, "unmapped_ratings": unmapped_rating_rows,
        "unmapped_movies": len(unmapped_movie_ids),
        "mapping_available": bool(link_to_tmdb), "mapping_rows": len(link_to_tmdb),
        "invalid_timestamps": invalid_timestamp_rows,
        "ratings_output_path": str(ratings_path_out),
        "ratings_output_size_bytes": ratings_output_size,
        "ratings_output_size_mb": round(ratings_output_size / (1024 * 1024), 2),
        "excluded_movie_rows": max(0, len(source) - len(movies)),
        "missing_overviews": int(movie_frame["overview"].fillna("").eq("").sum()),
    }


def load_catalog(processed_dir: str | Path) -> pd.DataFrame:
    path = Path(processed_dir) / "movies_clean.csv"
    if not path.is_file():
        raise DatasetError(f"Prepared movie catalog not found: {path}. Run scripts/prepare_data.py.")
    frame = pd.read_csv(path, low_memory=False)
    missing = [column for column in MOVIE_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetError(f"Processed movie catalog is missing columns: {', '.join(missing)}. Re-run preparation.")
    return normalize_catalog(frame)


def load_ratings(processed_dir: str | Path, movie_ids: set[str] | None = None) -> pd.DataFrame:
    path = Path(processed_dir) / "ratings_clean.csv"
    if not path.is_file():
        return pd.DataFrame(columns=RATING_COLUMNS)
    dtype = {"user_id": "string", "movie_id": "string", "rating": "float32"}
    if movie_ids is None:
        frame = pd.read_csv(path, low_memory=False, dtype=dtype)
        source_rows = len(frame)
        unmatched_rows = 0
    else:
        wanted = {str(movie_id) for movie_id in movie_ids}
        frames = []
        source_rows = 0
        matched_rows = 0
        for chunk in pd.read_csv(path, low_memory=False, dtype=dtype, chunksize=250_000):
            source_rows += len(chunk)
            if wanted:
                match = chunk["movie_id"].isin(wanted)
                matched_rows += int(match.sum())
                if match.any():
                    frames.append(chunk.loc[match])
        unmatched_rows = source_rows - matched_rows
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RATING_COLUMNS)
        frame.attrs["source_rows"] = source_rows
        frame.attrs["unmatched_rows"] = unmatched_rows
    for column in RATING_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[RATING_COLUMNS].dropna(subset=["user_id", "movie_id", "rating"])
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame = frame.dropna(subset=["rating"])
    frame.attrs.setdefault("source_rows", source_rows)
    frame.attrs.setdefault("unmatched_rows", unmatched_rows)
    return frame

