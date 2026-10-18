"""Add missing local posters from Kaggle's IMDb-keyed poster archive."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path, PurePosixPath
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import BadZipFile, ZipFile

from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "data" / "raw" / "kaggle_movie_posters.zip"
CATALOG = ROOT / "data" / "processed" / "movies_clean.csv"
POSTER_DIR = ROOT / "assets" / "posters"
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/raman77768/movie-classifier"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
IMDB_ID = re.compile(r"^tt\d{7,10}$", re.IGNORECASE)


def _has_value(value: str | None) -> bool:
    return bool(value and value.strip().casefold() not in {"", "nan", "none", "null", "<na>"})


def _download_archive(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    print("Downloading the Kaggle poster archive (about 260 MB)...")
    downloaded = 0
    try:
        with urlopen(DOWNLOAD_URL, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                downloaded += len(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"Downloaded {downloaded / (1024 * 1024):.1f} MB.")


def _catalog_rows() -> tuple[list[str], list[dict[str, str]]]:
    with CATALOG.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        rows = list(reader)
    imdb_column = next((name for name in fields if name.casefold() in {"imdb_id", "imdbid"}), None)
    if not imdb_column or not {"poster_path", "poster_url"}.issubset(fields):
        raise ValueError("The catalog must contain imdb_id, poster_path, and poster_url columns.")
    return fields, rows


def _write_catalog(fields: list[str], rows: list[dict[str, str]]) -> None:
    temporary = CATALOG.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, CATALOG)


def merge(archive: Path) -> tuple[int, int]:
    fields, rows = _catalog_rows()
    imdb_column = next(name for name in fields if name.casefold() in {"imdb_id", "imdbid"})
    missing_ids = {
        row.get(imdb_column, "").strip()
        for row in rows
        if not _has_value(row.get("poster_path"))
        and not _has_value(row.get("poster_url"))
        and IMDB_ID.fullmatch(row.get(imdb_column, "").strip())
    }
    if not missing_ids:
        print("The catalog has no missing poster sources.")
        return 0, 0

    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    try:
        with ZipFile(archive) as bundle:
            entries: dict[str, str] = {}
            for name in bundle.namelist():
                path = PurePosixPath(name)
                imdb_id = path.stem
                if path.suffix.lower() in IMAGE_SUFFIXES and imdb_id in missing_ids and IMDB_ID.fullmatch(imdb_id):
                    entries.setdefault(imdb_id, name)

            for index, (imdb_id, member) in enumerate(entries.items(), start=1):
                target = POSTER_DIR / f"{imdb_id}.webp"
                try:
                    with bundle.open(member) as source, Image.open(source) as image:
                        poster = ImageOps.exif_transpose(image).convert("RGB")
                        poster.thumbnail((300, 450), Image.Resampling.LANCZOS)
                        poster.save(target, format="WEBP", quality=78, method=4)
                    saved[imdb_id] = target.relative_to(ROOT).as_posix()
                except (OSError, ValueError, UnidentifiedImageError) as error:
                    print(f"Skipped {imdb_id}: {error}", file=sys.stderr)
                if index % 500 == 0:
                    print(f"Processed {index} matching poster files...")
    except BadZipFile as error:
        raise ValueError(f"The poster archive is not a valid ZIP file: {archive}") from error

    changed = 0
    for row in rows:
        imdb_id = row.get(imdb_column, "").strip()
        if imdb_id in saved and not _has_value(row.get("poster_path")) and not _has_value(row.get("poster_url")):
            row["poster_path"] = saved[imdb_id]
            changed += 1

    if changed:
        _write_catalog(fields, rows)
    size_mb = sum((ROOT / path).stat().st_size for path in saved.values()) / (1024 * 1024)
    return changed, round(size_mb)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge missing Kaggle movie posters into CineMatch.")
    parser.add_argument("--archive", type=Path,
                        help="Use a local Kaggle ZIP instead of downloading it automatically.")
    parser.add_argument("--refresh", action="store_true",
                        help="Download the archive again and check for newly matchable movies.")
    args = parser.parse_args()
    archive = args.archive.expanduser().resolve() if args.archive else DEFAULT_ARCHIVE
    if not archive.is_file() and not args.refresh and any(POSTER_DIR.glob("*.webp")):
        print("Kaggle posters are already merged. Use --refresh to check an updated catalog.")
        return 0

    downloaded_here = not archive.is_file() and args.archive is None
    if not archive.is_file():
        try:
            _download_archive(archive)
        except (OSError, URLError) as error:
            print(f"Could not download the Kaggle archive: {error}", file=sys.stderr)
            return 1

    try:
        count, size_mb = merge(archive)
    except (OSError, ValueError, BadZipFile) as error:
        print(error, file=sys.stderr)
        return 1
    if downloaded_here:
        archive.unlink(missing_ok=True)
    print(f"Added {count:,} local posters ({size_mb} MB) to {CATALOG.relative_to(ROOT)}.")
    print(f"Poster files are in {POSTER_DIR.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
