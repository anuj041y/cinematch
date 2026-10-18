"""Prepare downloaded CSVs into a normalized catalog and ratings table."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data_loader import DatasetError, prepare_data


def main() -> int:
    try:
        summary = prepare_data(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    except DatasetError as exc:
        print(f"Data preparation stopped: {exc}")
        print(f"Expected raw CSV files in: {RAW_DATA_DIR}")
        return 1
    print("CineMatch data preparation complete")
    for name, value in summary.items():
        if isinstance(value, dict):
            details = ", ".join(f"{reason}: {count:,}" for reason, count in sorted(value.items())) or "none"
            print(f"  {name.replace('_', ' ').title()}: {details}")
        elif isinstance(value, bool):
            print(f"  {name.replace('_', ' ').title()}: {'Yes' if value else 'No'}")
        elif isinstance(value, (int, float)):
            print(f"  {name.replace('_', ' ').title()}: {value:,}")
        else:
            print(f"  {name.replace('_', ' ').title()}: {value}")
    print(f"  Processed output: {PROCESSED_DATA_DIR}")
    if summary["valid_ratings"] and not summary["mapped_ratings"] and not summary["mapping_available"]:
        print("  Note: ratings were saved with MovieLens IDs. Add the matching links.csv to connect them to TMDB movies.")
    elif summary["unmapped_ratings"]:
        print("  Note: unmatched rating IDs were saved but cannot be used for recommendations until mapped to catalog movies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

