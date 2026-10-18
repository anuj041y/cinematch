"""Train and persist CineMatch content and collaborative model artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import MODEL_DIR, MODEL_PATH, PROCESSED_DATA_DIR
from src.data_loader import DatasetError, load_catalog, load_ratings
from src.evaluation import evaluate_item_cf
from src.recommender import fit_collaborative_model, fit_content_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-evaluation", action="store_true", help="Skip the sampled user-aware holdout evaluation.")
    parser.add_argument("--evaluation-users", type=int, default=300, help="Maximum sampled users for evaluation (default: 300).")
    args = parser.parse_args()
    try:
        movies = load_catalog(PROCESSED_DATA_DIR)
        ratings = load_ratings(PROCESSED_DATA_DIR, set(movies["movie_id"].astype(str)))
    except DatasetError as exc:
        print(f"Training stopped: {exc}")
        print("Run python scripts/prepare_data.py after placing real dataset CSV files in data/raw/.")
        return 1
    if movies.empty:
        print("Training stopped: the processed movie catalog is empty.")
        return 1
    source_rating_rows = int(ratings.attrs.get("source_rows", len(ratings)))
    unmapped_rating_rows = int(ratings.attrs.get("unmatched_rows", 0))
    print(f"Ratings in cleaned file: {source_rating_rows:,}; matched to catalog: {len(ratings):,}; "
          f"unmatched movie IDs: {unmapped_rating_rows:,}.")
    print(f"Matched users: {ratings['user_id'].nunique():,}; matched movies: {ratings['movie_id'].nunique():,}.")
    if unmapped_rating_rows:
        print("Unmatched ratings were not used. Check that ratings.csv movie IDs have a valid links.csv mapping to the catalog.")
    print(f"Fitting content features for {len(movies):,} movies…")
    content = fit_content_model(movies)
    print(f"Fitting item-based collaborative features from {len(ratings):,} catalog-matched ratings…")
    collaborative = fit_collaborative_model(ratings)
    evaluation = None
    if not args.skip_evaluation and not ratings.empty:
        print("Running seeded, per-user holdout evaluation on a user sample…")
        evaluation = evaluate_item_cf(ratings, max_users=max(1, args.evaluation_users))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        import joblib
    except ImportError as exc:
        print("Install requirements before training: joblib is missing.")
        return 1
    artifact = {"content": content, "collaborative": collaborative,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "movies_count": len(movies), "ratings_count": len(ratings)}
    joblib.dump(artifact, MODEL_PATH, compress=3)
    metadata = {
        "trained_at": artifact["trained_at"],
        "movies": len(movies),
        "ratings": len(ratings),
        "rating_distribution": {
            str(float(rating)): int(count)
            for rating, count in ratings["rating"].value_counts().sort_index().items()
        },
        "users": int(ratings["user_id"].nunique()) if not ratings.empty else 0,
        "rated_movies": int(ratings["movie_id"].nunique()) if not ratings.empty else 0,
        "ratings_in_file": source_rating_rows,
        "unmatched_ratings": unmapped_rating_rows,
        "content_model": "TF-IDF n-grams + multi-hot genres, row-normalized cosine similarity",
        "collaborative_model": "Centered item-based cosine similarity from observed MovieLens user ratings",
        "evaluation": evaluation,
        "evaluation_note": "Evaluation uses a sampled holdout of user ratings.",
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved model artifacts: {MODEL_PATH}")
    if evaluation:
        print("Evaluation summary:")
        for key, value in evaluation.items():
            print(f"  {key}: {value}")
    elif ratings.empty:
        print("No cleaned ratings match the catalog IDs; collaborative recommendations are unavailable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

