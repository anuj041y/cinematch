"""Evaluate item-based recommendations with held-out user ratings."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.recommender import fit_collaborative_model, _collaborative_scores


def evaluate_item_cf(ratings: pd.DataFrame, *, seed: int = 42, max_users: int = 300,
                     holdout_fraction: float = 0.2, k: int = 10) -> dict[str, float | int | None]:
    """Evaluate latest ratings per sampled user where timestamps exist.

    Users are sampled with a fixed seed. Every selected user's held-out rows are
    removed before fitting; users with fewer than three observations are skipped.
    Popularity uses only the corresponding training rows.
    """
    columns = {"user_id", "movie_id", "rating"}
    if ratings.empty or not columns.issubset(ratings.columns):
        return {"users_evaluated": 0, "test_ratings": 0, "mae": None, "rmse": None,
                "recall_at_k": None, "popularity_recall_at_k": None}
    ordered_columns = ["user_id", "movie_id", "rating"] + (["timestamp"] if "timestamp" in ratings.columns else [])
    frame = ratings[ordered_columns].copy()
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
    frame = frame.dropna(subset=["user_id", "movie_id", "rating"])
    eligible = frame.groupby("user_id", observed=True).size()
    eligible_users = eligible[eligible >= 3].index.to_numpy()
    if len(eligible_users) == 0:
        return {"users_evaluated": 0, "test_ratings": 0, "mae": None, "rmse": None,
                "recall_at_k": None, "popularity_recall_at_k": None}
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible_users, size=min(max_users, len(eligible_users)), replace=False)
    selected = frame[frame["user_id"].isin(chosen)].copy()
    selected["user_id"] = selected["user_id"].astype(str)
    selected["movie_id"] = selected["movie_id"].astype(str)
    selected = selected.drop_duplicates(["user_id", "movie_id"], keep="last")
    train_parts, test_parts = [], []
    has_timestamp = "timestamp" in selected and selected["timestamp"].notna().any()
    for _, group in selected.groupby("user_id", sort=False):
        group = group.sort_values("timestamp", na_position="first") if has_timestamp else group.sample(frac=1, random_state=seed)
        n_test = max(1, int(math.ceil(len(group) * holdout_fraction)))
        if len(group) - n_test < 2:
            n_test = len(group) - 2
        if n_test < 1:
            continue
        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])
    if not train_parts or not test_parts:
        return {"users_evaluated": 0, "test_ratings": 0, "mae": None, "rmse": None,
                "recall_at_k": None, "popularity_recall_at_k": None}
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    model = fit_collaborative_model(train)
    train_seen = train.groupby("user_id")["movie_id"].apply(set).to_dict()
    test_by_user = {str(user): group for user, group in test.groupby("user_id")}
    popularity = train.groupby("movie_id").agg(mean=("rating", "mean"), count=("rating", "size"))
    prior = float(train["rating"].mean())
    shrinkage = max(1.0, float(popularity["count"].quantile(0.6)))
    popularity["score"] = ((popularity["count"] * popularity["mean"] + shrinkage * prior) /
                            (popularity["count"] + shrinkage))
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    model_recalls: list[float] = []
    baseline_recalls: list[float] = []
    evaluated_users = 0
    for user_id, user_test in test_by_user.items():
        profile = train[train["user_id"].eq(user_id)].set_index("movie_id")["rating"].to_dict()
        if not profile:
            continue
        scores, _ = _collaborative_scores(model, profile)
        known_candidates = [(movie_id, score) for movie_id, score in scores.items()
                            if movie_id not in train_seen.get(user_id, set())]
        test_lookup = user_test.set_index("movie_id")["rating"].to_dict()
        predicted = [(1.0 + 4.0 * score, float(test_lookup[movie_id])) for movie_id, score in known_candidates
                     if movie_id in test_lookup]
        for estimate, actual in predicted:
            absolute_errors.append(abs(estimate - actual))
            squared_errors.append((estimate - actual) ** 2)
        relevant = {movie_id for movie_id, rating in test_lookup.items() if float(rating) >= 4.0}
        if relevant:
            cf_top = {movie_id for movie_id, _ in sorted(known_candidates, key=lambda item: item[1], reverse=True)[:k]}
            base_candidates = [(movie_id, float(row["score"])) for movie_id, row in popularity.iterrows()
                               if movie_id not in train_seen.get(user_id, set())]
            base_top = {movie_id for movie_id, _ in sorted(base_candidates, key=lambda item: item[1], reverse=True)[:k]}
            model_recalls.append(len(cf_top & relevant) / len(relevant))
            baseline_recalls.append(len(base_top & relevant) / len(relevant))
        evaluated_users += 1
    mae = float(np.mean(absolute_errors)) if absolute_errors else None
    rmse = float(np.sqrt(np.mean(squared_errors))) if squared_errors else None
    return {
        "users_evaluated": evaluated_users,
        "test_ratings": len(test),
        "predicted_test_ratings": len(absolute_errors),
        "mae": mae,
        "rmse": rmse,
        "recall_at_k": float(np.mean(model_recalls)) if model_recalls else None,
        "popularity_recall_at_k": float(np.mean(baseline_recalls)) if baseline_recalls else None,
        "k": k,
        "seed": seed,
        "split": "per-user temporal latest holdout" if has_timestamp else "per-user seeded random holdout",
    }

