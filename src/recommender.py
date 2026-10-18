"""Movie recommendation models and ranking functions."""

from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
import pandas as pd


def _split(value: Any) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def _text_for_movie(row: pd.Series) -> str:
    fields = ("overview", "genres", "keywords", "cast", "director")
    parts: list[str] = []
    for field in fields:
        value = row.get(field, "")
        if value is not None and not pd.isna(value) and str(value).strip():
            # Give genres, keywords, and director extra weight in text similarity.
            parts.extend([str(value).replace("|", " ")] * (2 if field in {"genres", "keywords", "director"} else 1))
    return " ".join(parts)


def fit_content_model(movies: pd.DataFrame) -> dict[str, Any]:
    """Build content features from movie metadata and genres."""
    from scipy.sparse import hstack, csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import MultiLabelBinarizer, normalize

    frame = movies.reset_index(drop=True).copy()
    if frame.empty:
        raise ValueError("Cannot fit a content model without movies.")
    documents = [_text_for_movie(row) for _, row in frame.iterrows()]
    if not any(doc.strip() for doc in documents):
        documents = ["unknown metadata"] * len(frame)
    vectorizer = TfidfVectorizer(stop_words="english", max_features=120_000, ngram_range=(1, 2), min_df=1,
                                 dtype=np.float32)
    try:
        text_features = vectorizer.fit_transform(documents)
    except ValueError:
        # Retry without English stop words for small or sparse catalogs.
        vectorizer = TfidfVectorizer(max_features=120_000, ngram_range=(1, 2), min_df=1, dtype=np.float32)
        try:
            text_features = vectorizer.fit_transform(documents)
        except ValueError:
            documents = ["unknown metadata"] * len(frame)
            vectorizer = TfidfVectorizer(max_features=120_000, ngram_range=(1, 2), min_df=1, dtype=np.float32)
            text_features = vectorizer.fit_transform(documents)
    genre_lists = [_split(value) for value in frame.get("genres", pd.Series([""] * len(frame)))]
    binarizer = MultiLabelBinarizer(sparse_output=True)
    genre_features = binarizer.fit_transform(genre_lists).astype(np.float32)
    if genre_features.shape[1]:
        feature_matrix = hstack([text_features, genre_features * 1.5], format="csr", dtype=np.float32)
    else:
        feature_matrix = csr_matrix(text_features, dtype=np.float32)
    feature_matrix = normalize(feature_matrix, norm="l2", copy=False).tocsr()
    return {
        "kind": "content_tfidf_genre_v1",
        "movie_ids": frame["movie_id"].astype(str).tolist(),
        "movie_to_index": {str(movie_id): index for index, movie_id in enumerate(frame["movie_id"])},
        "features": feature_matrix,
        "vectorizer": vectorizer,
        "genre_binarizer": binarizer,
    }


def fit_collaborative_model(ratings: pd.DataFrame) -> dict[str, Any]:
    """Build item similarities from user ratings centered around each user's mean."""
    from scipy.sparse import csr_matrix
    from sklearn.preprocessing import normalize

    needed = {"user_id", "movie_id", "rating"}
    if not needed.issubset(ratings.columns) or ratings.empty:
        return {"kind": "item_cf_centered_v1", "movie_ids": [], "movie_to_index": {},
                "item_vectors": csr_matrix((0, 0), dtype=np.float32),
                "item_presence": csr_matrix((0, 0), dtype=np.float32),
                "global_mean": 3.0, "ratings_count": 0}
    frame = ratings[["user_id", "movie_id", "rating"]].copy()
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce").astype("float32")
    frame = frame.dropna(subset=["user_id", "movie_id", "rating"])
    if frame.duplicated(["user_id", "movie_id"]).any():
        frame = frame.groupby(["user_id", "movie_id"], as_index=False, observed=True)["rating"].mean()
    user_codes, user_values = pd.factorize(frame["user_id"], sort=True)
    movie_codes, movie_values = pd.factorize(frame["movie_id"], sort=True)
    user_positions = user_codes.astype(np.int32, copy=False)
    movie_positions = movie_codes.astype(np.int32, copy=False)
    del user_codes, movie_codes
    movie_ids = [str(value) for value in movie_values]
    user_ids = [str(value) for value in user_values]
    movie_to_index = {value: index for index, value in enumerate(movie_ids)}
    values = frame["rating"].to_numpy(dtype=np.float32, copy=False)
    user_sums = np.bincount(user_positions, weights=values, minlength=len(user_ids))
    user_counts = np.bincount(user_positions, minlength=len(user_ids))
    user_means = (user_sums / np.maximum(user_counts, 1)).astype(np.float32)
    centered_values = values - user_means[user_positions]
    user_item_centered = csr_matrix((centered_values, (user_positions, movie_positions)),
                                    shape=(len(user_ids), len(movie_ids)), dtype=np.float32)
    item_vectors = normalize(user_item_centered.T.tocsr(), norm="l2", axis=1, copy=False).tocsr()
    item_presence = csr_matrix((np.ones(len(frame), dtype=np.float32), (movie_positions, user_positions)),
                               shape=(len(movie_ids), len(user_ids)), dtype=np.float32)
    return {
        "kind": "item_cf_centered_v1",
        "movie_ids": movie_ids,
        "movie_to_index": movie_to_index,
        "item_vectors": item_vectors,
        "item_presence": item_presence,
        "global_mean": float(frame["rating"].mean()),
        "ratings_count": int(len(frame)),
        "users_count": int(len(user_ids)),
    }


def _sparse_column_scores(matrix: Any, row_index: int) -> np.ndarray:
    result = matrix @ matrix.getrow(row_index).T
    return np.asarray(result.toarray()).ravel().astype(float)


def _content_scores(model: dict[str, Any] | None, seed_id: str | None,
                    profile_ratings: dict[str, float] | None = None) -> dict[str, float]:
    if not model or not model.get("movie_ids"):
        return {}
    features = model["features"]
    mapping = model["movie_to_index"]
    if seed_id and seed_id in mapping:
        row_index = mapping[seed_id]
        return dict(zip(model["movie_ids"], _sparse_column_scores(features, row_index)))
    profile = profile_ratings or {}
    liked = [(mapping[mid], max(0.0, float(rating) - 3.0)) for mid, rating in profile.items()
             if mid in mapping and float(rating) > 3.0]
    if not liked:
        return {}
    query = None
    for row_index, weight in liked:
        contribution = features.getrow(row_index) * weight
        query = contribution if query is None else query + contribution
    from sklearn.preprocessing import normalize
    query = normalize(query, norm="l2", copy=False)
    result = features @ query.T
    scores = np.asarray(result.toarray()).ravel().astype(float)
    return dict(zip(model["movie_ids"], scores))


def _collaborative_scores(model: dict[str, Any] | None, profile_ratings: dict[str, float] | None,
                          seed_id: str | None = None) -> tuple[dict[str, float], dict[str, tuple[str, int]]]:
    if not model or not model.get("movie_ids"):
        return {}, {}
    mapping = model["movie_to_index"]
    profile = dict(profile_ratings or {})
    if seed_id and seed_id in mapping and seed_id not in profile:
        profile[seed_id] = 5.0
    eligible = [(mapping[mid], float(rating)) for mid, rating in profile.items() if mid in mapping]
    global_mean = float(model.get("global_mean", 3.0))
    # Use only the 50 ratings farthest from the dataset mean.
    seeds = sorted(eligible, key=lambda item: abs(item[1] - global_mean), reverse=True)[:50]
    if not seeds:
        return {}, {}
    vectors = model["item_vectors"]
    presence = model["item_presence"]
    total = np.zeros(len(model["movie_ids"]), dtype=float)
    weights = np.zeros_like(total)
    best_seed = np.zeros_like(total)
    explanations: dict[str, tuple[str, int]] = {}
    for seed_index, rating in seeds:
        sims = _sparse_column_scores(vectors, seed_index)
        # Negative correlations count against a candidate; zero correlations add nothing.
        preference = rating - global_mean
        total += sims * preference
        weights += np.abs(sims)
        seed_title = model["movie_ids"][seed_index]
        binary = presence @ presence.getrow(seed_index).T
        overlap = np.asarray(binary.toarray()).ravel().astype(int)
        for candidate_index, similarity in enumerate(sims):
            key = model["movie_ids"][candidate_index]
            if key == seed_title:
                continue
            candidate_influence = abs(similarity * preference)
            existing = best_seed[candidate_index]
            if candidate_influence > existing:
                best_seed[candidate_index] = candidate_influence
                explanations[key] = (seed_title, int(overlap[candidate_index]))
    predictions = np.full_like(total, global_mean)
    active = weights > 0
    predictions[active] = global_mean + total[active] / weights[active]
    predictions = np.clip(predictions, 1.0, 5.0)
    scores = (predictions - 1.0) / 4.0
    return dict(zip(model["movie_ids"], scores)), explanations


def _popularity_scores(movies: pd.DataFrame) -> dict[str, float]:
    count = pd.to_numeric(movies["rating_count"], errors="coerce").fillna(0) if "rating_count" in movies else pd.Series(np.zeros(len(movies)), index=movies.index)
    rating = pd.to_numeric(movies["avg_rating"], errors="coerce").fillna(0) if "avg_rating" in movies else pd.Series(np.zeros(len(movies)), index=movies.index)
    popularity = pd.to_numeric(movies["popularity"], errors="coerce").fillna(0) if "popularity" in movies else pd.Series(np.zeros(len(movies)), index=movies.index)
    valid = rating[rating > 0]
    mean = float(valid.mean()) if len(valid) else 3.0
    minimum = float(count[count > 0].quantile(0.60)) if (count > 0).any() else 1.0
    minimum = max(minimum, 1.0)
    bayes = ((count / (count + minimum)) * rating + (minimum / (count + minimum)) * mean) / 5.0
    pop_max = float(popularity.max()) if len(popularity) else 0.0
    pop_norm = popularity / pop_max if pop_max > 0 else popularity * 0
    score = 0.85 * bayes.clip(0, 1) + 0.15 * pop_norm.clip(0, 1)
    return dict(zip(movies["movie_id"].astype(str), score.astype(float)))


def _reason(movie: pd.Series, seed_movie: pd.Series | None, mode: str,
            collab_explanation: tuple[str, int] | None = None,
            content_signal: float = 0.0, collab_signal: float = 0.0) -> str:
    if mode == "popular cold-start":
        return "Popular in the supplied movie and rating dataset."
    if mode == "content" or (content_signal > 0 and collab_signal <= 0):
        if seed_movie is not None:
            common = sorted(set(_split(movie.get("genres", ""))) & set(_split(seed_movie.get("genres", ""))))
            if common:
                return "Shares genres with " + str(seed_movie.get("title", "your selection")) + ": " + ", ".join(common[:3]) + "."
            return "Similar plot and metadata to " + str(seed_movie.get("title", "your selection")) + "."
        return "Matches the plot and metadata of movies you rated above 3/5."
    if collab_explanation and collab_explanation[1] > 0:
        seed_id, overlap = collab_explanation
        return f"Co-rated by {overlap:,} users in the supplied dataset with {seed_id}."
    if mode == "collaborative":
        return "Supported by item rating patterns across dataset users."
    if content_signal > 0 and collab_signal > 0:
        return "Combines your content matches with collaborative rating patterns."
    if content_signal > 0:
        return "Selected from content similarity; collaborative data was unavailable for this title."
    if collab_signal > 0:
        return "Selected from collaborative rating patterns; metadata was unavailable for this title."
    return "Popular in the supplied movie and rating dataset."


def _diverse_rank(rows: list[dict[str, Any]], movies_by_id: dict[str, pd.Series], penalty: float = 0.12) -> list[dict[str, Any]]:
    remaining = rows[:]
    selected: list[dict[str, Any]] = []
    genre_counts: dict[str, int] = {}
    while remaining:
        best_position, best_value = 0, float("-inf")
        for position, candidate in enumerate(remaining):
            genres = _split(movies_by_id[candidate["movie_id"]].get("genres", ""))
            repetition = sum(genre_counts.get(genre, 0) for genre in genres) / max(1, len(genres))
            adjusted = candidate["score"] - penalty * repetition
            if adjusted > best_value:
                best_position, best_value = position, adjusted
        chosen = remaining.pop(best_position)
        selected.append(chosen)
        for genre in _split(movies_by_id[chosen["movie_id"]].get("genres", "")):
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
    return selected


def recommend(movies: pd.DataFrame, content_model: dict[str, Any] | None,
              collaborative_model: dict[str, Any] | None, *,
              mode: str = "hybrid", profile_ratings: dict[str, float] | None = None,
              seed_movie_id: str | None = None, excluded_ids: Iterable[str] = (),
              alpha: float = 0.5, limit: int = 20, diversity_penalty: float = 0.12) -> list[dict[str, Any]]:
    """Rank movies and include the score and reason for each suggestion."""
    if mode not in {"content", "collaborative", "hybrid"}:
        raise ValueError("mode must be content, collaborative, or hybrid")
    alpha = float(np.clip(alpha, 0.0, 1.0))
    frame = movies.drop_duplicates("movie_id").copy()
    by_id = {str(row["movie_id"]): row for _, row in frame.iterrows()}
    excluded = {str(value) for value in excluded_ids}
    if seed_movie_id:
        excluded.add(str(seed_movie_id))
    profile = {str(key): float(value) for key, value in (profile_ratings or {}).items()}
    excluded.update(profile.keys())
    content = _content_scores(content_model, seed_movie_id, profile)
    collab, collab_explanations = _collaborative_scores(collaborative_model, profile, seed_movie_id)
    popularity = _popularity_scores(frame)
    content_active = any(value > 1e-9 for value in content.values())
    collab_active = any(value > 1e-9 for value in collab.values())
    if mode == "content":
        mode_label, selected_scores = ("content", content) if content_active else ("popular cold-start", popularity)
    elif mode == "collaborative":
        mode_label, selected_scores = ("collaborative", collab) if collab_active else ("popular cold-start", popularity)
    elif content_active and collab_active:
        mode_label = "hybrid"
        selected_scores = {}
        for movie_id in set(content) | set(collab):
            content_score = content.get(movie_id)
            collab_score = collab.get(movie_id)
            if content_score is None:
                selected_scores[movie_id] = float(collab_score)
            elif collab_score is None:
                selected_scores[movie_id] = float(content_score)
            else:
                selected_scores[movie_id] = alpha * content_score + (1.0 - alpha) * collab_score
    elif content_active:
        mode_label, selected_scores = "content fallback", content
    elif collab_active:
        mode_label, selected_scores = "collaborative fallback", collab
    else:
        mode_label, selected_scores = "popular cold-start", popularity
    rows: list[dict[str, Any]] = []
    for movie_id, score in selected_scores.items():
        if movie_id in excluded or movie_id not in by_id or not np.isfinite(score):
            continue
        movie = by_id[movie_id]
        seed_row = by_id.get(str(seed_movie_id)) if seed_movie_id else None
        content_value = float(content.get(movie_id, 0.0))
        collab_value = float(collab.get(movie_id, 0.0))
        explanation = collab_explanations.get(movie_id)
        if explanation and explanation[0] in by_id:
            explanation = (str(by_id[explanation[0]].get("title", explanation[0])), explanation[1])
        rows.append({
            "movie_id": movie_id, "title": str(movie.get("title", "")),
            "score": float(np.clip(score, 0, 1)),
            "content_score": content_value, "collaborative_score": collab_value,
            "predicted_rating": 1.0 + 4.0 * collab_value if movie_id in collab else None,
            "method": mode_label,
            "reason": _reason(movie, seed_row, mode_label, explanation, content_value, collab_value),
        })
    rows.sort(key=lambda item: item["score"], reverse=True)
    requested = max(0, int(limit))
    if requested == 0:
        return []
    # Rerank a limited pool to keep large catalogs responsive.
    pool_size = min(len(rows), max(requested * 6, 60))
    ranked = _diverse_rank(rows[:pool_size], by_id, diversity_penalty)
    return ranked[:requested]

