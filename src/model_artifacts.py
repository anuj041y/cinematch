"""Compatibility checks for persisted recommendation models."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Iterable


def movie_ids_fingerprint(movie_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for movie_id in movie_ids:
        value = str(movie_id).encode("utf-8")
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def artifact_matches_data(artifact: Any, expected_movie_ids_hash: str,
                          expected_ratings_count: int | None = None) -> bool:
    if not isinstance(artifact, dict):
        return False
    content = artifact.get("content")
    collaborative = artifact.get("collaborative")
    if not isinstance(content, dict) or not isinstance(collaborative, dict):
        return False
    if content.get("kind") != "content_tfidf_genre_v1" or collaborative.get("kind") != "item_cf_centered_v1":
        return False
    movie_ids = content.get("movie_ids")
    content_features = content.get("features")
    content_mapping = content.get("movie_to_index")
    if not isinstance(movie_ids, list) or not movie_ids or not isinstance(content_mapping, dict):
        return False
    if movie_ids_fingerprint(movie_ids) != expected_movie_ids_hash:
        return False
    if len(content_mapping) != len(movie_ids) or any(content_mapping.get(movie_id) != index
                                                     for index, movie_id in enumerate(movie_ids)):
        return False
    if getattr(content_features, "shape", (None,))[0] != len(movie_ids):
        return False
    try:
        saved_rating_count = int(artifact.get("ratings_count", -1))
        collaborative_rating_count = int(collaborative.get("ratings_count", -1))
    except (TypeError, ValueError, OverflowError):
        return False
    if saved_rating_count < 0 or (expected_ratings_count is not None
                                  and saved_rating_count != int(expected_ratings_count)):
        return False
    if saved_rating_count == 0:
        if collaborative_rating_count != 0:
            return False
    elif not 0 < collaborative_rating_count <= saved_rating_count:
        return False
    collab_ids = collaborative.get("movie_ids")
    collab_mapping = collaborative.get("movie_to_index")
    item_vectors = collaborative.get("item_vectors")
    if not isinstance(collab_ids, list) or not isinstance(collab_mapping, dict):
        return False
    if saved_rating_count == 0 and collab_ids:
        return False
    if len(collab_mapping) != len(collab_ids) or any(collab_mapping.get(movie_id) != index
                                                     for index, movie_id in enumerate(collab_ids)):
        return False
    if getattr(item_vectors, "shape", (None,))[0] != len(collab_ids):
        return False
    return True


def load_model_artifact(path: str | Path, expected_movie_ids_hash: str,
                        expected_ratings_count: int | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """Load an artifact that matches the catalog and, when supplied, the rating count."""
    try:
        import joblib
        artifact = joblib.load(path)
    except Exception as exc:
        logging.exception("Could not load model artifact at %s.", path)
        return None, f"Could not load saved models: {type(exc).__name__}."
    if not artifact_matches_data(artifact, expected_movie_ids_hash, expected_ratings_count):
        return None, "Saved models do not match the current catalog or ratings."
    return artifact, None
