import sys
import tempfile
import unittest
from pathlib import Path

from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_artifacts import artifact_matches_data, load_model_artifact, movie_ids_fingerprint


def empty_rating_artifact():
    ids = ["tmdb:1", "tmdb:2"]
    return ids, {
        "content": {
            "kind": "content_tfidf_genre_v1",
            "movie_ids": ids,
            "movie_to_index": {movie_id: index for index, movie_id in enumerate(ids)},
            "features": csr_matrix((2, 3)),
        },
        "collaborative": {
            "kind": "item_cf_centered_v1",
            "movie_ids": [],
            "movie_to_index": {},
            "item_vectors": csr_matrix((0, 0)),
            "ratings_count": 0,
        },
        "ratings_count": 0,
    }


class ModelArtifactTests(unittest.TestCase):
    def test_artifact_must_match_current_ids_and_rating_count(self):
        ids, artifact = empty_rating_artifact()
        fingerprint = movie_ids_fingerprint(ids)
        self.assertTrue(artifact_matches_data(artifact, fingerprint, 0))
        self.assertFalse(artifact_matches_data(artifact, movie_ids_fingerprint(["tmdb:1", "tmdb:3"]), 0))
        self.assertFalse(artifact_matches_data(artifact, fingerprint, 5))

    def test_valid_artifact_loads_and_invalid_file_returns_diagnostic(self):
        ids, artifact = empty_rating_artifact()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "models.joblib"
            import joblib
            joblib.dump(artifact, path)
            loaded, error = load_model_artifact(path, movie_ids_fingerprint(ids), 0)
            self.assertIsNotNone(loaded)
            self.assertIsNone(error)
            path.write_bytes(b"not a joblib artifact")
            with self.assertLogs(level="ERROR"):
                loaded, error = load_model_artifact(path, movie_ids_fingerprint(ids), 0)
            self.assertIsNone(loaded)
            self.assertIn("Could not load saved models", error)


if __name__ == "__main__":
    unittest.main()
