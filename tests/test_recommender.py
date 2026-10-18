import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import scipy  # noqa: F401
    import sklearn  # noqa: F401
    HAS_ML = True
except ImportError:
    HAS_ML = False

from src.recommender import fit_collaborative_model, fit_content_model, recommend


class ColdStartTests(unittest.TestCase):
    def test_cold_start_uses_labeled_popularity_fallback(self):
        movies = pd.DataFrame([
            {"movie_id": "m1", "title": "One", "avg_rating": 4.5, "rating_count": 10, "popularity": 3},
            {"movie_id": "m2", "title": "Two", "avg_rating": 4.0, "rating_count": 2, "popularity": 1},
        ])
        result = recommend(movies, None, None, mode="hybrid", limit=2)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(row["method"] == "popular cold-start" for row in result))

    @unittest.skipUnless(HAS_ML, "scikit-learn and SciPy are required for recommendation tests")
    def test_empty_real_ratings_keep_content_and_label_hybrid_fallback(self):
        movies = pd.DataFrame([
            {"movie_id": "m1", "title": "Space Rescue", "genres": "Sci-Fi|Action", "overview": "A crew rescues astronauts"},
            {"movie_id": "m2", "title": "Orbit Mission", "genres": "Sci-Fi|Action", "overview": "Astronauts save a distant crew"},
            {"movie_id": "m3", "title": "Garden Letters", "genres": "Romance", "overview": "Two friends exchange letters"},
        ])
        content = fit_content_model(movies)
        empty = fit_collaborative_model(pd.DataFrame(columns=["user_id", "movie_id", "rating"]))
        self.assertEqual(empty["ratings_count"], 0)
        content_only = recommend(movies, content, empty, mode="content", seed_movie_id="m1", limit=2)
        hybrid = recommend(movies, content, empty, mode="hybrid", seed_movie_id="m1", limit=2)
        self.assertEqual([item["movie_id"] for item in hybrid], [item["movie_id"] for item in content_only])
        self.assertTrue(all(item["method"] == "content fallback" for item in hybrid))
        self.assertEqual([item["score"] for item in hybrid], [item["score"] for item in content_only])

    @unittest.skipUnless(HAS_ML, "scikit-learn and SciPy are required for recommendation tests")
    def test_unmapped_movielens_ids_do_not_score_tmdb_movies(self):
        movies = pd.DataFrame([
            {"movie_id": "tmdb:42", "title": "Space Rescue", "genres": "Sci-Fi|Action", "overview": "A crew rescues astronauts"},
            {"movie_id": "tmdb:43", "title": "Orbit Mission", "genres": "Sci-Fi|Action", "overview": "Astronauts save a distant crew"},
            {"movie_id": "tmdb:44", "title": "Garden Letters", "genres": "Romance", "overview": "Two friends exchange letters"},
        ])
        ratings = pd.DataFrame([
            {"user_id": "u1", "movie_id": "ml:42", "rating": 5},
            {"user_id": "u1", "movie_id": "ml:43", "rating": 4},
            {"user_id": "u2", "movie_id": "ml:42", "rating": 4},
            {"user_id": "u2", "movie_id": "ml:44", "rating": 1},
        ])
        content = fit_content_model(movies)
        collaborative = fit_collaborative_model(ratings)
        result = recommend(movies, content, collaborative, mode="hybrid",
                           profile_ratings={"tmdb:42": 5}, limit=2)
        self.assertTrue(result)
        self.assertTrue(all(item["method"] == "content fallback" for item in result))
        self.assertTrue(all(item["movie_id"].startswith("tmdb:") for item in result))


@unittest.skipUnless(HAS_ML, "scikit-learn and SciPy are required for recommendation tests")
class RecommenderTests(unittest.TestCase):
    def setUp(self):
        self.movies = pd.DataFrame([
            {"movie_id": "m1", "title": "Orbit Rescue", "genres": "Action|Sci-Fi", "overview": "Astronauts rescue a crew in deep space", "keywords": "orbit|rescue", "cast": "Alex Star", "director": "A. Director", "avg_rating": 4.4, "rating_count": 30, "popularity": 9},
            {"movie_id": "m2", "title": "Beyond Orbit", "genres": "Action|Sci-Fi", "overview": "A crew launches into space to save astronauts", "keywords": "space|rescue", "cast": "Alex Star", "director": "A. Director", "avg_rating": 4.1, "rating_count": 20, "popularity": 7},
            {"movie_id": "m3", "title": "Summer Letters", "genres": "Romance", "overview": "Two people meet during a summer holiday", "keywords": "summer|love", "cast": "Jamie Sun", "director": "B. Director", "avg_rating": 4.0, "rating_count": 10, "popularity": 3},
            {"movie_id": "m4", "title": "Quiet Planet", "genres": "Drama|Sci-Fi", "overview": "A quiet colony faces a distant future", "keywords": "planet|future", "cast": "Lee Moon", "director": "C. Director", "avg_rating": 3.4, "rating_count": 5, "popularity": 1},
        ])
        self.ratings = pd.DataFrame([
            {"user_id": "u1", "movie_id": "m1", "rating": 5}, {"user_id": "u1", "movie_id": "m2", "rating": 4}, {"user_id": "u1", "movie_id": "m3", "rating": 1},
            {"user_id": "u2", "movie_id": "m1", "rating": 5}, {"user_id": "u2", "movie_id": "m2", "rating": 5}, {"user_id": "u2", "movie_id": "m4", "rating": 2},
            {"user_id": "u3", "movie_id": "m1", "rating": 4}, {"user_id": "u3", "movie_id": "m2", "rating": 4}, {"user_id": "u3", "movie_id": "m3", "rating": 2},
            {"user_id": "u4", "movie_id": "m3", "rating": 5}, {"user_id": "u4", "movie_id": "m4", "rating": 2},
        ])
        self.content = fit_content_model(self.movies)
        self.collaborative = fit_collaborative_model(self.ratings)

    def test_content_recommends_similar_metadata_and_excludes_seed(self):
        result = recommend(self.movies, self.content, self.collaborative, mode="content", seed_movie_id="m1", limit=3)
        self.assertNotIn("m1", [row["movie_id"] for row in result])
        self.assertEqual(result[0]["movie_id"], "m2")

    def test_collaborative_uses_rating_patterns_and_excludes_rated(self):
        result = recommend(self.movies, self.content, self.collaborative, mode="collaborative",
                           profile_ratings={"m1": 5}, limit=3)
        ids = [row["movie_id"] for row in result]
        self.assertNotIn("m1", ids)
        self.assertIn("m2", ids)

    def test_hybrid_scores_are_bounded_and_exclusions_apply(self):
        result = recommend(self.movies, self.content, self.collaborative, mode="hybrid",
                           profile_ratings={"m1": 5}, excluded_ids={"m4"}, alpha=0.4)
        self.assertTrue(all(0 <= row["score"] <= 1 for row in result))
        self.assertNotIn("m1", [row["movie_id"] for row in result])
        self.assertNotIn("m4", [row["movie_id"] for row in result])

if __name__ == "__main__":
    unittest.main()

