import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import (DatasetError, format_genres, format_overview, format_people,
                             format_rating_and_votes, format_runtime, format_year,
                             load_catalog, load_ratings, normalize_catalog,
                             normalize_external_id, parse_names, prepare_data)


class DataLoaderTests(unittest.TestCase):
    def test_external_ids_are_normalized_without_truncation(self):
        self.assertEqual(normalize_external_id("12.0"), "12")
        self.assertIsNone(normalize_external_id("12.5"))
        self.assertEqual(parse_names('[{"id": 1, "name": "Drama"}]'), ["Drama"])
        self.assertEqual(parse_names("Action|Drama"), ["Action", "Drama"])
        self.assertEqual(parse_names("(no genres listed)"), [])
        self.assertEqual(parse_names(pd.NA), [])

    def test_metadata_normalization_and_formatting_handle_missing_values(self):
        frame = normalize_catalog(pd.DataFrame([{
            "movie_id": "tmdb:4", "title": float("nan"), "release_year": "bad",
            "genres": [{"id": 1, "name": "Drama"}, {"id": 2, "name": "Sci-Fi"}],
            "overview": "None", "keywords": '[{"name": "time travel"}]',
            "cast": ["Actor One", "Actor Two"], "director": pd.NA,
            "avg_rating": float("nan"), "rating_count": "nan", "runtime": -2,
            "popularity": "invalid", "poster_path": "nan", "poster_url": None,
            "tmdb_id": "4", "imdb_id": "None",
        }]))
        movie = frame.iloc[0]
        self.assertEqual(movie["title"], "Untitled movie")
        self.assertEqual(movie["genres"], "Drama|Sci-Fi")
        self.assertEqual(movie["keywords"], "time travel")
        self.assertEqual(movie["cast"], "Actor One|Actor Two")
        self.assertEqual(movie["overview"], "")
        self.assertEqual(movie["poster_path"], "")
        self.assertTrue(pd.isna(movie["release_year"]))
        self.assertTrue(pd.isna(movie["avg_rating"]))
        self.assertTrue(pd.isna(movie["runtime"]))
        self.assertIsNone(format_year(float("nan")))
        self.assertIsNone(format_runtime(pd.NA))
        self.assertEqual(format_genres('[{"name":"Drama"}, {"name":"Comedy"}]'), "Drama · Comedy")
        self.assertEqual(format_people("Actor One|Actor Two"), "Actor One, Actor Two")
        self.assertEqual(format_overview(float("nan")), "")
        self.assertIsNone(format_rating_and_votes(pd.NA, 8))

    def test_missing_kaggle_links_do_not_match_numeric_ids_directly(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, out = Path(temp) / "raw", Path(temp) / "processed"
            raw.mkdir()
            pd.DataFrame({"id": ["42"], "title": ["TMDB Film"], "genres": ["[]"]}).to_csv(
                raw / "movies_metadata.csv", index=False)
            pd.DataFrame({"userId": [1, 2], "movieId": [42, 99], "rating": [4.0, 5.0],
                          "timestamp": [100, "invalid"]}).to_csv(
                raw / "ratings.csv", index=False)
            summary = prepare_data(raw, out)
            ratings = load_ratings(out)
            self.assertEqual(summary["source_rating_rows"], 2)
            self.assertEqual(summary["valid_ratings"], 2)
            self.assertEqual(summary["rejected_ratings"], 0)
            self.assertEqual(summary["unique_rated_movies"], 2)
            self.assertEqual(summary["mapped_ratings"], 0)
            self.assertEqual(summary["unmapped_ratings"], 2)
            self.assertEqual(summary["invalid_timestamps"], 1)
            self.assertFalse(summary["mapping_available"])
            self.assertEqual(ratings["movie_id"].astype(str).tolist(), ["ml:42", "ml:99"])
            self.assertEqual(ratings["timestamp"].iloc[0], 100)
            self.assertTrue(pd.isna(ratings["timestamp"].iloc[1]))
            self.assertEqual(ratings.columns.tolist(), ["user_id", "movie_id", "rating", "timestamp"])

    def test_invalid_rows_are_counted_and_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, out = Path(temp) / "raw", Path(temp) / "processed"
            raw.mkdir()
            out.mkdir()
            pd.DataFrame({"id": ["42"], "title": ["TMDB Film"], "genres": ["[]"]}).to_csv(
                raw / "movies_metadata.csv", index=False)
            pd.DataFrame({"userId": [None, 2, 3], "movieId": [1, "bad", 3],
                          "rating": [4.0, 4.0, "missing"]}).to_csv(raw / "ratings.csv", index=False)
            prior_ratings = "user_id,movie_id,rating,timestamp\nold,ml:5,4.0,100\n"
            prior_movies = "old movie catalog\n"
            (out / "ratings_clean.csv").write_text(prior_ratings, encoding="utf-8")
            (out / "movies_clean.csv").write_text(prior_movies, encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, "All 3 rows.*invalid_movie_id: 1.*invalid_rating: 1.*invalid_user_id: 1"):
                prepare_data(raw, out)
            self.assertEqual((out / "ratings_clean.csv").read_text(encoding="utf-8"), prior_ratings)
            self.assertEqual((out / "movies_clean.csv").read_text(encoding="utf-8"), prior_movies)
            self.assertEqual(list(out.glob(".ratings_clean.*.tmp")), [])

    def test_empty_ratings_input_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, out = Path(temp) / "raw", Path(temp) / "processed"
            raw.mkdir()
            out.mkdir()
            pd.DataFrame({"id": ["42"], "title": ["TMDB Film"], "genres": ["[]"]}).to_csv(
                raw / "movies_metadata.csv", index=False)
            (raw / "ratings.csv").write_text("userId,movieId,rating,timestamp\n", encoding="utf-8")
            prior = "existing cleaned ratings\n"
            (out / "ratings_clean.csv").write_text(prior, encoding="utf-8")
            with self.assertRaisesRegex(DatasetError, "contains no data rows"):
                prepare_data(raw, out)
            self.assertEqual((out / "ratings_clean.csv").read_text(encoding="utf-8"), prior)

    def test_explicit_tmdb_rating_id_column_maps_without_links(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, out = Path(temp) / "raw", Path(temp) / "processed"
            raw.mkdir()
            pd.DataFrame({"id": ["42"], "title": ["TMDB Film"], "genres": ["[]"]}).to_csv(
                raw / "movies_metadata.csv", index=False)
            pd.DataFrame({"userId": [1], "tmdbId": [42], "rating": [4.0]}).to_csv(
                raw / "ratings.csv", index=False)
            summary = prepare_data(raw, out)
            ratings = load_ratings(out)
            self.assertEqual(summary["ratings"], 1)
            self.assertEqual(ratings.iloc[0]["movie_id"], "tmdb:42")

    def test_movielens_cleaning_preserves_valid_rows_and_missing_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, out = Path(temp) / "raw", Path(temp) / "processed"
            raw.mkdir()
            pd.DataFrame({"movieId": [1, 2, 3], "title": ["Space Film (2001)", None, "Quiet Drama (2004)"],
                          "genres": ["Action|Sci-Fi", "Comedy", "Drama"]}).to_csv(raw / "movies.csv", index=False)
            pd.DataFrame({"userId": [10, 11, 10, 12, 13], "movieId": [1, 1, 1, 99, 3],
                          "rating": [5, 3, 4, 4, 6], "timestamp": [100, 200, 300, 400, 500]}).to_csv(raw / "ratings.csv", index=False)
            summary = prepare_data(raw, out)
            movies = pd.read_csv(out / "movies_clean.csv")
            ratings = pd.read_csv(out / "ratings_clean.csv")
            loaded_catalog = load_catalog(out)
            loaded_ratings = load_ratings(out)
            self.assertEqual(summary["movies"], 2)
            self.assertEqual(summary["ratings"], 4)
            self.assertEqual(summary["source_rating_rows"], 5)
            self.assertEqual(summary["valid_ratings"], 4)
            self.assertEqual(summary["rejected_ratings"], 1)
            self.assertEqual(summary["rejected_reasons"], {"invalid_rating": 1})
            self.assertEqual(summary["users"], 3)
            self.assertEqual(summary["unique_rated_movies"], 2)
            self.assertEqual(summary["mapped_ratings"], 3)
            self.assertEqual(summary["unmapped_ratings"], 1)
            row = movies[movies["movie_id"] == "ml:1"].iloc[0]
            self.assertEqual(row["release_year"], 2001)
            self.assertAlmostEqual(row["avg_rating"], 4.0)
            self.assertEqual(len(ratings), 4)
            self.assertEqual(ratings["movie_id"].astype(str).tolist(), ["ml:1", "ml:1", "ml:1", "ml:99"])
            self.assertEqual(ratings["timestamp"].tolist(), [100, 200, 300, 400])
            self.assertEqual(len(loaded_catalog), 2)
            self.assertEqual(len(loaded_ratings), 4)
            self.assertGreater(summary["ratings_output_size_bytes"], 0)
            self.assertTrue(Path(summary["ratings_output_path"]).is_file())

    def test_load_ratings_can_filter_to_catalog_ids_and_reports_mismatches(self):
        with tempfile.TemporaryDirectory() as temp:
            processed = Path(temp)
            pd.DataFrame({"user_id": ["1", "2"], "movie_id": ["ml:42", "tmdb:42"],
                          "rating": [4.0, 5.0], "timestamp": [100, 200]}).to_csv(
                processed / "ratings_clean.csv", index=False)
            ratings = load_ratings(processed, {"tmdb:42"})
            self.assertEqual(ratings["movie_id"].astype(str).tolist(), ["tmdb:42"])
            self.assertEqual(ratings.attrs["source_rows"], 2)
            self.assertEqual(ratings.attrs["unmatched_rows"], 1)

    def test_kaggle_ratings_join_through_links(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, out = Path(temp) / "raw", Path(temp) / "processed"
            raw.mkdir()
            pd.DataFrame({"id": ["101"], "title": ["A Film"], "genres": ['[{"id": 1, "name": "Drama"}]',],
                          "release_date": ["2001-01-02"], "overview": ["A story"]}).to_csv(raw / "movies_metadata.csv", index=False)
            pd.DataFrame({"movieId": [7], "imdbId": [""], "tmdbId": [101]}).to_csv(raw / "links.csv", index=False)
            pd.DataFrame({"userId": [1, 2], "movieId": [7, 7], "rating": [4.5, 3.5],
                          "timestamp": [10, 20]}).to_csv(raw / "ratings.csv", index=False)
            summary = prepare_data(raw, out)
            movies = pd.read_csv(out / "movies_clean.csv")
            self.assertEqual(summary["ratings"], 2)
            self.assertEqual(movies.iloc[0]["movie_id"], "tmdb:101")
            self.assertEqual(movies.iloc[0]["genres"], "Drama")
            self.assertAlmostEqual(movies.iloc[0]["avg_rating"], 4.0)

    def test_tmdb_vote_average_is_converted_to_five_point_scale(self):
        with tempfile.TemporaryDirectory() as temp:
            raw, out = Path(temp) / "raw", Path(temp) / "processed"
            raw.mkdir()
            pd.DataFrame({"id": ["201"], "title": ["Low Rated Film"], "genres": ["[]"],
                          "vote_average": [4.0], "vote_count": [12]}).to_csv(raw / "movies_metadata.csv", index=False)
            prepare_data(raw, out)
            movies = pd.read_csv(out / "movies_clean.csv")
            self.assertEqual(movies.iloc[0]["avg_rating"], 2.0)

    def test_missing_dataset_is_actionable(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(DatasetError):
                prepare_data(temp, Path(temp) / "out")


if __name__ == "__main__":
    unittest.main()

