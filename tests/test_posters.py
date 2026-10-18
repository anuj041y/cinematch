import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import posters
from src.posters import poster_html, poster_source


class PosterTests(unittest.TestCase):
    def setUp(self):
        posters._LOOKUP_CACHE.clear()
        posters._WATCHMODE_CACHE.clear()
        posters._REMOTE_IMAGE_CACHE.clear()

    def test_missing_and_nan_values_return_existing_bundled_fallback(self):
        source = poster_source({"title": "Unlisted Movie", "poster_url": pd.NA,
                                "poster_path": float("nan"), "poster": "None"}, ROOT)
        self.assertTrue(Path(source).is_file())
        self.assertEqual(Path(source).name, "default_poster.png")

    def test_existing_local_image_is_used_before_url(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "assets").mkdir()
            local = root / "poster.png"
            Image.new("RGB", (40, 60), "coral").save(local)
            source = poster_source({"poster_path": "poster.png", "poster_url": "https://example.com/p.jpg"}, root)
            self.assertEqual(Path(source), local.resolve())

    def test_url_and_tmdb_path_variants_are_resolved(self):
        complete = poster_source({"poster_url": "https://cdn.example.org/poster.webp"}, ROOT)
        self.assertEqual(complete, "https://cdn.example.org/poster.webp")
        tmdb = poster_source({"poster_path": "/abc123.jpg"}, ROOT)
        self.assertEqual(tmdb, "https://image.tmdb.org/t/p/w500/abc123.jpg")

    def test_invalid_url_and_relative_path_use_local_fallback(self):
        for value in ("nan", "None", "ftp://example.com/poster.jpg", "javascript:alert(1)", "poster.jpg"):
            with self.subTest(value=value):
                source = poster_source({"poster_url": value, "poster_path": value}, ROOT)
                self.assertTrue(Path(source).is_file())

    def test_offline_remote_markup_keeps_local_fallback_layer(self):
        fallback = ROOT / "assets" / "default_poster.png"
        markup = poster_html("https://image.tmdb.org/t/p/w500/abc.jpg", ROOT, "Poster")
        self.assertTrue(fallback.is_file())
        self.assertIn("aspect-ratio:2/3", markup)
        self.assertIn("https://image.tmdb.org/t/p/w500/abc.jpg", markup)
        self.assertIn("data:image/svg+xml;base64,", markup)

    def test_card_resolution_does_not_make_network_requests(self):
        with patch("src.posters.requests.get") as get:
            source = poster_source({"title": "A Movie", "poster_path": "", "tmdbId": "862"}, ROOT,
                                   api_key="configured")
        self.assertTrue(Path(source).is_file())
        get.assert_not_called()

    def test_broken_tmdb_image_uses_watchmode_fallback(self):
        broken_image = Mock(status_code=404, headers={"Content-Type": "text/html"})
        fallback = Mock()
        fallback.json.return_value = {"results": [{
            "name": "Planet Earth II", "year": 2016, "poster": "http://images.example/planet-earth-ii.jpg",
        }]}
        fallback.raise_for_status.return_value = None
        with patch("src.posters.requests.get", side_effect=[broken_image, fallback]) as get:
            source = poster_source({"title": "Planet Earth II", "release_year": 2016,
                                    "poster_path": "/broken.jpg"}, ROOT,
                                   api_key="", allow_lookup=True,
                                   watchmode_api_key="watchmode-test-key")
        self.assertEqual(source, "https://images.example/planet-earth-ii.jpg")
        self.assertEqual(get.call_count, 2)

    def test_unverified_tmdb_image_is_kept_when_network_is_unavailable(self):
        url = "https://image.tmdb.org/t/p/w500/planet.jpg"
        with patch("src.posters.requests.get", side_effect=posters.requests.ConnectionError("offline")) as get:
            source = poster_source({"title": "Planet Earth II", "poster_path": "/planet.jpg"}, ROOT,
                                   api_key="", allow_lookup=True,
                                   watchmode_api_key="watchmode-test-key")
        self.assertEqual(source, url)
        get.assert_called_once()

    def test_tmdb_failure_is_temporarily_cached_and_uses_fallback(self):
        with patch("src.posters.requests.get", side_effect=ConnectionError("offline")) as get:
            first = poster_source({"title": "Missing Poster"}, ROOT, api_key="configured", allow_lookup=True,
                                  watchmode_api_key="")
            second = poster_source({"title": "Missing Poster"}, ROOT, api_key="configured", allow_lookup=True,
                                   watchmode_api_key="")
        self.assertEqual(first, second)
        self.assertTrue(Path(first).is_file())
        get.assert_called_once()
        self.assertEqual(get.call_args.kwargs["timeout"], (2, 4))

    def test_watchmode_fills_missing_poster_with_exact_title_and_year(self):
        response = Mock()
        response.json.return_value = {"results": [{
            "name": "The Example Movie", "year": 2020, "image_url": "http://images.example/poster.jpg",
            "relevance": 10,
        }]}
        response.raise_for_status.return_value = None
        movie = {"title": "The Example Movie", "release_year": 2020}
        with patch("src.posters.requests.get", return_value=response) as get:
            first = poster_source(movie, ROOT, api_key="", allow_lookup=True,
                                  watchmode_api_key="watchmode-test-key")
            second = poster_source(movie, ROOT, api_key="", allow_lookup=False,
                                   watchmode_api_key="watchmode-test-key")
        self.assertEqual(first, "https://images.example/poster.jpg")
        self.assertEqual(second, first)
        get.assert_called_once()
        self.assertEqual(get.call_args.kwargs["headers"]["X-API-Key"], "watchmode-test-key")
        self.assertEqual(get.call_args.kwargs["params"], {"search_value": "The Example Movie", "search_type": 3})
        self.assertEqual(get.call_args.kwargs["timeout"], (2, 3))

    def test_watchmode_accepts_a_subtitle_and_one_year_release_difference(self):
        response = Mock()
        response.json.return_value = {"results": [{
            "name": "Planet Earth II: A World of Wonder", "year": 2017,
            "image_url": "https://images.example/planet-earth-ii.jpg",
        }]}
        response.raise_for_status.return_value = None
        with patch("src.posters.requests.get", return_value=response):
            source = poster_source({"title": "Planet Earth II", "release_year": 2016}, ROOT,
                                   api_key="", allow_lookup=True,
                                   watchmode_api_key="watchmode-test-key")
        self.assertEqual(source, "https://images.example/planet-earth-ii.jpg")

    def test_watchmode_does_not_use_different_year_or_title(self):
        for result in (
            {"name": "The Example Movie", "year": 2010, "image_url": "https://images.example/wrong.jpg"},
            {"name": "A Different Movie", "year": 2020, "image_url": "https://images.example/wrong.jpg"},
        ):
            with self.subTest(result=result):
                posters._WATCHMODE_CACHE.clear()
                response = Mock()
                response.json.return_value = {"results": [result]}
                response.raise_for_status.return_value = None
                with patch("src.posters.requests.get", return_value=response):
                    source = poster_source({"title": "The Example Movie", "release_year": 2020}, ROOT,
                                           api_key="", allow_lookup=True,
                                           watchmode_api_key="watchmode-test-key")
                self.assertEqual(Path(source).name, "default_poster.png")

    def test_tmdb_id_alias_is_used_without_confusing_movielens_movie_id(self):
        response = Mock()
        response.json.return_value = {"poster_path": "/found.jpg"}
        response.raise_for_status.return_value = None
        image_response = Mock(status_code=200, headers={"Content-Type": "image/jpeg"})
        with patch("src.posters.requests.get", side_effect=[response, image_response]) as get:
            source = poster_source({"title": "A Movie", "tmdbId": "862"}, ROOT,
                                   api_key="configured", allow_lookup=True)
        self.assertEqual(source, "https://image.tmdb.org/t/p/w500/found.jpg")
        self.assertIn("/movie/862", get.call_args_list[0].args[0])
        get.reset_mock()
        posters._LOOKUP_CACHE.clear()
        response.json.return_value = {"results": [{"poster_path": "/found.jpg"}]}
        with patch("src.posters.requests.get", side_effect=[response, image_response]) as get:
            source = poster_source({"title": "A Movie", "movieId": "862"}, ROOT,
                                   api_key="configured", allow_lookup=True)
        self.assertEqual(source, "https://image.tmdb.org/t/p/w500/found.jpg")
        self.assertIn("/search/movie", get.call_args_list[0].args[0])


if __name__ == "__main__":
    unittest.main()
