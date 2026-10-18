import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.widget_keys import widget_key


class WidgetKeyTests(unittest.TestCase):
    def test_key_is_stable_across_reruns(self):
        first = widget_key("view-details", "recommended", 7, "tmdb:862")
        second = widget_key("view-details", "recommended", 7, "tmdb:862")
        self.assertEqual(first, second)

    def test_same_movie_gets_unique_key_per_section_and_action(self):
        recommended = widget_key("view-details", "recommended", 7, "tmdb:862")
        top_rated = widget_key("view-details", "top-rated", 7, "tmdb:862")
        favourite = widget_key("toggle-favourite", "recommended", 7, "tmdb:862")
        watchlist = widget_key("toggle-watchlist", "recommended", 7, "tmdb:862")
        self.assertEqual(len({recommended, top_rated, favourite, watchlist}), 4)

    def test_keys_change_for_profile_or_movie(self):
        base = widget_key("rate", "history", 1, "ml:4")
        variants = {
            widget_key("rate", "history", 2, "ml:4"),
            widget_key("rate", "history", 1, "ml:5"),
            widget_key("rate", "movie-details", 1, "ml:4"),
        }
        self.assertNotIn(base, variants)


if __name__ == "__main__":
    unittest.main()
