import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import database


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp.name) / "test.sqlite3"
        database.init_db(self.db_path)
        self.profile = database.create_profile("Test Profile", self.db_path)

    def tearDown(self):
        self.temp.cleanup()

    def test_favourites_are_unique_and_persistent(self):
        database.set_favourite(self.profile, "tmdb:42", path=self.db_path)
        database.set_favourite(self.profile, "tmdb:42", path=self.db_path)
        self.assertEqual(len(database.get_collection("favourites", self.profile, self.db_path)), 1)
        database.init_db(self.db_path)
        self.assertEqual(len(database.get_collection("favourites", self.profile, self.db_path)), 1)

    def test_rating_updates_existing_row(self):
        database.set_rating(self.profile, "ml:8", 4, self.db_path)
        database.set_rating(self.profile, "ml:8", 2.5, self.db_path)
        values = database.get_collection("ratings", self.profile, self.db_path)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0]["rating"], 2.5)
        self.assertEqual(len(database.get_collection("rating_history", self.profile, self.db_path)), 2)

    def test_watchlist_and_history_transitions(self):
        database.set_watchlist(self.profile, "tmdb:9", path=self.db_path)
        self.assertEqual(len(database.get_collection("watchlist", self.profile, self.db_path)), 1)
        database.mark_watched(self.profile, "tmdb:9", path=self.db_path)
        self.assertEqual(database.get_collection("watchlist", self.profile, self.db_path), [])
        self.assertEqual(len(database.get_collection("watch_history", self.profile, self.db_path)), 1)
        database.mark_watched(self.profile, "tmdb:9", False, self.db_path)
        self.assertEqual(database.get_collection("watch_history", self.profile, self.db_path), [])


if __name__ == "__main__":
    unittest.main()

