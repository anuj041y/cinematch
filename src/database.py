"""SQLite persistence for local CineMatch profiles and activity."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import DATABASE_PATH


def _connect(path: str | Path | None = None) -> sqlite3.Connection:
    db_path = Path(path or DATABASE_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def _connection(path: str | Path | None = None):
    connection = _connect(path)
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_db(path: str | Path | None = None) -> None:
    with _connection(path) as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS profiles (
                profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ratings (
                profile_id INTEGER NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
                movie_id TEXT NOT NULL,
                rating REAL NOT NULL CHECK(rating >= 1 AND rating <= 5),
                updated_at TEXT NOT NULL,
                PRIMARY KEY(profile_id, movie_id)
            );
            CREATE TABLE IF NOT EXISTS rating_history (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
                movie_id TEXT NOT NULL,
                rating REAL NOT NULL CHECK(rating >= 1 AND rating <= 5),
                rated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS favourites (
                profile_id INTEGER NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
                movie_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(profile_id, movie_id)
            );
            CREATE TABLE IF NOT EXISTS watchlist (
                profile_id INTEGER NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
                movie_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(profile_id, movie_id)
            );
            CREATE TABLE IF NOT EXISTS watch_history (
                profile_id INTEGER NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
                movie_id TEXT NOT NULL,
                watched_at TEXT NOT NULL,
                PRIMARY KEY(profile_id, movie_id)
            );
            CREATE TABLE IF NOT EXISTS preferences (
                profile_id INTEGER NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
                preference_key TEXT NOT NULL,
                preference_value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY(profile_id, preference_key)
            );
        """)
        connection.execute(
            "INSERT OR IGNORE INTO profiles(name, created_at) VALUES(?, ?)",
            ("Default", _now()),
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_profiles(path: str | Path | None = None) -> list[dict[str, Any]]:
    init_db(path)
    with _connection(path) as connection:
        rows = connection.execute("SELECT profile_id, name, created_at FROM profiles ORDER BY profile_id").fetchall()
    return [dict(row) for row in rows]


def create_profile(name: str, path: str | Path | None = None) -> int:
    clean_name = " ".join(name.split()).strip()
    if not clean_name:
        raise ValueError("Enter a profile name.")
    if len(clean_name) > 40:
        raise ValueError("Profile names must be 40 characters or fewer.")
    init_db(path)
    with _connection(path) as connection:
        connection.execute("INSERT OR IGNORE INTO profiles(name, created_at) VALUES(?, ?)", (clean_name, _now()))
        row = connection.execute("SELECT profile_id FROM profiles WHERE name = ?", (clean_name,)).fetchone()
    return int(row["profile_id"])


def _set_membership(table: str, profile_id: int, movie_id: str, enabled: bool,
                    path: str | Path | None = None) -> None:
    if table not in {"favourites", "watchlist"}:
        raise ValueError("Unsupported collection.")
    init_db(path)
    with _connection(path) as connection:
        if enabled:
            connection.execute(
                f"INSERT OR IGNORE INTO {table}(profile_id, movie_id, created_at) VALUES(?, ?, ?)",
                (int(profile_id), str(movie_id), _now()),
            )
        else:
            connection.execute(f"DELETE FROM {table} WHERE profile_id = ? AND movie_id = ?", (int(profile_id), str(movie_id)))


def set_favourite(profile_id: int, movie_id: str, enabled: bool = True, path: str | Path | None = None) -> None:
    _set_membership("favourites", profile_id, movie_id, enabled, path)


def set_watchlist(profile_id: int, movie_id: str, enabled: bool = True, path: str | Path | None = None) -> None:
    _set_membership("watchlist", profile_id, movie_id, enabled, path)


def set_rating(profile_id: int, movie_id: str, rating: float, path: str | Path | None = None) -> None:
    value = float(rating)
    if not 1 <= value <= 5:
        raise ValueError("Ratings must be between 1 and 5.")
    init_db(path)
    with _connection(path) as connection:
        timestamp = _now()
        connection.execute("""
            INSERT INTO ratings(profile_id, movie_id, rating, updated_at) VALUES(?, ?, ?, ?)
            ON CONFLICT(profile_id, movie_id) DO UPDATE SET rating=excluded.rating, updated_at=excluded.updated_at
        """, (int(profile_id), str(movie_id), value, timestamp))
        connection.execute("INSERT INTO rating_history(profile_id, movie_id, rating, rated_at) VALUES(?, ?, ?, ?)",
                           (int(profile_id), str(movie_id), value, timestamp))


def remove_rating(profile_id: int, movie_id: str, path: str | Path | None = None) -> None:
    init_db(path)
    with _connection(path) as connection:
        connection.execute("DELETE FROM ratings WHERE profile_id = ? AND movie_id = ?", (int(profile_id), str(movie_id)))


def mark_watched(profile_id: int, movie_id: str, watched: bool = True, path: str | Path | None = None) -> None:
    init_db(path)
    with _connection(path) as connection:
        if watched:
            connection.execute("""
                INSERT INTO watch_history(profile_id, movie_id, watched_at) VALUES(?, ?, ?)
                ON CONFLICT(profile_id, movie_id) DO UPDATE SET watched_at=excluded.watched_at
            """, (int(profile_id), str(movie_id), _now()))
            connection.execute("DELETE FROM watchlist WHERE profile_id = ? AND movie_id = ?", (int(profile_id), str(movie_id)))
        else:
            connection.execute("DELETE FROM watch_history WHERE profile_id = ? AND movie_id = ?", (int(profile_id), str(movie_id)))


def get_collection(table: str, profile_id: int, path: str | Path | None = None) -> list[dict[str, Any]]:
    if table not in {"ratings", "rating_history", "favourites", "watchlist", "watch_history"}:
        raise ValueError("Unsupported collection.")
    init_db(path)
    with _connection(path) as connection:
        rows = connection.execute(f"SELECT * FROM {table} WHERE profile_id = ? ORDER BY rowid DESC", (int(profile_id),)).fetchall()
    return [dict(row) for row in rows]


def get_preference(profile_id: int, key: str, default: str | None = None,
                   path: str | Path | None = None) -> str | None:
    init_db(path)
    with _connection(path) as connection:
        row = connection.execute("SELECT preference_value FROM preferences WHERE profile_id = ? AND preference_key = ?",
                                 (int(profile_id), key)).fetchone()
    return str(row["preference_value"]) if row else default


def set_preference(profile_id: int, key: str, value: str, path: str | Path | None = None) -> None:
    init_db(path)
    with _connection(path) as connection:
        connection.execute("""
            INSERT INTO preferences(profile_id, preference_key, preference_value, updated_at) VALUES(?, ?, ?, ?)
            ON CONFLICT(profile_id, preference_key) DO UPDATE SET
                preference_value=excluded.preference_value, updated_at=excluded.updated_at
        """, (int(profile_id), str(key), str(value), _now()))

