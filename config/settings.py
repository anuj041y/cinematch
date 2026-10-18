"""Project paths and optional API settings."""

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def _load_local_env() -> None:
    """Load local KEY=VALUE settings without overriding environment variables."""
    env_file = ROOT_DIR / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or "=" not in clean:
            continue
        key, value = clean.split("=", 1)
        key, value = key.strip(), value.strip()
        if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env()


def _setting(name: str) -> str:
    """Read a value from the environment, local .env, or Streamlit secrets."""
    value = os.getenv(name, "").strip()
    if value:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        # Secrets are only available while the app runs in Streamlit.
        return ""

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = Path(os.getenv("CINEMATCH_RAW_DATA", DATA_DIR / "raw"))
PROCESSED_DATA_DIR = Path(os.getenv("CINEMATCH_PROCESSED_DATA", DATA_DIR / "processed"))
MODEL_DIR = Path(os.getenv("CINEMATCH_MODEL_DIR", ROOT_DIR / "models"))
ASSETS_DIR = ROOT_DIR / "assets"
DATABASE_PATH = Path(os.getenv("CINEMATCH_DB_PATH", DATA_DIR / "cinematch.sqlite3"))
TMDB_API_KEY = _setting("TMDB_API_KEY")
WATCHMODE_API_KEY = _setting("WATCHMODE_API_KEY")
MODEL_PATH = MODEL_DIR / "cinematch_models.joblib"

