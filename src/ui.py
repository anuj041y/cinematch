"""Shared Streamlit rendering helpers."""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from config.settings import MODEL_DIR, MODEL_PATH, PROCESSED_DATA_DIR, ROOT_DIR
from src import database
from src.data_loader import (DatasetError, format_genres, format_overview, format_people,
                             format_rating_and_votes, format_runtime, format_title,
                             format_year, load_catalog)
from src.model_artifacts import load_model_artifact, movie_ids_fingerprint
from src.posters import poster_html, poster_source
from src.widget_keys import widget_key


@st.cache_data(show_spinner=False)
def get_catalog(processed_dir: str, modified: float) -> pd.DataFrame:
    del modified
    return load_catalog(processed_dir)


@st.cache_data(show_spinner=False)
def get_genre_counts(processed_dir: str, modified: float) -> dict[str, int]:
    movies = get_catalog(processed_dir, modified)
    counts: dict[str, int] = {}
    for value in movies["genres"]:
        for genre in set(str(value).split("|")):
            genre = genre.strip()
            if genre:
                counts[genre] = counts.get(genre, 0) + 1
    return counts


@st.cache_data(show_spinner=False)
def get_rating_distribution(metadata_path: str, modified: float) -> pd.DataFrame:
    del modified
    try:
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame(columns=["rating", "ratings"])
    distribution = metadata.get("rating_distribution", {})
    rows = []
    if isinstance(distribution, dict):
        for rating, count in distribution.items():
            try:
                rows.append((float(rating), int(count)))
            except (TypeError, ValueError, OverflowError):
                continue
    frame = pd.DataFrame(sorted(rows), columns=["rating", "ratings"])
    try:
        frame.attrs["matched_rating_count"] = int(metadata.get("ratings", 0))
    except (TypeError, ValueError, OverflowError):
        frame.attrs["matched_rating_count"] = 0
    return frame


@st.cache_resource(show_spinner="Loading recommendation models…")
def get_models(model_path: str, processed_dir: str, movies_mtime: float, ratings_mtime: float,
               model_mtime: float, movie_ids_hash: str):
    del processed_dir, movies_mtime, ratings_mtime, model_mtime
    target = Path(model_path)
    if target.is_file():
        artifact, error = load_model_artifact(target, movie_ids_hash)
        if artifact:
            return artifact["content"], artifact["collaborative"], "Recommendations are ready."
        logging.warning("%s", error)
    return None, None, "Saved models are missing or out of date. Run `python scripts/train_models.py` to refresh them."


def data_files_mtime() -> tuple[float, float, float]:
    movie_file = PROCESSED_DATA_DIR / "movies_clean.csv"
    rating_file = PROCESSED_DATA_DIR / "ratings_clean.csv"
    model_file = MODEL_PATH
    return (movie_file.stat().st_mtime if movie_file.exists() else 0.0,
            rating_file.stat().st_mtime if rating_file.exists() else 0.0,
            model_file.stat().st_mtime if model_file.exists() else 0.0)


def inject_theme() -> None:
    st.markdown("""
    <style>
      .stApp { background: radial-gradient(ellipse at top, #241a28 0%, #111217 42%, #101114 100%); }
      [data-testid="stSidebar"] { background: #15161c; border-right: 1px solid #2b2c34; }
      .block-container { padding-top: 2.2rem; max-width: 1400px; }
      h1, h2, h3 { letter-spacing: -0.025em; }
      .cine-kicker { color:#ef6a72; text-transform:uppercase; letter-spacing:.16em; font-size:.74rem; font-weight:700; }
      .cine-subtle { color:#a9aab4; }
      .cine-reason { color:#bebfc8; font-size:.84rem; line-height:1.4; min-height:2.4em; }
      div[data-testid="stVerticalBlockBorderWrapper"] { border-color:#34353e; background:rgba(28,29,36,.60); }
      .stButton button { border-radius: 10px; }
      div[data-testid="stMetric"] { background:#1c1d24; border:1px solid #34353e; border-radius:12px; padding:12px; }
    </style>
    """, unsafe_allow_html=True)


def setup_page(title: str) -> int:
    database.init_db()
    inject_theme()
    st.sidebar.markdown("# 🎬 CineMatch")
    st.sidebar.caption("Movie discovery, tuned to your taste")
    profiles = database.list_profiles()
    profile_ids = [int(profile["profile_id"]) for profile in profiles]
    names = {int(profile["profile_id"]): profile["name"] for profile in profiles}
    current = st.session_state.get("profile_id")
    if current not in profile_ids:
        current = profile_ids[0]
    chosen = st.sidebar.selectbox("Local profile", profile_ids, index=profile_ids.index(current),
                                  format_func=lambda profile_id: names.get(profile_id, "Profile"),
                                  key=widget_key("select-profile", "navigation"))
    st.session_state["profile_id"] = int(chosen)
    with st.sidebar.form(widget_key("create-profile-form", "navigation"), clear_on_submit=True):
        new_name = st.text_input("Create another profile", max_chars=40, placeholder="Profile name",
                                 key=widget_key("profile-name", "navigation"))
        submitted = st.form_submit_button("Add profile", width="stretch",
                                          key=widget_key("submit-profile", "navigation"))
    if submitted and new_name.strip():
        try:
            st.session_state["profile_id"] = database.create_profile(new_name)
            st.rerun()
        except ValueError as exc:
            st.sidebar.error(str(exc))
    st.sidebar.divider()
    st.sidebar.caption("Your ratings, favourites, watchlist, and history are stored locally in SQLite.")
    return int(chosen)


def load_page_data() -> tuple[pd.DataFrame, dict[str, Any] | None, dict[str, Any] | None, str]:
    movies_path = PROCESSED_DATA_DIR / "movies_clean.csv"
    if not movies_path.exists():
        raise DatasetError("Prepared data is missing. Place the downloaded CSVs in data/raw/ and run `python scripts/prepare_data.py`.")
    movie_mtime, rating_mtime, model_mtime = data_files_mtime()
    movies = get_catalog(str(PROCESSED_DATA_DIR), movie_mtime)
    movie_ids_hash = movie_ids_fingerprint(movies["movie_id"].astype(str))
    content, collaborative, message = get_models(str(MODEL_PATH), str(PROCESSED_DATA_DIR),
                                                  movie_mtime, rating_mtime, model_mtime,
                                                  movie_ids_hash)
    return movies, content, collaborative, message


def show_dataset_setup() -> None:
    st.title("CineMatch")
    st.markdown("### Add movie data")
    st.info("Put a supported movie catalog in `data/raw/`, then prepare it before using the app.")
    st.code("""python -m pip install -r requirements.txt\npython scripts/prepare_data.py\npython scripts/train_models.py\nstreamlit run app.py""", language="powershell")
    st.markdown("Dataset instructions and supported CSV schemas are in [README.md](README.md).")


def profile_ratings(profile_id: int) -> dict[str, float]:
    return {str(row["movie_id"]): float(row["rating"])
            for row in database.get_collection("ratings", profile_id)}


def recommendations_frame(movies: pd.DataFrame, recommendations: list[dict[str, Any]]) -> pd.DataFrame:
    order = {str(item["movie_id"]): index for index, item in enumerate(recommendations)}
    result = movies[movies["movie_id"].astype(str).isin(order)].copy()
    result["_recommendation_order"] = result["movie_id"].astype(str).map(order)
    return result.sort_values("_recommendation_order").drop(columns="_recommendation_order")


def _collection_ids(table: str, profile_id: int) -> set[str]:
    return {str(row["movie_id"]) for row in database.get_collection(table, profile_id)}


def go_to_movie(movie_id: str) -> None:
    st.session_state["selected_movie_id"] = str(movie_id)
    st.switch_page("pages/7_Movie_Details.py")


def render_poster(movie: Any, *, alt: str | None = None) -> None:
    source = poster_source(movie, ROOT_DIR)
    title = alt or format_title(movie.get("title"))
    st.markdown(poster_html(source, ROOT_DIR, title), unsafe_allow_html=True)


def _movie_rating_caption(movie: pd.Series) -> str:
    year_label = format_year(movie.get("release_year")) or "Year unavailable"
    rating_label = format_rating_and_votes(movie.get("avg_rating"), movie.get("rating_count")) or "Rating unavailable"
    return f"{year_label} · {rating_label}"


def render_movie_grid(movies: pd.DataFrame, profile_id: int, *, columns: int = 4,
                      section: str, show_actions: bool = True,
                      reasons: dict[str, str] | None = None) -> None:
    if movies.empty:
        st.info("No movies match these filters yet.")
        return
    favourites = _collection_ids("favourites", profile_id) if show_actions else set()
    watchlist = _collection_ids("watchlist", profile_id) if show_actions else set()
    column_set = st.columns(columns, gap="medium")
    for index, (_, movie) in enumerate(movies.iterrows()):
        movie_id = str(movie["movie_id"])
        with column_set[index % columns]:
            with st.container(border=True):
                render_poster(movie)
                st.markdown(f"**{format_title(movie.get('title'))}**")
                st.caption(_movie_rating_caption(movie))
                if reasons and movie_id in reasons:
                    st.markdown(f"<div class='cine-reason'>{html.escape(reasons[movie_id])}</div>", unsafe_allow_html=True)
                if st.button("View details", key=widget_key("view-details", section, profile_id, movie_id),
                             width="stretch"):
                    go_to_movie(movie_id)
                if show_actions:
                    left, right = st.columns(2)
                    with left:
                        label = "♥ Saved" if movie_id in favourites else "♡ Favourite"
                        if st.button(label, key=widget_key("toggle-favourite", section, profile_id, movie_id),
                                     width="stretch"):
                            database.set_favourite(profile_id, movie_id, movie_id not in favourites)
                            st.rerun()
                    with right:
                        label = "✓ Listed" if movie_id in watchlist else "+ Watchlist"
                        if st.button(label, key=widget_key("toggle-watchlist", section, profile_id, movie_id),
                                     width="stretch"):
                            database.set_watchlist(profile_id, movie_id, movie_id not in watchlist)
                            st.rerun()


def render_onboarding(movies: pd.DataFrame, profile_id: int) -> None:
    existing = profile_ratings(profile_id)
    if existing:
        return
    st.info("Rate a few movies you know to get suggestions based on your taste. Recommendations based on other viewers are available when the catalog includes linked ratings.")
    ranked = movies.sort_values(["rating_count", "avg_rating"], ascending=False, na_position="last").head(500)
    choices = ranked["movie_id"].astype(str).tolist()
    title_by_id = dict(zip(ranked["movie_id"].astype(str), ranked["title"].astype(str)))
    with st.form(widget_key("onboarding-form", "for-you", profile_id)):
        st.markdown("#### Choose up to three movies and rate them")
        selected: list[str | None] = []
        values: list[float] = []
        for slot in range(3):
            left, right = st.columns([3, 1])
            with left:
                selected.append(st.selectbox(f"Movie {slot + 1}", [None] + choices,
                                             format_func=lambda movie_id: "Choose a movie…" if movie_id is None else title_by_id.get(movie_id, movie_id),
                                             index=0, key=widget_key(f"choose-movie-{slot}", "onboarding", profile_id)))
            with right:
                values.append(float(st.select_slider(f"Rating {slot + 1}", options=[1, 2, 3, 4, 5], value=4,
                                                     key=widget_key(f"rate-movie-{slot}", "onboarding", profile_id))))
        submitted = st.form_submit_button("Save ratings and personalize", type="primary",
                                          key=widget_key("submit-onboarding", "for-you", profile_id))
    if submitted:
        unique = [movie_id for movie_id in selected if movie_id]
        if len(unique) < 2:
            st.warning("Choose at least two different movies to personalize recommendations.")
        elif len(set(unique)) != len(unique):
            st.warning("Please choose a different movie in each selected row.")
        else:
            for movie_id, rating in zip(selected, values):
                if movie_id:
                    database.set_rating(profile_id, movie_id, rating)
            st.success("Your ratings were saved to this local profile.")
            st.rerun()


def render_details(movies: pd.DataFrame, profile_id: int) -> None:
    movie_id = str(st.session_state.get("selected_movie_id", ""))
    matches = movies[movies["movie_id"].astype(str).eq(movie_id)]
    if matches.empty:
        st.info("Choose a movie from Home, Explore, or Recommendations to view its details.")
        return
    movie = matches.iloc[0]
    st.markdown("<div class='cine-kicker'>Movie details</div>", unsafe_allow_html=True)
    left, right = st.columns([1, 2.4], gap="large")
    with left:
        render_poster(movie)
    with right:
        st.title(format_title(movie.get("title")))
        st.caption(_movie_rating_caption(movie))
        genres = format_genres(movie.get("genres"))
        if genres:
            st.markdown(f"**Genres:** {genres}")
        overview = format_overview(movie.get("overview"))
        if overview:
            st.write(overview)
        details = (("Runtime", format_runtime(movie.get("runtime"))),
                   ("Director", format_people(movie.get("director"))),
                   ("Cast", format_people(movie.get("cast"))))
        for label, formatted in details:
            if formatted:
                st.markdown(f"**{label}:** {formatted}")
        favourites = _collection_ids("favourites", profile_id)
        watchlist = _collection_ids("watchlist", profile_id)
        watched = _collection_ids("watch_history", profile_id)
        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("Remove favourite" if movie_id in favourites else "Add favourite",
                         key=widget_key("toggle-favourite", "movie-details", profile_id, movie_id),
                         width="stretch"):
                database.set_favourite(profile_id, movie_id, movie_id not in favourites)
                st.rerun()
        with b2:
            if st.button("Remove from watchlist" if movie_id in watchlist else "Add to watchlist",
                         key=widget_key("toggle-watchlist", "movie-details", profile_id, movie_id),
                         width="stretch"):
                database.set_watchlist(profile_id, movie_id, movie_id not in watchlist)
                st.rerun()
        with b3:
            if st.button("Watched ✓" if movie_id in watched else "Mark watched",
                         key=widget_key("toggle-watch-history", "movie-details", profile_id, movie_id),
                         width="stretch"):
                database.mark_watched(profile_id, movie_id, movie_id not in watched)
                st.rerun()
        current_ratings = profile_ratings(profile_id)
        with st.form(widget_key("rating-form", "movie-details", profile_id, movie_id)):
            current = int(round(current_ratings[movie_id])) if movie_id in current_ratings else 3
            value = st.select_slider("Your rating (1–5)", options=[1, 2, 3, 4, 5], value=current,
                                     key=widget_key("select-rating", "movie-details", profile_id, movie_id))
            c1, c2 = st.columns(2)
            save_rating = c1.form_submit_button("Save rating", type="primary",
                                                key=widget_key("save-rating", "movie-details", profile_id, movie_id))
            clear_rating = c2.form_submit_button("Remove rating",
                                                 key=widget_key("remove-rating", "movie-details", profile_id, movie_id))
        if save_rating:
            database.set_rating(profile_id, movie_id, value)
            st.success("Rating saved.")
            st.rerun()
        if clear_rating:
            database.remove_rating(profile_id, movie_id)
            st.rerun()

