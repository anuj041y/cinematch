import streamlit as st

st.set_page_config(page_title="Insights | CineMatch", page_icon="🎬", layout="wide")

from src import database
from src.data_loader import DatasetError
from config.settings import MODEL_DIR, PROCESSED_DATA_DIR
from src.ui import (data_files_mtime, get_genre_counts, get_rating_distribution, load_page_data,
                    setup_page, show_dataset_setup)
import pandas as pd

profile_id = setup_page("Insights")
try:
    movies, _, _, _ = load_page_data()
except DatasetError:
    show_dataset_setup()
    st.stop()

st.title("Insights")
st.caption("Charts use the supplied movie catalog, dataset ratings, and this profile’s saved ratings.")
try:
    import plotly.express as px
except ImportError:
    st.error("Plotly is missing. Install project dependencies with `python -m pip install -r requirements.txt`.")
    st.stop()

left, right = st.columns(2)
with left:
    metadata_path = MODEL_DIR / "metadata.json"
    rating_distribution = get_rating_distribution(str(metadata_path),
                                                  metadata_path.stat().st_mtime if metadata_path.exists() else 0.0)
    if rating_distribution.empty:
        if rating_distribution.attrs.get("matched_rating_count", 0):
            st.info("The saved rating summary is outdated. Run `python scripts/train_models.py` to refresh it.")
        else:
            st.info("No ratings are mapped to this movie catalog. Add a matching `links.csv` to connect MovieLens ratings to TMDB movies.")
    else:
        fig = px.bar(rating_distribution, x="rating", y="ratings", title="Rating distribution in supplied data",
                     color="ratings", color_continuous_scale="Reds")
        fig.update_layout(template="plotly_dark", xaxis_title="Rating (dataset scale)", yaxis_title="Count")
        st.plotly_chart(fig, width="stretch")
with right:
    genre_counts = get_genre_counts(str(PROCESSED_DATA_DIR), data_files_mtime()[0])
    if genre_counts:
        genres = sorted(genre_counts.items(), key=lambda item: item[1], reverse=True)[:15]
        genre_frame = pd.DataFrame(genres, columns=["Genre", "Movies"])
        fig = px.bar(genre_frame.sort_values("Movies"), x="Movies", y="Genre", orientation="h",
                     title="Most common genres", color="Movies", color_continuous_scale="Purples")
        fig.update_layout(template="plotly_dark", yaxis_title="")
        st.plotly_chart(fig, width="stretch")

local_rows = database.get_collection("ratings", profile_id)
local = pd.DataFrame(local_rows)
if local.empty:
    st.info("Rate movies to see profile insights and genre preferences here.")
else:
    local = local.merge(movies[["movie_id", "title", "genres"]], on="movie_id", how="left")
    left, right = st.columns(2)
    with left:
        rating_counts = local.groupby("rating").size().rename("movies").reset_index()
        fig = px.bar(rating_counts, x="rating", y="movies", title="Your current ratings",
                     color="rating", color_continuous_scale="RdPu")
        fig.update_layout(template="plotly_dark", xaxis_title="Rating (1–5)")
        st.plotly_chart(fig, width="stretch")
    with right:
        genre_values = []
        for _, row in local.iterrows():
            for genre in str(row.get("genres", "") or "").split("|"):
                if genre.strip():
                    genre_values.append({"Genre": genre.strip(), "Your rating": float(row["rating"])})
        if genre_values:
            genre_pref = pd.DataFrame(genre_values).groupby("Genre")["Your rating"].mean().reset_index()
            fig = px.bar(genre_pref.sort_values("Your rating"), x="Your rating", y="Genre", orientation="h",
                         title="Average rating by genre", color="Your rating", color_continuous_scale="Sunset")
            fig.update_layout(template="plotly_dark", xaxis_range=[1, 5], yaxis_title="")
            st.plotly_chart(fig, width="stretch")
    st.subheader("Your movie ratings")
    rated_titles = local.sort_values("updated_at", ascending=False)[["title", "rating", "updated_at"]].head(20)
    st.dataframe(rated_titles, hide_index=True, width="stretch")

history = database.get_collection("watch_history", profile_id)
st.metric("Movies manually marked watched", len(history))
rating_events = database.get_collection("rating_history", profile_id)
if len(rating_events) > 1:
    events = pd.DataFrame(rating_events).merge(movies[["movie_id", "title"]], on="movie_id", how="left")
    events["rated_at"] = pd.to_datetime(events["rated_at"], errors="coerce", utc=True)
    events = events.dropna(subset=["rated_at"]).sort_values("rated_at")
    if not events.empty:
        timeline = px.scatter(events, x="rated_at", y="rating", color="title", title="Rating activity over time",
                              hover_data=["title"], color_discrete_sequence=px.colors.qualitative.Safe)
        timeline.update_layout(template="plotly_dark", yaxis_range=[0.5, 5.5], xaxis_title="Saved at (UTC)", yaxis_title="Rating (1–5)")
        st.plotly_chart(timeline, width="stretch")

