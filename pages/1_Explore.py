import streamlit as st

st.set_page_config(page_title="Explore | CineMatch", page_icon="🎬", layout="wide")

from config.settings import PROCESSED_DATA_DIR
from src.ui import (data_files_mtime, get_genre_counts, load_page_data,
                    render_movie_grid, setup_page, show_dataset_setup)
from src.widget_keys import widget_key
from src.data_loader import DatasetError

profile_id = setup_page("Explore")
try:
    movies, _, _, _ = load_page_data()
except DatasetError:
    show_dataset_setup()
    st.stop()

st.title("Explore movies")
st.caption("Search is for catalog discovery. It does not drive recommendation scoring.")
with st.container(border=True):
    query = st.text_input("Title search", placeholder="Type a title", key=widget_key("search-title", "explore"))
    genres = sorted(get_genre_counts(str(PROCESSED_DATA_DIR), data_files_mtime()[0]))
    selected_genres = st.multiselect("Genres", genres, key=widget_key("filter-genres", "explore"))
    years = movies["release_year"].dropna()
    if not years.empty:
        min_year, max_year = int(years.min()), int(years.max())
        year_range = st.slider("Release year", min_year, max_year, (min_year, max_year),
                               key=widget_key("filter-year", "explore")) if min_year < max_year else (min_year, max_year)
    else:
        year_range = None
    rating_values = movies["avg_rating"].dropna()
    min_rating = float(rating_values.min()) if not rating_values.empty else 0.0
    max_rating = float(rating_values.max()) if not rating_values.empty else 5.0
    rating_range = st.slider("Dataset rating", min_value=min_rating, max_value=max_rating,
                             value=(min_rating, max_rating), step=0.1,
                             key=widget_key("filter-rating", "explore")) if max_rating > min_rating else (min_rating, max_rating)
    sort_by = st.selectbox("Sort by", ["Popularity", "Average rating", "Release year", "Title"],
                           key=widget_key("sort", "explore"))

filtered = movies.copy()
if query.strip():
    filtered = filtered[filtered["title"].astype(str).str.contains(query.strip(), case=False, regex=False)]
for genre in selected_genres:
    filtered = filtered[filtered["genres"].fillna("").str.contains(genre, case=False, regex=False)]
if year_range:
    filtered = filtered[filtered["release_year"].between(year_range[0], year_range[1])]
filtered = filtered[filtered["avg_rating"].fillna(0).between(rating_range[0], rating_range[1])]
sort_column = {"Popularity": "popularity", "Average rating": "avg_rating", "Release year": "release_year", "Title": "title"}[sort_by]
filtered = filtered.sort_values(sort_column, ascending=(sort_by == "Title"), na_position="last")

page_size = 24
signature = (query, tuple(selected_genres), year_range, rating_range, sort_by)
if st.session_state.get("explore_signature") != signature:
    st.session_state["explore_signature"] = signature
    st.session_state["explore_page"] = 0
page = int(st.session_state.get("explore_page", 0))
start = page * page_size
st.caption(f"{len(filtered):,} movies · Showing {start + 1 if len(filtered) else 0}–{min(start + page_size, len(filtered))}")
render_movie_grid(filtered.iloc[start:start + page_size], profile_id, columns=4,
                  section=f"explore-page-{page}")
left, center, right = st.columns([1, 2, 1])
with left:
    if st.button("← Previous", key=widget_key("previous-page", "explore"),
                 disabled=page == 0, width="stretch"):
        st.session_state["explore_page"] = max(0, page - 1)
        st.rerun()
with center:
    st.caption(f"Page {page + 1} of {max(1, (len(filtered) + page_size - 1) // page_size)}")
with right:
    if st.button("Next →", key=widget_key("next-page", "explore"),
                 disabled=start + page_size >= len(filtered), width="stretch"):
        st.session_state["explore_page"] = page + 1
        st.rerun()

