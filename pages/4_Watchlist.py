import streamlit as st

st.set_page_config(page_title="Watchlist | CineMatch", page_icon="🎬", layout="wide")

from src import database
from src.data_loader import DatasetError, format_genres, format_title, format_year
from src.ui import load_page_data, render_poster, setup_page, show_dataset_setup
from src.widget_keys import widget_key

profile_id = setup_page("Watchlist")
try:
    movies, _, _, _ = load_page_data()
except DatasetError:
    show_dataset_setup()
    st.stop()

st.title("My watchlist")
entries = database.get_collection("watchlist", profile_id)
watchlist_ids = {str(row["movie_id"]) for row in entries}
items = movies[movies["movie_id"].astype(str).isin(watchlist_ids)].copy()
st.caption(f"{len(items)} movies saved to watch")
if items.empty:
    st.info("Your watchlist is empty. Use the Watchlist button on any movie card or details page.")
for _, movie in items.iterrows():
    movie_id = str(movie["movie_id"])
    left, center, right = st.columns([1, 5, 2], vertical_alignment="center")
    with left:
        render_poster(movie)
    with center:
        st.subheader(format_title(movie.get("title")))
        facts = [value for value in (format_year(movie.get("release_year")), format_genres(movie.get("genres"))) if value]
        st.caption(" · ".join(facts) if facts else "Movie details are unavailable.")
    with right:
        if st.button("View details", key=widget_key("view-details", "watchlist", profile_id, movie_id), width="stretch"):
            st.session_state["selected_movie_id"] = movie_id
            st.switch_page("pages/7_Movie_Details.py")
        if st.button("Mark watched", key=widget_key("mark-watched", "watchlist", profile_id, movie_id),
                     type="primary", width="stretch"):
            database.mark_watched(profile_id, movie_id)
            st.rerun()
        if st.button("Remove", key=widget_key("remove-watchlist", "watchlist", profile_id, movie_id),
                     width="stretch"):
            database.set_watchlist(profile_id, movie_id, False)
            st.rerun()

