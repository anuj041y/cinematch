import streamlit as st

st.set_page_config(page_title="Watch history | CineMatch", page_icon="🎬", layout="wide")

from src import database
from src.data_loader import DatasetError, format_title
from src.ui import load_page_data, render_poster, setup_page, show_dataset_setup
from src.widget_keys import widget_key
import pandas as pd

profile_id = setup_page("History")
try:
    movies, _, _, _ = load_page_data()
except DatasetError:
    show_dataset_setup()
    st.stop()

st.title("Watch history")
st.caption("This is a history you maintain manually. CineMatch does not track streaming or viewing activity.")
entries = database.get_collection("watch_history", profile_id)
by_id = movies.set_index(movies["movie_id"].astype(str), drop=False)
ratings_by_id = {str(row["movie_id"]): float(row["rating"]) for row in database.get_collection("ratings", profile_id)}
if not entries:
    st.info("No movies marked as watched yet.")
visible = 0
for entry in entries:
    movie_id = str(entry["movie_id"])
    if movie_id not in by_id.index:
        continue
    movie = by_id.loc[movie_id]
    visible += 1
    with st.container(border=True):
        left, center, right = st.columns([1, 4, 2], vertical_alignment="center")
        with left:
            render_poster(movie)
        with center:
            st.subheader(format_title(movie.get("title")))
            st.caption(f"Marked watched · {str(entry['watched_at']).replace('T', ' ')[:16]} UTC")
            current = ratings_by_id.get(movie_id)
            st.write(f"Your rating: {current:g}/5" if current is not None else "You have not rated this movie yet.")
        with right:
            if st.button("View details", key=widget_key("view-details", "history", profile_id, movie_id),
                         width="stretch"):
                st.session_state["selected_movie_id"] = movie_id
                st.switch_page("pages/7_Movie_Details.py")
            with st.form(widget_key("rating-form", "history", profile_id, movie_id)):
                current_value = int(round(ratings_by_id[movie_id])) if movie_id in ratings_by_id else 3
                updated_rating = st.select_slider("Your rating", options=[1, 2, 3, 4, 5], value=current_value,
                                                  key=widget_key("select-rating", "history", profile_id, movie_id))
                save_rating = st.form_submit_button("Save rating", width="stretch",
                                                    key=widget_key("save-rating", "history", profile_id, movie_id))
            if save_rating:
                database.set_rating(profile_id, movie_id, updated_rating)
                st.rerun()
            if st.button("Remove from history", key=widget_key("remove-history", "history", profile_id, movie_id),
                         width="stretch"):
                database.mark_watched(profile_id, movie_id, False)
                st.rerun()
if entries and not visible:
    st.info("These history entries are not present in the current movie catalog.")

