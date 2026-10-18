import streamlit as st

st.set_page_config(page_title="Movie details | CineMatch", page_icon="🎬", layout="wide")

from src import database
from src.data_loader import DatasetError
from src.recommender import recommend
from src.ui import (load_page_data, profile_ratings, render_details,
                    render_movie_grid, setup_page, show_dataset_setup, recommendations_frame)

profile_id = setup_page("Movie details")
try:
    movies, content_model, collaborative_model, _ = load_page_data()
except DatasetError:
    show_dataset_setup()
    st.stop()

render_details(movies, profile_id)
movie_id = str(st.session_state.get("selected_movie_id", ""))
if movie_id and (movies["movie_id"].astype(str) == movie_id).any():
    local_ratings = profile_ratings(profile_id)
    excluded = set(local_ratings)
    excluded.update(str(row["movie_id"]) for row in database.get_collection("watch_history", profile_id))
    similar = recommend(movies, content_model, collaborative_model, mode="content",
                        seed_movie_id=movie_id, profile_ratings=local_ratings,
                        excluded_ids=excluded, limit=8)
    personalized = recommend(movies, content_model, collaborative_model, mode="hybrid",
                              seed_movie_id=movie_id, profile_ratings=local_ratings,
                              excluded_ids=excluded, limit=8)
    st.subheader("Similar movies")
    render_movie_grid(recommendations_frame(movies, similar), profile_id, columns=4,
                      section="details-similar",
                      reasons={item["movie_id"]: item["reason"] for item in similar})
    st.subheader("Personalized picks related to this movie")
    render_movie_grid(recommendations_frame(movies, personalized), profile_id, columns=4,
                      section="details-personalized",
                      reasons={item["movie_id"]: item["reason"] for item in personalized})

