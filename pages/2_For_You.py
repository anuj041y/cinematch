import streamlit as st

st.set_page_config(page_title="For You | CineMatch", page_icon="🎬", layout="wide")

from src import database
from src.data_loader import DatasetError
from src.recommender import recommend
from src.ui import (load_page_data, profile_ratings, render_movie_grid,
                    render_onboarding, setup_page, show_dataset_setup, recommendations_frame)
from src.widget_keys import widget_key

profile_id = setup_page("For You")
try:
    movies, content_model, collaborative_model, model_message = load_page_data()
except DatasetError:
    show_dataset_setup()
    st.stop()

st.title("For you")
st.caption(model_message)
render_onboarding(movies, profile_id)
local_ratings = profile_ratings(profile_id)
model_label = st.selectbox("Recommendation method", ["Hybrid", "Content-based", "Collaborative filtering"],
                           key=widget_key("select-method", "for-you"))
mode = {"Hybrid": "hybrid", "Content-based": "content", "Collaborative filtering": "collaborative"}[model_label]
alpha = 0.5
if mode == "hybrid":
    stored_alpha = float(database.get_preference(profile_id, "hybrid_alpha", "0.5") or 0.5)
    alpha = st.slider("Content weight", min_value=0.0, max_value=1.0, value=min(1.0, max(0.0, stored_alpha)), step=0.05,
                      help="This weight applies to movie details. The rest is used for dataset ratings when available.",
                      key=widget_key("content-weight", "for-you", profile_id))
    if alpha != stored_alpha:
        database.set_preference(profile_id, "hybrid_alpha", str(alpha))
    st.progress(alpha, text=f"Content {alpha:.0%} · Collaborative {1-alpha:.0%}")
seed_options = [None] + movies.sort_values("rating_count", ascending=False).head(1000)["movie_id"].astype(str).tolist()
title_by_id = dict(zip(movies["movie_id"].astype(str), movies["title"].astype(str)))
seed = st.selectbox("Base recommendations on a movie (optional)", seed_options,
                    format_func=lambda movie_id: "Your profile" if movie_id is None else title_by_id.get(movie_id, movie_id),
                    index=0, key=widget_key("seed-movie", "for-you", profile_id))
exclude_existing = st.checkbox("Hide movies I have already rated, watched, or saved", value=True,
                                key=widget_key("exclude-existing", "for-you", profile_id))
excluded = set()
if exclude_existing:
    excluded.update(local_ratings)
    excluded.update(str(row["movie_id"]) for row in database.get_collection("watch_history", profile_id))
    excluded.update(str(row["movie_id"]) for row in database.get_collection("watchlist", profile_id))
recommendations = recommend(movies, content_model, collaborative_model, mode=mode,
                            profile_ratings=local_ratings, seed_movie_id=seed,
                            excluded_ids=excluded, alpha=alpha, limit=24)
if not local_ratings and seed is None:
    st.info("These are cold-start picks based on catalog ratings and popularity. Rate at least two movies above to receive personalized results.")
if mode == "hybrid" and (not collaborative_model or not collaborative_model.get("ratings_count")):
    if any(item.get("method") == "content fallback" for item in recommendations):
        st.info("This catalog has no linked user ratings yet. These suggestions use movie details and your ratings.")
    else:
        st.info("This catalog has no linked user ratings yet. These suggestions are based on popularity.")
st.caption("Scores rank candidates; they are not calibrated probabilities that you will like a film.")
if mode == "collaborative" and (not collaborative_model or not collaborative_model.get("ratings_count")):
    st.warning("Collaborative recommendations need linked user ratings. For The Movies Dataset, add its `links.csv` file to `data/raw/`, then prepare the data and retrain the models.")
reason_map = {item["movie_id"]: f"{item['reason']} · Ranking score {item['score']:.2f}" for item in recommendations}
render_movie_grid(recommendations_frame(movies, recommendations), profile_id, columns=4,
                  section="for-you-results", reasons=reason_map)

