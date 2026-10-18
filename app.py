import streamlit as st
import pandas as pd
from config.settings import PROCESSED_DATA_DIR

st.set_page_config(page_title="CineMatch | Movie discovery", page_icon="🎬", layout="wide")

from src import database
from src.data_loader import DatasetError, format_year
from src.recommender import recommend
from src.ui import (data_files_mtime, get_genre_counts, load_page_data, profile_ratings,
                    render_movie_grid, render_poster, setup_page, show_dataset_setup,
                    recommendations_frame)
from src.widget_keys import widget_key

profile_id = setup_page("Home")
try:
    movies, content_model, collab_model, model_message = load_page_data()
except DatasetError:
    show_dataset_setup()
    st.stop()

st.markdown("<div class='cine-kicker'>Movie discovery</div>", unsafe_allow_html=True)
profile_name = next((item["name"] for item in database.list_profiles() if int(item["profile_id"]) == profile_id), "there")
st.title(f"Welcome, {profile_name}.")
st.markdown("Browse movies, save titles to your watchlist, and get suggestions based on what you rate.")
st.caption(model_message)

query_col, genre_col, action_col = st.columns([3, 2, 1])
with query_col:
    query = st.text_input("Search the catalog", placeholder="Search by title…", label_visibility="collapsed",
                          key=widget_key("search-title", "home"))
with genre_col:
    genre_counts = get_genre_counts(str(PROCESSED_DATA_DIR), data_files_mtime()[0])
    all_genres = sorted(genre_counts)
    chosen_genre = st.selectbox("Genre", ["All genres"] + all_genres, label_visibility="collapsed",
                                key=widget_key("filter-genre", "home"))
with action_col:
    if st.button("Explore all", key=widget_key("explore-all", "home"), width="stretch"):
        st.switch_page("pages/1_Explore.py")

featured_pool = movies.copy()
if "popularity" in featured_pool:
    featured_pool["popularity"] = featured_pool["popularity"].fillna(0)
featured = featured_pool.sort_values(["popularity", "rating_count"], ascending=False).iloc[0]
hero_left, hero_right = st.columns([1, 2.3], gap="large")
with hero_left:
    render_poster(featured)
with hero_right:
    st.markdown("<div class='cine-kicker'>Featured from your catalog</div>", unsafe_allow_html=True)
    year_value = format_year(featured.get("release_year"))
    year = f" · {year_value}" if year_value else ""
    st.header(f"{featured['title']}{year}")
    overview = str(featured.get("overview", "") or "").strip()
    st.write(overview[:650] + ("…" if len(overview) > 650 else "") if overview else "Explore this film and discover what viewers with similar tastes enjoyed.")
    if st.button("Open featured movie", type="primary", key=widget_key("open-featured", "home")):
        st.session_state["selected_movie_id"] = str(featured["movie_id"])
        st.switch_page("pages/7_Movie_Details.py")

if query.strip() or chosen_genre != "All genres":
    filtered = movies.copy()
    if query.strip():
        filtered = filtered[filtered["title"].astype(str).str.contains(query.strip(), case=False, regex=False)]
    if chosen_genre != "All genres":
        filtered = filtered[filtered["genres"].fillna("").str.contains(chosen_genre, case=False, regex=False)]
    st.subheader("Search results")
    render_movie_grid(filtered.head(8), profile_id, columns=4, section="home-search-results")
    st.divider()

local_ratings = profile_ratings(profile_id)
excluded = set(local_ratings)
excluded.update(str(item["movie_id"]) for item in database.get_collection("watch_history", profile_id))
personalized = recommend(movies, content_model, collab_model, mode="hybrid", profile_ratings=local_ratings,
                         excluded_ids=excluded, limit=6) if content_model or collab_model else []
st.subheader("Recommended for you")
if not local_ratings:
    st.caption("Cold start: these picks use catalog ratings and popularity. Rate movies to personalize future recommendations.")
elif collab_model and collab_model.get("ratings_count", 0):
    st.caption("Combines movie details with patterns in the dataset ratings.")
elif any(item.get("method") == "content fallback" for item in personalized):
    st.caption("Dataset ratings are unavailable, so these picks are based on movie details and your ratings.")
else:
    st.caption("Dataset ratings are unavailable, so these picks are based on catalog popularity.")
render_movie_grid(recommendations_frame(movies, personalized), profile_id, columns=4,
                  section="home-recommended",
                  reasons={item["movie_id"]: item["reason"] for item in personalized})

if local_ratings and content_model:
    local_entries = database.get_collection("ratings", profile_id)
    liked_entry = next((entry for entry in local_entries if float(entry["rating"]) >= 4.0), None)
    if liked_entry:
        liked_id = str(liked_entry["movie_id"])
        liked_title = next((str(row["title"]) for _, row in movies[movies["movie_id"].astype(str).eq(liked_id)].iterrows()), "a movie you liked")
        because = recommend(movies, content_model, collab_model, mode="content", profile_ratings=local_ratings,
                            seed_movie_id=liked_id, excluded_ids=excluded, limit=4)
        st.subheader(f"Because you liked {liked_title}")
        render_movie_grid(recommendations_frame(movies, because), profile_id, columns=4,
                          section="home-because-you-liked",
                          reasons={item["movie_id"]: item["reason"] for item in because})

min_votes = max(5, int(movies["rating_count"].fillna(0).quantile(0.50))) if movies["rating_count"].notna().any() else 5
top_rated = movies[movies["rating_count"].fillna(0) >= min_votes].sort_values("avg_rating", ascending=False).head(4)
if top_rated.empty:
    top_rated = movies.sort_values(["avg_rating", "rating_count"], ascending=False).head(4)
st.subheader("Top rated")
st.caption(f"Minimum rating count: {min_votes} · Scores are from the supplied dataset.")
render_movie_grid(top_rated, profile_id, columns=4, section="home-top-rated", show_actions=False)

if genre_counts:
    featured_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:3]
    st.subheader("Popular by genre")
    for genre in featured_genres:
        section = movies[movies["genres"].fillna("").str.contains(genre, case=False, regex=False)]
        section = section.sort_values(["rating_count", "avg_rating"], ascending=False).head(4)
        with st.expander(f"{genre} · {genre_counts[genre]:,} movies", expanded=False):
            render_movie_grid(section, profile_id, columns=4, section=f"home-genre-{genre}", show_actions=False)

years = pd.to_numeric(movies["release_year"], errors="coerce")
if years.notna().any():
    recent_year = int(years.max())
    recent = movies[years >= recent_year - 3].sort_values("release_year", ascending=False).head(4)
    st.subheader("Recently released in this catalog")
    st.caption(f"Newest available release year: {recent_year}")
    render_movie_grid(recent, profile_id, columns=4, section="home-recent", show_actions=False)

