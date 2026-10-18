import streamlit as st

st.set_page_config(page_title="Favourites | CineMatch", page_icon="🎬", layout="wide")

from src import database
from src.data_loader import DatasetError
from src.ui import load_page_data, render_movie_grid, setup_page, show_dataset_setup
from src.widget_keys import widget_key

profile_id = setup_page("Favourites")
try:
    movies, _, _, _ = load_page_data()
except DatasetError:
    show_dataset_setup()
    st.stop()

st.title("My favourites")
saved = database.get_collection("favourites", profile_id)
saved_ids = {str(row["movie_id"]) for row in saved}
items = movies[movies["movie_id"].astype(str).isin(saved_ids)].copy()
sort_by = st.selectbox("Sort", ["Recently added", "Title", "Rating"],
                       key=widget_key("sort", "favourites"))
if sort_by == "Title":
    items = items.sort_values("title")
elif sort_by == "Rating":
    items = items.sort_values("avg_rating", ascending=False, na_position="last")
else:
    order = {str(row["movie_id"]): index for index, row in enumerate(saved)}
    items["_order"] = items["movie_id"].astype(str).map(order)
    items = items.sort_values("_order")
st.caption(f"{len(items)} saved movies")
if items.empty:
    st.info("No favourites yet. Save a movie from a card or its details page.")
else:
    render_movie_grid(items.drop(columns=["_order"], errors="ignore"), profile_id, columns=4,
                      section="favourites-list")

