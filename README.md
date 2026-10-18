# CineMatch

CineMatch is a Streamlit app for browsing movies and getting recommendations. It uses movie details and personal ratings to suggest films, and saves favourites, watchlists, and viewing history in a local SQLite database.

## Start the app

Use Windows with Python 3.11 or 3.12. Open this folder in VS Code, open **Terminal → New Terminal**, and run:

```powershell
.\run_cinematch.bat
```

The first run creates `.venv` and installs the packages in `requirements.txt`. Later runs start the app directly. Open <http://localhost:8501> in a browser. Stop Streamlit with `Ctrl+C` in the terminal.

If you already have an activated virtual environment, you can start it with:

```powershell
python -m streamlit run app.py
```

## Movie data

The app reads its prepared catalog from `data/processed/movies_clean.csv`. The prepared catalog and model are included in this project copy, so the app can start without processing the raw CSVs.

The app does not load the large ratings file while browsing. Insights reads a small rating summary saved during model training, so the 751 MB local ratings file does not slow down page loads.

The raw files currently in `data/raw/` are `movies_metadata.csv`, `credits.csv`, and `ratings.csv`. The metadata and credits provide movie details, cast, directors, genres, and poster paths. The ratings file uses MovieLens movie IDs, while the catalog uses TMDB IDs. Preparation keeps valid ratings even without a mapping, but collaborative recommendations need the matching `links.csv` to connect those IDs to catalog movies. In this project copy, the processed ratings are retained locally under `ml:` IDs; no `links.csv` is present, so the app uses content recommendations until a mapping is added.

To rebuild from The Movies Dataset, put its `movies_metadata.csv`, `credits.csv`, `ratings.csv`, and `links.csv` in `data/raw/`, then run:

```powershell
python scripts/prepare_data.py
python scripts/train_models.py
```

The project also supports MovieLens data when `movies.csv` and `ratings.csv` are in `data/raw/`. Dataset downloads: [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) and [MovieLens](https://grouplens.org/datasets/movielens/).

## Recommendations

- **Content-based:** Uses plot, genres, keywords, cast, and director to find similar movies. Ratings you save in CineMatch can guide these matches.
- **Collaborative:** Uses patterns from mapped MovieLens user ratings.
- **Hybrid:** Combines content and collaborative scores when both are available.
- **Cold start:** Uses catalog ratings and popularity when there is not enough rating history.

Rated movies and movies in your viewing history are left out of your suggestions. A small genre adjustment helps prevent the list from being dominated by one genre.

## Posters

CineMatch keeps the original movie catalog and fills missing poster sources with local images matched by IMDb ID. This project includes 5,879 optimized posters from Kaggle's [Movie Posters dataset](https://www.kaggle.com/datasets/raman77768/movie-classifier), which is listed as CC0. Existing catalog poster links and recommendation data are kept. Movies with no matching image still show the bundled placeholder.

Page loading does not check poster links or call poster APIs, so a slow image server cannot block the app. To add posters to a fresh catalog, run:

```powershell
python scripts/merge_kaggle_posters.py
```

The script downloads the archive if needed, matches images to missing catalog entries by IMDb ID, converts matches to smaller WebP files, and updates `data/processed/movies_clean.csv`. The temporary download is removed after a successful merge. Use `--refresh` to search again after rebuilding the catalog, or pass a local ZIP with `--archive path\to\movie-posters.zip`.

For additional online replacements, set `WATCHMODE_API_KEY` or `TMDB_API_KEY` in `.env`. Watchmode matches use the movie title and release year, allowing a subtitle and common spelling variants.

To check every catalog poster link and replace broken or missing posters, run:

```powershell
python scripts/enrich_posters.py --repair-broken --all
```

Image links are checked concurrently and cached between runs. API searches run only for broken or missing posters and are capped at 100 per run. Run the same command again to continue without repeating the link scan. To process a smaller batch, use `--repair-broken --limit 50`. To repair selected titles, add one or more `--title "Movie title"` options. Verified matches are saved in `data/processed/movies_clean.csv`; titles with no provider match keep the placeholder. API quotas apply. Watchmode notes that some poster images come from third parties; review its [API terms](https://api.watchmode.com/tc) before publishing those images.

## Project files

```text
app.py                    Home page
pages/                    Explore, For You, favourites, watchlist, history, insights, movie details
src/data_loader.py        CSV cleanup and ID matching
src/recommender.py        Content and rating-based recommendation models
src/posters.py            Poster lookup and fallback images
src/database.py           Local profiles and saved activity
scripts/                  Data preparation, model training, poster enrichment and merge
assets/posters/           Local posters matched by IMDb ID
data/processed/           Prepared catalog and ratings
models/                   Saved recommendation model
tests/                    Unit tests
```

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Deploy a demo

The checked-in prepared catalog and model let the app start without raw CSV uploads or model training in the cloud. To deploy with Streamlit Community Cloud:

1. Push this project folder to a GitHub repository. Keep `data/processed/movies_clean.csv`, `models/cinematch_models.joblib`, and `models/metadata.json` in the repository. The 751 MB `ratings_clean.csv` stays local and is ignored by Git; this copy has no ID mapping, so it cannot contribute collaborative recommendations. `.gitignore` also excludes raw data, `.env`, and the local database.
2. Create a Community Cloud app for that repository and select `app.py` as the entrypoint. Choose Python 3.12.
3. Add `WATCHMODE_API_KEY` in the app's secrets if you want online poster lookups.

For collaborative recommendations, add the matching `links.csv`, prepare and retrain locally, then store the large cleaned ratings file in Git LFS or external storage and make it available to the app. Do not put the full ratings export in a normal Git commit.

The default SQLite database is local to the app instance. On a public deployment, saved profile changes are shared among visitors and may be cleared when the app restarts. Use a managed database and user sign-in for persistent personal accounts. See Streamlit's [deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [secrets guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management).
