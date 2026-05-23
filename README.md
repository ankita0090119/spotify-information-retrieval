# spotify-information-retrieval
A full end-to-end Information Retrieval system built on real Spotify data, implementing TF-IDF search, BM25 ranking, and a content-based song recommender — the same core techniques used by Spotify in production.

# system used
TF-IDF Search-Finds songs matching a text query using term frequency scoring + popularity signal
BM25 Search-Improved ranking with term saturation and document length normalisation
Content Recommender-Recommends songs based on audio feature similarity (danceability, energy, valence, etc.)
Precision@5 Evaluation-Measures recommendation quality using a standard IR metric

## Tech used
Python, scikit-learn, rank-bm25, pandas, Google Colab

## Dataset

- **Source:** [Spotify Tracks Dataset — Kaggle](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- **Size:** ~114,000 songs across 125 genre categories
- **Key fields used:** `track_name`, `artists`, `track_genre`, `popularity`, `danceability`, `energy`, `valence`, `tempo`, `acousticness`, `instrumentalness`, `speechiness`
