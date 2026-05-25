# ══════════════════════════════════════════════════════════════
# SPOTIFY INFORMATION RETRIEVAL
# ══════════════════════════════════════════════════════════════


# ── Install libraries ────────────────────────────────
!pip install pandas numpy scikit-learn rank_bm25 matplotlib seaborn --quiet

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import TSNE
from rank_bm25 import BM25Okapi

print("All libraries imported successfully!")


# ──  Load dataset ─────────────────────────────────────

df = pd.read_csv('dataset[1].csv')
print('Raw dataset shape:', df.shape)
print(df.head(3))


# ──  Clean data and build combined text ───────────────
df.dropna(subset=['track_name', 'artists'], inplace=True)
df.drop_duplicates(subset='track_id', inplace=True)
df.reset_index(drop=True, inplace=True)

# Expanding short genre codes into descriptive phrases
# so TF-IDF has richer text to work with
genre_map = {
    'j-dance': 'japanese dance electronic',
    'j-pop':   'japanese pop',
    'j-rock':  'japanese rock',
    'k-pop':   'korean pop',
    'dance':   'dance electronic pop',
    'acoustic':'acoustic folk soft',
    'pop':     'pop mainstream',
    'hip-hop': 'hip hop rap',
    'rock':    'rock guitar band',
    'classical':'classical instrumental orchestra',
    'r-n-b':   'rnb soul rhythm blues',
    'indie':   'indie alternative',
    'jazz':    'jazz blues soul',
    'metal':   'metal heavy guitar',
    'edm':     'electronic dance music',
}

def expand_genre(genre):
    genre = str(genre).lower().strip()
    return genre_map.get(genre, genre)

df['genre_expanded'] = df['track_genre'].apply(expand_genre)

df['combined_text'] = (
    df['track_name'].fillna('') + ' ' +
    df['artists'].fillna('') + ' ' +
    df['genre_expanded'] + ' ' +
    df['genre_expanded']
)

print('Cleaned dataset shape:', df.shape)
print('\nSample combined text:')
print(df['combined_text'].head(3).to_string())


# ── TF-IDF Vectorisation ─────────────────────────────
sample_df = df.sample(20000, random_state=42).reset_index(drop=True)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1, 2),
    sublinear_tf=True       # log-scaled TF for better score spread
)
tfidf_matrix = vectorizer.fit_transform(sample_df['combined_text'])
print(f'TF-IDF matrix shape: {tfidf_matrix.shape}')


# ──  TF-IDF Search with combined scoring ───────────────
# Score = 0.7 × TF-IDF relevance
#       + 0.2 × popularity (normalised)
#       + 0.1 × energy


def tfidf_search(query, top_n=5):
    query_vec = vectorizer.transform([query])
    tfidf_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    results = sample_df.copy()
    results['tfidf_score'] = tfidf_scores

    pop_max = results['popularity'].max()
    pop_min = results['popularity'].min()
    results['pop_norm'] = (results['popularity'] - pop_min) / (pop_max - pop_min + 1e-9)

    results['combined_score'] = (
        0.70 * results['tfidf_score'] +
        0.20 * results['pop_norm'] +
        0.10 * results['energy']
    )

    results = results[results['tfidf_score'] > 0]
    results = results.sort_values(by='combined_score', ascending=False)
    return results[['track_name', 'artists', 'track_genre',
                     'tfidf_score', 'popularity', 'combined_score']].head(top_n)

print('--- TF-IDF Search: "dance pop" ---')
print(tfidf_search('dance pop').to_string(index=False))

print('\n--- TF-IDF Search: "acoustic love" ---')
print(tfidf_search('acoustic love').to_string(index=False))

print('\n--- TF-IDF Search: "hip hop" ---')
print(tfidf_search('hip hop').to_string(index=False))

print('\n--- TF-IDF Search: "rock guitar" ---')
print(tfidf_search('rock guitar').to_string(index=False))


# ── BM25 Search ──────────────────────────────────────
def preprocess_for_bm25(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1]
    return tokens

tokenized_corpus = [preprocess_for_bm25(t) for t in sample_df['combined_text']]
bm25 = BM25Okapi(tokenized_corpus)
print('BM25 model ready!')

def bm25_search(query, top_n=5):
    tokenized_query = preprocess_for_bm25(query)
    scores = bm25.get_scores(tokenized_query)
    results = sample_df.copy()
    results['score'] = scores
    results = results[results['score'] > 0]
    results = results.sort_values(by='score', ascending=False)
    return results[['track_name', 'artists', 'track_genre', 'score']].head(top_n)

print('\n--- BM25 Search: "dance pop" ---')
print(bm25_search('dance pop').to_string(index=False))

print('\n--- BM25 Search: "rock guitar" ---')
print(bm25_search('rock guitar').to_string(index=False))


# ── FIXED Content-Based Recommender ──────────────────
# BUG IN ORIGINAL: MinMaxScaler + cosine similarity
# gives ~0.999 for every song because all vectors point
# in nearly the same direction after scaling.
# FIX: StandardScaler + Euclidean distance

AUDIO_FEATURES = [
    'danceability', 'energy', 'acousticness',
    'valence', 'tempo', 'instrumentalness', 'speechiness'
]

scaler = StandardScaler()
feature_matrix = scaler.fit_transform(sample_df[AUDIO_FEATURES].fillna(0))
sample_df_features = pd.DataFrame(feature_matrix, columns=AUDIO_FEATURES)

def content_recommend(liked_songs, top_k=5):
    mask = sample_df['track_name'].isin(liked_songs)
    liked_idx = sample_df[mask].index.tolist()

    if not liked_idx:
        print('Songs not found in sample. Try different names.')
        return None

    user_profile = sample_df_features.iloc[liked_idx].mean().values.reshape(1, -1)
    distances = euclidean_distances(user_profile, sample_df_features.values).flatten()

    results = sample_df.copy()
    results['distance'] = distances
    results = results[~results['track_name'].isin(liked_songs)]
    results = results.sort_values(by='distance', ascending=True)
    results['similarity_score'] = 1 / (1 + results['distance'])

    return results[['track_name', 'artists', 'track_genre', 'similarity_score']].head(top_k)

print('\n--- Recommendations for: Blinding Lights, Levitating ---')
recs = content_recommend(['Blinding Lights', 'Levitating'])
if recs is not None:
    print(recs.to_string(index=False))

print('\n--- Recommendations for: Someone Like You, The Night We Met ---')
recs2 = content_recommend(['Someone Like You', 'The Night We Met'])
if recs2 is not None:
    print(recs2.to_string(index=False))


# ──Precision@5 Evaluation ───────────────────────────
def evaluate_precision_at_k(liked_songs, expected_genre, k=5):
    recs = content_recommend(liked_songs, top_k=k)
    if recs is None:
        return 0.0
    relevant = recs['track_genre'].str.contains(expected_genre, case=False, na=False)
    precision = relevant.sum() / k
    print(f'Precision@{k} for "{expected_genre}": {precision:.2f}  ({relevant.sum()}/{k} matched)')
    return precision

print('\n=== Evaluation ===')
evaluate_precision_at_k(['Blinding Lights', 'Levitating'], expected_genre='pop')
evaluate_precision_at_k(['Someone Like You', 'The Night We Met'], expected_genre='acoustic')


# ── Radar Chart ───────────────────────────────────────
def plot_radar_chart(liked_songs, rec_songs=None):
    features = ['danceability', 'energy', 'acousticness',
                'valence', 'instrumentalness', 'speechiness']
    N = len(features)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#ffffff')

    liked_data = sample_df[sample_df['track_name'].isin(liked_songs)][features]
    if not liked_data.empty:
        profile = liked_data.mean().values.tolist()
        profile += profile[:1]
        ax.plot(angles, profile, 'o-', linewidth=2, color='#1DB954', label='Your taste profile')
        ax.fill(angles, profile, alpha=0.2, color='#1DB954')

    if rec_songs:
        rec_data = sample_df[sample_df['track_name'].isin(rec_songs)][features]
        if not rec_data.empty:
            rec_profile = rec_data.mean().values.tolist()
            rec_profile += rec_profile[:1]
            ax.plot(angles, rec_profile, 'o-', linewidth=2, color='#FF6B6B', label='Recommendations')
            ax.fill(angles, rec_profile, alpha=0.15, color='#FF6B6B')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, size=11)
    ax.set_ylim(0, 1)
    ax.set_title('Audio Feature Radar Chart', size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig('radar_chart.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: radar_chart.png')

plot_radar_chart(
    liked_songs=['Blinding Lights', 'Levitating'],
    rec_songs=['Save Your Tears', "Don't Start Now", 'Physical']
)


# ──t-SNE Genre Scatter Plot ────────────────────────
print('Running t-SNE (takes ~1 minute)...')

top_genres = sample_df['track_genre'].value_counts().head(6).index.tolist()
plot_df = sample_df[sample_df['track_genre'].isin(top_genres)].sample(2000, random_state=42)
plot_features = scaler.transform(plot_df[AUDIO_FEATURES].fillna(0))

tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
tsne_result = tsne.fit_transform(plot_features)

plot_df = plot_df.reset_index(drop=True)
plot_df['tsne_x'] = tsne_result[:, 0]
plot_df['tsne_y'] = tsne_result[:, 1]

palette = ['#1DB954','#FF6B6B','#4A90D9','#F5A623','#9B59B6','#E67E22']
colors = {genre: palette[i] for i, genre in enumerate(top_genres)}

fig, ax = plt.subplots(figsize=(10, 7))
for genre in top_genres:
    subset = plot_df[plot_df['track_genre'] == genre]
    ax.scatter(subset['tsne_x'], subset['tsne_y'],
               c=colors[genre], label=genre, alpha=0.5, s=15)

ax.set_title('t-SNE: Songs Clustered by Audio Features', fontsize=13, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.legend(title='Genre', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('tsne_genre_clusters.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: tsne_genre_clusters.png')

print('\n✓ All done! Upload to GitHub: notebook + radar_chart.png + tsne_genre_clusters.png')
