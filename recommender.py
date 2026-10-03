from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


MODEL_PATH = Path(__file__).parent / "models" / "movie_recommender.joblib"


def load_recommender(model_path=MODEL_PATH):
    """Load the saved TF-IDF recommender components."""
    return joblib.load(model_path)


def _normalize_title(title):
    return " ".join(str(title).casefold().split())


def recommend_movies(movie_title, n=5, model=None):
    """Return similar movies by title, or a message if the title is missing."""
    if model is None:
        model = load_recommender()

    movie_key = _normalize_title(movie_title)
    title_to_index = model["title_to_index"]
    if movie_key not in title_to_index:
        return "Movie not found in the dataset."
    if n <= 0:
        return pd.DataFrame(columns=["title", "genres", "similarity"])

    movie_index = title_to_index[movie_key]
    scores = cosine_similarity(
        model["tfidf_matrix"][movie_index], model["tfidf_matrix"]
    ).ravel()
    ranked_indices = scores.argsort()[::-1]
    ranked_indices = [index for index in ranked_indices if index != movie_index][:n]

    results = pd.DataFrame({
        "title": [model["titles"][index] for index in ranked_indices],
        "genres": [model["genres"][index] for index in ranked_indices],
        "similarity": scores[ranked_indices],
    })
    return results
