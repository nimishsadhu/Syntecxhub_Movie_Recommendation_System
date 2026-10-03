# Movie Recommendation System

## Project Overview

This project recommends movies similar to a title selected by the user. It uses the movie metadata in the supplied CSV, TF-IDF text vectors, and cosine similarity.

## Dataset

The provided CSV contains 4,803 rows and 20 columns. It has 3,941 missing cells, 0 exact duplicate rows, and no blank movie titles. The most incomplete original fields are `homepage` (3,091 missing) and `tagline` (844 missing). The source is described only as a dataset provided for this project.

The file includes movie-level `vote_average` and `vote_count`, but has no user IDs or per-user movie ratings. For that reason, this project uses content-based filtering instead of collaborative filtering.

## Features Used

The recommender combines these available text fields:

- `genres`, parsed into genre names
- `keywords`, parsed into keyword names
- `overview`
- `tagline`

The CSV has 20 distinct genre labels. Genre, keyword, overview, and tagline coverage are 4,775, 4,391, 4,800, and 3,959 rows respectively. Director and cast fields are not present in this CSV.

## Data Preprocessing

Exact duplicate rows and rows without a title are removed if found. There were 0 exact duplicate rows and 0 missing or blank titles, so the cleaned data retains all 4,803 movie rows. Missing overview and tagline text is replaced with empty strings. Genres and keywords are parsed into readable name columns; numeric fields and release dates are converted, and `release_year` is derived. The cleaned file is `data/cleaned_data.csv`.

## EDA

The notebooks show missing values and numeric summaries, the most common genre labels, release counts by year, runtime, and vote-average distributions. Genre counts include every genre label attached to a movie. The three most frequent genres are Drama (2,297), Comedy (1,722), Thriller (1,274). Release years in the source range from 1916 to 2017. The runtime plot is limited to 1–300 minutes for readability; it does not remove rows from the dataset.

Generated plots are saved in `figures/` and also appear in the executed EDA notebook.

## Recommendation Approach

1. Use genres, keywords, overview, and tagline as movie metadata.
2. Join the fields into one text string per movie and normalize case and whitespace.
3. Use `TfidfVectorizer` to represent the text numerically.
4. Compare one movie vector with the catalog using cosine similarity.
5. Return the highest scoring movies, excluding the selected movie itself.

Movie title matching ignores capitalization and extra spaces. If a title appears more than once, the first matching dataset row is used. No similarity matrix is stored for every pair; scores are computed for the requested movie.

## Project Structure

```text
movie-recommendation-system/
├── data/
│   ├── original_dataset.csv
│   └── cleaned_data.csv
├── figures/
│   ├── movies_by_year.png
│   ├── runtime_and_vote_average.png
│   └── top_genres.png
├── models/
│   └── movie_recommender.joblib
├── notebooks/
│   ├── inspection.ipynb
│   ├── preprocessing.ipynb
│   ├── eda.ipynb
│   └── recommendation.ipynb
├── recommender.py
├── README.md
└── requirements.txt
```

## Workflow

```text
Provided CSV → Inspection → Preprocessing → EDA → Metadata text
→ TF-IDF → Cosine similarity → Recommendations → Qualitative review
→ Saved recommender
```

## Sample Recommendations

These results were generated from the saved model built from the supplied CSV:

**The Dark Knight**
1. The Dark Knight Rises (cosine similarity 0.4384)
2. Batman Returns (cosine similarity 0.3605)
3. Batman Forever (cosine similarity 0.3526)
4. Batman Begins (cosine similarity 0.3396)
5. Batman: The Dark Knight Returns, Part 2 (cosine similarity 0.3380)

**Toy Story**
1. Toy Story 3 (cosine similarity 0.5053)
2. Toy Story 2 (cosine similarity 0.4662)
3. Small Soldiers (cosine similarity 0.2753)
4. The 40 Year Old Virgin (cosine similarity 0.2712)
5. Factory Girl (cosine similarity 0.1871)

**The Hangover**
1. Last Vegas (cosine similarity 0.2545)
2. The Hangover Part II (cosine similarity 0.2314)
3. Very Bad Things (cosine similarity 0.2090)
4. What Happens in Vegas (cosine similarity 0.1410)
5. Doug's 1st Movie (cosine similarity 0.1257)

The recommendation notebook also runs examples for `The Shawshank Redemption` and `The Conjuring`, and checks shared genres alongside similarity scores.

## Evaluation

Evaluation is qualitative: the notebook displays several queries, their top recommendations, cosine scores, and any genres shared with the input movie. The CSV has aggregate votes but no user-movie relevance labels, so precision or recall cannot be measured from this data alone. Similarity scores indicate text closeness, not recommendation accuracy.

## Installation

From this project directory, install the listed packages:

```bash
pip install -r requirements.txt
```

## Running the Project

Open and run the notebooks in this order:

1. `notebooks/inspection.ipynb`
2. `notebooks/preprocessing.ipynb`
3. `notebooks/eda.ipynb`
4. `notebooks/recommendation.ipynb`

The notebooks use paths relative to the project root. The provided dataset copy is in `data/original_dataset.csv`.

## Saved Model

The model and its TF-IDF components are saved at `models/movie_recommender.joblib`. After installing the requirements, use the helper module from the project directory:

```python
from recommender import load_recommender, recommend_movies

model = load_recommender()
recommendations = recommend_movies("The Dark Knight", 5, model)
print(recommendations)
```

The module returns a message if the title is not in the dataset. The executed recommendation notebook reloads the saved model and generates recommendations from it.

## Limitations

- Recommendations depend on the quality and coverage of the available metadata. Some movies have no genre labels, and many have no tagline.
- The system compares movie content and does not personalize results for an individual viewer.
- The dataset has no per-user ratings or relevance labels, so a supervised ranking metric is not available.
- Repeated title names are possible; a title search uses the first match.

## Future Improvements

- Add cast or director metadata if a compatible source becomes available.
- Collect user feedback or interaction data to evaluate personalized recommendations.
