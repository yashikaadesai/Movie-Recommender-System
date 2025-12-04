"""Recommender utilities for collaborative filtering algorithms.
Implements user-based and item-based collaborative filtering with
cosine similarity, along with functions to load data from MongoDB,
Create user-item matrices, and evaluate predictions using RMSE.
"""

from math import sqrt
from typing import Tuple, List
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix

load_dotenv()

def _ensure_spark_session():
    """Create or return an existing SparkSession. Requires `pyspark` to be installed."""
    try:
        from pyspark.sql import SparkSession
    except Exception as e:  # pragma: no cover - optional dependency
        raise ImportError("pyspark is required for Hadoop/Spark similarity functions") from e

    return SparkSession.builder.getOrCreate()


def compute_user_similarity_hadoop(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute user-user cosine similarities using Spark (suitable for Hadoop clusters).

    This transposes the user-item matrix to produce an item x user RowMatrix and
    calls `columnSimilarities()` which computes similarities between users (columns).
    Returns a pandas DataFrame indexed/columned by the original user ids.
    """
    spark = _ensure_spark_session()
    sc = spark.sparkContext

    # Ensure a numeric numpy array (items x users)
    mat = user_item_matrix.fillna(0).T.values

    rows = [Vectors.dense([float(x) for x in row]) for row in mat]
    rdd = sc.parallelize(rows)
    rm = RowMatrix(rdd)
    coord = rm.columnSimilarities()  # returns CoordinateMatrix

    # Build a dense pandas DataFrame to return
    user_ids = list(user_item_matrix.index)
    n = len(user_ids)
    sim_df = pd.DataFrame(0.0, index=user_ids, columns=user_ids)

    for entry in coord.entries.collect():
        i, j, v = int(entry.i), int(entry.j), float(entry.value)
        ui, uj = user_ids[i], user_ids[j]
        sim_df.at[ui, uj] = v
        sim_df.at[uj, ui] = v

    # set self-similarity to 1.0
    for u in user_ids:
        sim_df.at[u, u] = 1.0

    return sim_df


def compute_item_similarity_hadoop(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute item-item cosine similarities using Spark (suitable for Hadoop clusters).

    This treats the user-item matrix rows as users and computes columnSimilarities
    which returns similarity between items (columns).
    """
    spark = _ensure_spark_session()
    sc = spark.sparkContext

    mat = user_item_matrix.fillna(0).values
    rows = [Vectors.dense([float(x) for x in row]) for row in mat]
    rdd = sc.parallelize(rows)
    rm = RowMatrix(rdd)
    coord = rm.columnSimilarities()

    item_ids = list(user_item_matrix.columns)
    m = len(item_ids)
    sim_df = pd.DataFrame(0.0, index=item_ids, columns=item_ids)

    for entry in coord.entries.collect():
        i, j, v = int(entry.i), int(entry.j), float(entry.value)
        ii, jj = item_ids[i], item_ids[j]
        sim_df.at[ii, jj] = v
        sim_df.at[jj, ii] = v

    for it in item_ids:
        sim_df.at[it, it] = 1.0

    return sim_df


def load_ratings_from_db() -> pd.DataFrame:
    """Load ratings from MongoDB and coerce types.

    Returns a DataFrame with columns ['userId','movieId','rating','timestamp'].
    """
    uri = os.environ.get("MONGODB_URL")
    if not uri:
        raise ValueError("No MongoDB URI found in environment variables")

    client = MongoClient(uri)
    db = client["Movie-Recommender"]
    ratings = pd.DataFrame(list(db["ratings"].find()))

    # Coerce expected types when columns exist
    if "userId" in ratings.columns:
        ratings["userId"] = ratings["userId"].astype(int)
    if "movieId" in ratings.columns:
        ratings["movieId"] = ratings["movieId"].astype(int)
    if "rating" in ratings.columns:
        ratings["rating"] = ratings["rating"].astype(float)
    if "timestamp" in ratings.columns:
        ratings["timestamp"] = ratings["timestamp"].astype(int)

    return ratings


def create_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Build a user-item matrix with missing ratings filled by 0."""
    uim = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    return uim.fillna(0)


def compute_user_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """Return cosine similarity (users x users)."""
    sim = cosine_similarity(user_item_matrix)
    return pd.DataFrame(sim, index=user_item_matrix.index, columns=user_item_matrix.index)


def compute_item_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """Return cosine similarity (items x items)."""
    item_matrix = user_item_matrix.T
    sim = cosine_similarity(item_matrix)
    return pd.DataFrame(sim, index=item_matrix.index, columns=item_matrix.index)


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if not series.empty else 0.0


def predict_rating_user_based(
    user_id: int,
    movie_id: int,
    user_item_matrix: pd.DataFrame,
    user_similarity: pd.DataFrame,
    k: int = 5,
) -> float:
    """Predict rating using user-based collaborative filtering.

    Falls back to the user's mean when no neighbors provide information.
    """
    if user_id not in user_item_matrix.index:
        raise KeyError(f"User {user_id} not found in user-item matrix")

    # If movie not present, return user's mean as fallback
    if movie_id not in user_item_matrix.columns:
        return _safe_mean(user_item_matrix.loc[user_id])

    sims = user_similarity.loc[user_id].drop(user_id, errors="ignore")
    top_k = sims.sort_values(ascending=False).head(k)
    neighbors = top_k.index.tolist()

    ratings_top_k = user_item_matrix.loc[neighbors, movie_id]
    weights = top_k.loc[neighbors]

    if weights.sum() > 0:
        return float(np.dot(ratings_top_k.fillna(0).values, weights.values) / weights.sum())
    return _safe_mean(user_item_matrix.loc[user_id])


def predict_rating_item_based(
    user_id: int,
    movie_id: int,
    user_item_matrix: pd.DataFrame,
    item_similarity: pd.DataFrame,
    k: int = 5,
) -> float:
    """Predict rating using item-based collaborative filtering.

    Uses the user's ratings on the k most similar items.
    """
    if user_id not in user_item_matrix.index:
        raise KeyError(f"User {user_id} not found in user-item matrix")

    # If movie not present in similarity matrix, fallback to user's mean
    if movie_id not in item_similarity.index:
        return _safe_mean(user_item_matrix.loc[user_id])

    user_ratings = user_item_matrix.loc[user_id]
    rated = user_ratings[user_ratings > 0]
    rated = rated.drop(movie_id, errors="ignore")
    if rated.empty:
        return _safe_mean(user_ratings)

    sims = item_similarity.loc[movie_id]
    sims = sims[rated.index]
    top_k = sims.sort_values(ascending=False).head(k)
    items = top_k.index.tolist()

    ratings_top_k = user_ratings.loc[items]
    weights = top_k.loc[items]

    if weights.sum() > 0:
        return float(np.dot(ratings_top_k.fillna(0).values, weights.values) / weights.sum())
    return _safe_mean(user_ratings)


def evaluate_predictions(true_ratings: List[float], predicted_ratings: List[float]) -> float:
    """Compute RMSE between true and predicted ratings."""
    mse = mean_squared_error(true_ratings, predicted_ratings)
    return sqrt(mse)


if __name__ == "__main__":
    ratings = load_ratings_from_db()
    user_item_matrix = create_user_item_matrix(ratings)
    user_similarity = compute_user_similarity(user_item_matrix)
    item_similarity = compute_item_similarity(user_item_matrix)

    user_based_pred = predict_rating_user_based(1, 1, user_item_matrix, user_similarity, k=5)
    item_based_pred = predict_rating_item_based(1, 1, user_item_matrix, item_similarity, k=5)

    print("User-based predicted rating for user 1 on movie 1:", user_based_pred)
    print("Item-based predicted rating for user 1 on movie 1:", item_based_pred)