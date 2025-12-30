"""Exploratory Data Analysis for Movie-Recommender data.

This module connects to MongoDB using the `MONGODB_URI` env var,
loads `movies`, `ratings`, and `tags` collections into Pandas
DataFrames, performs lightweight preprocessing, and saves three
visualizations as PNG files.
"""

from pathlib import Path
import os
from typing import Tuple

import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go


load_dotenv()

# Configure Plotly defaults
pio.templates.default = "plotly_white"
DEFAULT_COLORSEQ = px.colors.sequential.Plasma


def get_db() -> MongoClient:
    uri = os.environ.get("MONGODB_URL")
    if not uri:
        raise ValueError("No MongoDB URI found in environment variables")
    client = MongoClient(uri)
    return client


def load_collections(client: MongoClient) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    db = client["Movie-Recommender"]
    movies = pd.DataFrame(list(db["movies"].find()))
    ratings = pd.DataFrame(list(db["ratings"].find()))
    tags = pd.DataFrame(list(db["tags"].find()))
    return movies, ratings, tags


def preprocess(movies: pd.DataFrame, ratings: pd.DataFrame, tags: pd.DataFrame):
    # Ensure expected columns exist and cast types where possible.
    if not movies.empty and "movieId" in movies.columns:
        movies["movieId"] = movies["movieId"].astype(int)
    if not ratings.empty:
        for col, dtype in {
            "movieId": int,
            "userId": int,
            "rating": float,
            "timestamp": int,
        }.items():
            if col in ratings.columns:
                ratings[col] = ratings[col].astype(dtype)
    if not tags.empty:
        for col in ("movieId", "userId", "timestamp"):
            if col in tags.columns:
                tags[col] = tags[col].astype(int)

    # Split genres into a list for counting
    movies["genres_list"] = movies.get("genres", "").apply(
        lambda x: x.split("|") if isinstance(x, str) and x else []
    )


def plot_rating_distribution(ratings: pd.DataFrame, out_path: Path):
    # Create integer bins 0..5 and draw bars so they are contiguous.
    if ratings.empty or "rating" not in ratings.columns:
        return

    # Define integer bins 0-1,1-2,...,5-6 (we show ticks 0..5)
    bin_edges = np.arange(0, 7)  # 0..6 edges for 6 bins (0-5)
    cat = pd.cut(ratings["rating"], bins=bin_edges, include_lowest=True, right=False)
    counts = cat.value_counts(sort=False)
    # x positions are 0..5
    x_bins = list(range(0, 6))
    y_counts = counts.values.tolist()

    # Choose colors from Antique palette (repeat if fewer than bins)
    palette = px.colors.qualitative.Antique
    colors = [palette[i % len(palette)] for i in range(len(x_bins))]

    # Build smoothed density line using fine-grained histogram and gaussian smoothing
    x_fine = np.linspace(0, 5, 400)
    hist_vals, hist_edges = np.histogram(ratings["rating"], bins=400, range=(0, 5))
    # Gaussian smoothing kernel
    kernel_width = 9
    kernel = np.exp(-0.5 * (np.linspace(-3, 3, kernel_width) ** 2))
    kernel = kernel / kernel.sum()
    hist_smooth = np.convolve(hist_vals, kernel, mode="same")
    # map hist_smooth centers
    hist_centers = 0.5 * (hist_edges[:-1] + hist_edges[1:])
    # Interpolate smoothed histogram to x_fine
    y_smooth = np.interp(x_fine, hist_centers, hist_smooth)
    # Scale smooth line to match bar heights visually
    if y_smooth.max() > 0:
        y_smooth = y_smooth * (max(y_counts) / y_smooth.max())

    # Create figure with bars and a line trace
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x_bins,
            y=y_counts,
            marker_color=colors,
            marker_line_color="white",
            marker_line_width=0.7,
            width=0.9,
            name="Count",
        )
    )
    # Add a smooth spline line with subtle fill for a cleaner look
    fig.add_trace(
        go.Scatter(
            x=x_fine,
            y=y_smooth,
            mode="lines",
            line=dict(color="rgba(40,40,40,0.9)", width=2, shape="spline", smoothing=1.3),
            fill="tozeroy",
            fillcolor="rgba(40,40,40,0.08)",
            hoverinfo="x+y",
            name="Smoothed distribution",
        )
    )

    fig.update_layout(
        title="Distribution of Movie Ratings",
        xaxis=dict(title="Rating", tickmode="array", tickvals=x_bins, ticktext=[str(i) for i in x_bins]),
        yaxis=dict(title="Frequency"),
        bargap=0.05,
    )

    out_file = out_path / "rating_distribution.png"
    try:
        fig.write_image(str(out_file))
    except Exception:
        fig.write_html(str(out_path / "rating_distribution.html"))


def plot_ratings_per_movie(ratings: pd.DataFrame, out_path: Path):
    ratings_per_movie = ratings.groupby("movieId").size().rename("count").reset_index()
    fig = px.histogram(
        ratings_per_movie,
        x="count",
        nbins=50,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Distribution of Number of Ratings per Movie",
        labels={"count": "Number of Ratings"},
    )
    fig.update_layout(yaxis_title="Frequency")
    out_file = out_path / "ratings_per_movie.png"
    try:
        fig.write_image(str(out_file))
    except Exception:
        fig.write_html(str(out_path / "ratings_per_movie.html"))
    return ratings_per_movie


def plot_genre_distribution(movies: pd.DataFrame, out_path: Path):
    genres_series = movies["genres_list"].explode()
    genre_counts = genres_series.value_counts()
    df = genre_counts.reset_index()
    df.columns = ["genre", "count"]
    fig = px.bar(
        df,
        x="genre",
        y="count",
        color="count",
        color_continuous_scale="Blackbody",
        title="Distribution of Movie Genres",
        labels={"genre": "Genre", "count": "Count"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    out_file = out_path / "genre_distribution.png"
    try:
        fig.write_image(str(out_file))
    except Exception:
        fig.write_html(str(out_path / "genre_distribution.html"))
    return genre_counts


def main(output_dir: str = "."):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    client = get_db()
    movies, ratings, tags = load_collections(client)
    preprocess(movies, ratings, tags)

    # Create visualizations and save summaries
    plot_rating_distribution(ratings, out)
    print("Summary statistics for movie ratings:")
    print(ratings["rating"].describe())

    ratings_per_movie = plot_ratings_per_movie(ratings, out)
    print("\nSummary statistics for number of ratings per movie:")
    print(ratings_per_movie["count"].describe())

    genre_counts = plot_genre_distribution(movies, out)
    print("\nGenre counts:")
    print(genre_counts)

    print("Data analysis complete. Visualizations saved as PNG files.")


if __name__ == "__main__":
    main()