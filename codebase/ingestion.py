"""Dataset ingestion utility.

Keeps the original behavior: reads CSV files from a relative
`../Datasets/ml-latest-small` directory and inserts them into
MongoDB collections. This file is a clearer, slightly more
robust rewrite of the original implementation.
"""

import csv
import os
from pathlib import Path
from typing import Callable, Dict, Optional

from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


# Load environment variables from .env (if present)
load_dotenv()


# MongoDB connection
MONGODB_URI = os.environ.get("MONGODB_URL")
if not MONGODB_URI:
    raise ValueError("No MongoDB URI found in environment variables")

client = MongoClient(MONGODB_URI, server_api=ServerApi("1"))
db = client["Movie-Recommender"]

try:
    client.admin.command("ping")
    print("Successfully connected to MongoDB!")
except Exception as exc:  # pragma: no cover - runtime check
    print("Connection error:", exc)


# Collections
movies_collection = db["movies"]
ratings_collection = db["ratings"]
tags_collection = db["tags"]
links_collection = db["links"]


# Dataset directory
DATASET_DIR = Path(__file__).resolve().parent.joinpath("..", "dataset", "ml-latest-small")


def load_csv_to_collection(
    csv_filename: str,
    collection,
    transform_row: Optional[Callable[[Dict[str, str]], Dict]] = None,
) -> None:
    """Load CSV rows into `collection`.

    Args:
        csv_filename: Name of the CSV file inside `DATASET_DIR`.
        collection: A PyMongo collection object.
        transform_row: Optional function to convert CSV row strings to desired types.
    """

    file_path = DATASET_DIR.joinpath(csv_filename)
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    try:
        with file_path.open(mode="r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            records = []
            for row in reader:
                if transform_row:
                    try:
                        row = transform_row(row)
                    except Exception as e:  # keep behavior tolerant
                        print(f"Row transform error for {collection.name}: {e}")
                        continue
                records.append(row)

            if records:
                collection.insert_many(records)
                print(f"Inserted {len(records)} records into '{collection.name}' collection.")
    except Exception as exc:
        print(f"Error inserting data into {collection.name}: {exc}")


# Transformation helpers
def transform_movies(row: Dict[str, str]) -> Dict:
    row["movieId"] = int(row["movieId"])
    return row


def transform_ratings(row: Dict[str, str]) -> Dict:
    row["userId"] = int(row["userId"]) if row.get("userId") else None
    row["movieId"] = int(row["movieId"]) if row.get("movieId") else None
    row["rating"] = float(row["rating"]) if row.get("rating") else None
    row["timestamp"] = int(row["timestamp"]) if row.get("timestamp") else None
    return row


def transform_tags(row: Dict[str, str]) -> Dict:
    row["userId"] = int(row["userId"]) if row.get("userId") else None
    row["movieId"] = int(row["movieId"]) if row.get("movieId") else None
    row["timestamp"] = int(row["timestamp"]) if row.get("timestamp") else None
    return row


def transform_links(row: Dict[str, str]) -> Dict:
    row["movieId"] = int(row["movieId"]) if row.get("movieId") else None
    tmdb = row.get("tmdbId")
    if tmdb is None or tmdb == "":
        row["tmdbId"] = tmdb
    else:
        row["tmdbId"] = int(tmdb) if str(tmdb).isdigit() else tmdb
    return row


load_csv_to_collection("movies.csv", movies_collection, transform_row=transform_movies)
load_csv_to_collection("ratings.csv", ratings_collection, transform_row=transform_ratings)
load_csv_to_collection("tags.csv", tags_collection, transform_row=transform_tags)
load_csv_to_collection("links.csv", links_collection, transform_row=transform_links)

print("Data insertion complete.")