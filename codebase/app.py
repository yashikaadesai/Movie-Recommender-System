from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
import pandas as pd
import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)


uri = os.environ.get('MONGODB_URI')
if not uri:
    raise ValueError("No MongoDB URI found in environment variables")

def load_movies_from_db():
    """Load movie data from MongoDB"""
    client = MongoClient(uri)
    db = client["Movie-Recommender"]
    movies_cursor = db["movies"].find()
    movies = pd.DataFrame(list(movies_cursor))
    if 'movieId' in movies.columns:
        movies['movieId'] = movies['movieId'].astype(int)
    return movies
