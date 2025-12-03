from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
import pandas as pd
import numpy as np
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

uri = os.environ.get('MONGODB_URL')
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

def get_data():
    if not hasattr(app, 'data'):
        print("Loading ratings from database...")
        ratings = load_ratings_from_db()
        print(f"Loaded {len(ratings)} ratings")
        
        print("Creating user-item matrix...")
        user_item_matrix = create_user_item_matrix(ratings)
        print(f"User-item matrix shape: {user_item_matrix.shape}")
        
        print("Computing user similarity...")
        user_similarity = compute_user_similarity(user_item_matrix)
        
        print("Computing item similarity...")
        item_similarity = compute_item_similarity(user_item_matrix)
        
        print("Loading movies from database...")
        movies = load_movies_from_db()
        print(f"Loaded {len(movies)} movies")
        
        app.data = {
            'ratings': ratings,
            'user_item_matrix': user_item_matrix,
            'user_similarity': user_similarity,
            'item_similarity': item_similarity,
            'movies': movies
        }
    return app.data

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        try:
            user_id = int(request.form["userId"])
            data = get_data()  # Lazy-load data here
            if user_id not in data['user_item_matrix'].index:
                flash(f"User ID {user_id} not found in the dataset.", "error")
                return render_template('login.html')
            session['user_id'] = user_id
            return redirect(url_for('index'))
        except ValueError:
            flash("Invalid User ID. Please enter a numeric value.", "error")
            return render_template('login.html')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))