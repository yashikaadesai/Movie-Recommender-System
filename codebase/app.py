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

@app.route("/", methods=["GET", "POST"])
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    data = get_data()  # Load data on demand
    user_item_matrix = data['user_item_matrix']
    user_similarity = data['user_similarity']
    item_similarity = data['item_similarity']
    movies = data['movies']
    
    user_id = session['user_id']
    error_message = None
    rated_movies = []
    recommendations = []
    similar_users = []
    selected_algorithm = "user"
    
    # Get the logged-in user's ratings
    user_ratings = user_item_matrix.loc[user_id]
    rated_movie_ids = user_ratings[user_ratings > 0].index.tolist()
    for movie_id in rated_movie_ids:
        try:
            title = movies[movies["movieId"] == movie_id]["title"].values[0]
            rating = round(user_ratings[movie_id], 2)
            rated_movies.append((movie_id, title, rating))
        except (IndexError, KeyError):
            continue

    if request.method == "POST":
        try:
            selected_algorithm = request.form["algorithm"]
            unrated_movies = user_ratings[user_ratings == 0].index.tolist()
            predictions = {}
            for movie_id in unrated_movies:
                if selected_algorithm == "user":
                    pred_rating = predict_rating_user_based(user_id, movie_id, user_item_matrix, user_similarity, k=5)
                else:
                    pred_rating = predict_rating_item_based(user_id, movie_id, user_item_matrix, item_similarity, k=5)
                predictions[movie_id] = pred_rating
            top_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:10]
            for movie_id, rating in top_movies:
                try:
                    title = movies[movies["movieId"] == movie_id]["title"].values[0]
                    recommendations.append((movie_id, title, round(rating, 2)))
                except (IndexError, KeyError):
                    continue
            # Compute top 5 similar users (excluding current user)
            sim_scores = user_similarity.loc[user_id].drop(user_id)
            top_similar = sim_scores.sort_values(ascending=False).head(5)
            for similar_user_id, score in top_similar.items():
                similar_users.append((similar_user_id, round(score, 2)))
        except Exception as e:
            error_message = str(e)
    
    return render_template(
        "index.html",
        rated_movies=rated_movies,
        recommendations=recommendations,
        similar_users=similar_users,
        error_message=error_message,
        selected_algorithm=selected_algorithm
    )

def evaluate_algorithms():
    data = get_data()
    ratings = data['ratings']
    train_data = ratings.sample(frac=0.8, random_state=42)
    test_data = ratings.drop(train_data.index)
    train_matrix = create_user_item_matrix(train_data)
    train_user_sim = compute_user_similarity(train_matrix)
    train_item_sim = compute_item_similarity(train_matrix)
    
    user_based_predictions = []
    item_based_predictions = []
    actual_ratings = []
    
    for _, row in test_data.iterrows():
        user_id_val, movie_id, rating = row['userId'], row['movieId'], row['rating']
        try:
            user_pred = predict_rating_user_based(user_id_val, movie_id, train_matrix, train_user_sim, k=5)
            item_pred = predict_rating_item_based(user_id_val, movie_id, train_matrix, train_item_sim, k=5)
            user_based_predictions.append(user_pred)
            item_based_predictions.append(item_pred)
            actual_ratings.append(rating)
        except:
            continue
    
    user_based_rmse = evaluate_predictions(actual_ratings, user_based_predictions)
    item_based_rmse = evaluate_predictions(actual_ratings, item_based_predictions)
    return user_based_rmse, item_based_rmse

@app.route("/calculate_performance")
def calculate_performance():
    user_rmse, item_rmse = evaluate_algorithms()
    better_model = "User-based" if user_rmse < item_rmse else "Item-based"
    return jsonify({
        "user_rmse": user_rmse,
        "item_rmse": item_rmse,
        "better_model": better_model
    })

@app.route("/performance")
def performance():
    # The performance page loads metrics asynchronously via JS.
    return render_template("performance.html")

if __name__ == "__main__":
    app.run()