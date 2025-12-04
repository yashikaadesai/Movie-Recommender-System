"""
evaluation.py

This module splits the ratings dataset into training and test sets, applies the recommendation algorithms,
and evaluates their performance by calculating the RMSE for user-based and item-based predictions.
"""


import pandas as pd
from sklearn.model_selection import train_test_split
from rec_algos import (
    load_ratings_from_db,
    create_user_item_matrix,
    compute_user_similarity,
    compute_item_similarity,
    predict_rating_user_based,
    predict_rating_item_based,
    evaluate_predictions
)

def train_test_split_ratings(ratings, test_size=0.2, random_state=42):
    """
    Splits the ratings DataFrame into training and test sets.
    
    Args:
        ratings (DataFrame): DataFrame containing ratings.
        test_size (float): Proportion of data to be used for testing.
        random_state (int): Seed for reproducibility.
    
    Returns:
        train_ratings (DataFrame): Training set.
        test_ratings (DataFrame): Test set.
    """
    train_ratings, test_ratings = train_test_split(ratings, test_size=test_size, random_state=random_state)
    return train_ratings, test_ratings

def evaluate_recommender(train_ratings, test_ratings, k=5):
    """
    Evaluates both user-based and item-based collaborative filtering approaches.
    
    For each rating in the test set, predictions are made if the user and movie exist in the training data.
    The RMSE is then computed for both methods.
    
    Args:
        train_ratings (DataFrame): Training ratings.
        test_ratings (DataFrame): Test ratings.
        k (int): Number of neighbors/items to consider.
        
    Returns:
        rmse_user (float): RMSE for the user-based method.
        rmse_item (float): RMSE for the item-based method.
    """
    # Build user-item matrix from training data
    user_item_matrix = create_user_item_matrix(train_ratings)
    # Compute similarity matrices based on the training data
    user_similarity = compute_user_similarity(user_item_matrix)
    item_similarity = compute_item_similarity(user_item_matrix)
    
    true_ratings_user = []
    predicted_ratings_user = []
    true_ratings_item = []
    predicted_ratings_item = []
    
    # Iterate through each record in the test set
    for _, row in test_ratings.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        true_rating = row['rating']
        
        # Only predict if the user and movie are present in the training set
        if user_id not in user_item_matrix.index or movie_id not in user_item_matrix.columns:
            continue
        
        # Predict using user-based CF
        pred_user = predict_rating_user_based(user_id, movie_id, user_item_matrix, user_similarity, k)
        predicted_ratings_user.append(pred_user)
        true_ratings_user.append(true_rating)
        
        # Predict using item-based CF
        pred_item = predict_rating_item_based(user_id, movie_id, user_item_matrix, item_similarity, k)
        predicted_ratings_item.append(pred_item)
        true_ratings_item.append(true_rating)
    
    # Compute RMSE for both methods
    rmse_user = evaluate_predictions(true_ratings_user, predicted_ratings_user)
    rmse_item = evaluate_predictions(true_ratings_item, predicted_ratings_item)
    
    return rmse_user, rmse_item

if __name__ == "__main__":
    # Load full ratings data from MongoDB
    ratings = load_ratings_from_db()
    # Split the data into training and test sets
    train_ratings, test_ratings = train_test_split_ratings(ratings, test_size=0.2, random_state=42)
    # Evaluate the recommendation methods on the test set
    rmse_user, rmse_item = evaluate_recommender(train_ratings, test_ratings, k=5)
    
    print("User-based Collaborative Filtering RMSE:", rmse_user)
    print("Item-based Collaborative Filtering RMSE:", rmse_item)