from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get MongoDB URI from environment variables
uri = os.environ.get('MONGODB_URI')
if not uri:
    raise ValueError("No MongoDB URI found in environment variables")

# Connect to MongoDB Atlas cluster
client = MongoClient(uri)
db = client["Movie-Recommender"]

# Select the required  collections
movies_collection = db["movies"]
ratings_collection = db["ratings"]
tags_collection = db["tags"]
links_collection = db["links"]

# Running some sample queries
print("One movie:", movies_collection.find_one())  
print("One rating:", ratings_collection.find_one())
print("One tag:", tags_collection.find_one())
print("One link:", links_collection.find_one())