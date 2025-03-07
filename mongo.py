from pymongo import MongoClient
from datetime import datetime

# Set up MongoDB connection
client = MongoClient('localhost', 27017)  # Connect to MongoDB running on localhost
db = client['air_quality']  # Create (or connect to) a database called 'air_quality'
collection = db['delhi_data']  # Create (or connect to) a collection called 'delhi_data'
