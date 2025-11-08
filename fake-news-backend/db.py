from pymongo import MongoClient
import os
from dotenv import load_dotenv
from pymongo.errors import ConnectionFailure

# Load environment variables
load_dotenv()

# Get MongoDB connection details
MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "fake_news_db")

print("MONGO_URI:", MONGO_URI)

try:
    # Connect to MongoDB
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    
    # Ping the server to check connection
    client.admin.command("ping")
    
    print("✅ MongoDB connected successfully!")

    # Access the database
    db = client[DATABASE_NAME]

    # Collections
    users_col = db["users"]
    verifications_col = db["verifications"]

except ConnectionFailure as e:
    print("❌ MongoDB connection failed:", e)
