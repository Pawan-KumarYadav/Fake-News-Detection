# fake-news-backend/train_model.py

"""
This script trains a Fake News Detection model using TF-IDF + PassiveAggressiveClassifier.
It ensures consistent results by fixing random seeds and saves the trained model + vectorizer.
"""

# ------------------ IMPORTS ------------------
import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ------------------ RANDOM SEED FIX ------------------
# Fix random seeds for reproducibility — this ensures stable, repeatable results
random.seed(42)
np.random.seed(42)

# ------------------ LOAD DATASET ------------------
data_path = os.path.join("data", "news.csv")
print("📂 Looking for file at:", os.path.abspath(data_path))
print("📄 Exists?", os.path.exists(data_path))

if not os.path.exists(data_path):
    raise FileNotFoundError("❌ Dataset file not found. Please place 'news.csv' inside 'data/' folder.")

# Read the CSV file
df = pd.read_csv(data_path)
print("✅ Dataset loaded successfully!")
print(df.head(), "\n")

# ------------------ CLEAN DATA ------------------
# Drop missing or empty rows to prevent errors
df = df.dropna(subset=["text", "label"])

# Define feature and label columns
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

X = df[TEXT_COLUMN]
y = df[LABEL_COLUMN]

# ------------------ SPLIT DATA ------------------
# Fixed random_state=42 ensures the same split every time
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"📊 Dataset split into {len(X_train)} training and {len(X_test)} testing samples.\n")

# ------------------ VECTORIZATION ------------------
# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

# ------------------ TRAIN MODEL ------------------
# PassiveAggressiveClassifier works well for text classification
model = PassiveAggressiveClassifier(max_iter=50, random_state=42)
model.fit(tfidf_train, y_train)

# ------------------ EVALUATE MODEL ------------------
y_pred = model.predict(tfidf_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Accuracy: {round(accuracy * 100, 2)}%")

# Optional: show confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("🧩 Confusion Matrix:\n", cm, "\n")

# ------------------ SAVE MODEL AND VECTORIZER ------------------
# Create folder if not exists
os.makedirs("saved_models", exist_ok=True)

model_path = os.path.join("saved_models", "fake_news_model.pkl")
vectorizer_path = os.path.join("saved_models", "tfidf_vectorizer.pkl")

with open(model_path, "wb") as f:
    pickle.dump(model, f)

with open(vectorizer_path, "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model and vectorizer saved successfully in 'saved_models/' folder!")
print("🚀 Training complete! Model ready to be used for predictions.")

