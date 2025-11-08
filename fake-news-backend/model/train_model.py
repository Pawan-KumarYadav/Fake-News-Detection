# fake-news-backend/model/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os

# ✅ Step 1: Load dataset
dataset_path = os.path.join("data", "train.csv")  # replace with your dataset file name
df = pd.read_csv(dataset_path)

print("Dataset loaded successfully!")
print(df.head())

# ✅ Step 2: Basic preprocessing
df = df.dropna()
X = df['text']  # change this to the column that contains news text
y = df['label']  # change this to the label column (e.g., FAKE/REAL)

# ✅ Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# ✅ Step 4: TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# ✅ Step 5: Model Training
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

# ✅ Step 6: Evaluate
y_pred = model.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {round(score*100,2)}%")

# ✅ Step 7: Save Model and Vectorizer
os.makedirs("saved_models", exist_ok=True)
with open("saved_models/fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("saved_models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

print("Model and vectorizer saved successfully!")
