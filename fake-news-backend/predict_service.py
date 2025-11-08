# fake-news-backend/predict_service.py
import os
import pickle
import numpy as np
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "saved_models", "fake_news_model.pkl")
VECT_PATH = os.path.join(BASE_DIR, "saved_models", "tfidf_vectorizer.pkl")

# load model & vectorizer once at startup
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECT_PATH, "rb") as f:
    vectorizer = pickle.load(f)


def _simple_preprocess(text: str) -> str:
    """Minimal preprocessing — keep same as used during training."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # remove URLs, emails, special chars (keep spaces and alphanumerics)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_text(text: str, threshold: float = 0.5):
    """
    Returns:
      {
        "prediction": "FAKE" or "REAL" or "UNCERTAIN",
        "confidence": float (0..1),
        "proba": { "FAKE": float, "REAL": float }
      }
    threshold: minimum probability required to pick the top class. If top < threshold -> UNCERTAIN
    """
    cleaned = _simple_preprocess(text)
    X = vectorizer.transform([cleaned])

    # try predict_proba; if not available, use decision_function and softmax
    proba_dict = {}
    try:
        probs = model.predict_proba(X)[0]  # array, order matches model.classes_
        classes = list(model.classes_)
        proba_dict = {str(c): float(probs[i]) for i, c in enumerate(classes)}
        top_idx = int(np.argmax(probs))
        top_class = classes[top_idx]
        top_prob = float(probs[top_idx])
    except Exception:
        # fallback: use decision_function then sigmoid/softmax to approximate
        try:
            scores = model.decision_function(X)
            if scores.ndim == 1:
                # binary: convert score -> probability-like via sigmoid
                score = float(scores[0])
                top_prob = 1 / (1 + np.exp(-score))
                # model.classes_ ordering
                classes = list(model.classes_)
                # interpret >0.5 as classes[1], else classes[0]
                top_class = classes[1] if top_prob >= 0.5 else classes[0]
                proba_dict = {classes[0]: round(1 - top_prob, 4), classes[1]: round(top_prob, 4)}
            else:
                # multiclass: softmax
                scores = scores[0]
                exps = np.exp(scores - np.max(scores))
                probs = exps / exps.sum()
                classes = list(model.classes_)
                top_idx = int(np.argmax(probs))
                top_class = classes[top_idx]
                top_prob = float(probs[top_idx])
                proba_dict = {str(c): float(probs[i]) for i, c in enumerate(classes)}
        except Exception:
            # As last resort: use predict() only
            pred = model.predict(X)[0]
            return {"prediction": str(pred), "confidence": 0.0, "proba": {}}

    # apply threshold: if top_prob < threshold, mark UNCERTAIN
    if top_prob < threshold:
        prediction = "UNCERTAIN"
    else:
        prediction = str(top_class)

    return {
        "prediction": prediction,
        "confidence": round(float(top_prob), 4),
        "proba": proba_dict
    }
