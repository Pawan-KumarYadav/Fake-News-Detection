# fake-news-backend/services.py

from predict_service import predict_news

def predict_text(text: str):
    """
    Wrapper function for prediction.
    Called by /verify/text endpoint.
    """
    return predict_news(text)
