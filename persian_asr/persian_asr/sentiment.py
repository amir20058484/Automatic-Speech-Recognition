from shekar import SentimentClassifier

from .utils import action_logger, log_action, time_logger


@action_logger
@time_logger
def analyze_sentiment(text: str):
    """
    Analyze sentiment of Persian text using Shekar SentimentClassifier.
    Returns normalized score in range [-1, 1].
    """
    if not text or text.strip() == "":
        log_action("Empty text provided for sentiment analysis", level="warning")
        return {"label": "unknown", "score": 0.0}

    log_action("🔍 Initializing SentimentClassifier...")
    model = SentimentClassifier()

    label, confidence = model(text)

    score = confidence if label == "positive" else -confidence

    result = {"label": label, "score": round(score, 4)}
    log_action(f"✅ Sentiment: {label} ({score:.2f})")

    return result
