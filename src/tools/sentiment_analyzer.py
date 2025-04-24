def analyze_sentiment(text: str):
    """Analyzes the sentiment of the given text and returns a score and analysis."""
    from textblob import TextBlob

    analysis = TextBlob(text)
    sentiment_score = analysis.sentiment.polarity  # Score ranges from -1 (negative) to 1 (positive)
    sentiment_analysis = "Positive" if sentiment_score > 0 else "Negative" if sentiment_score < 0 else "Neutral"

    return {
        "score": sentiment_score,
        "analysis": sentiment_analysis
    }