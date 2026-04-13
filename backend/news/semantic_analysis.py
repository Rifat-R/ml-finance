from transformers import pipeline
from backend.news.provider import fetch_news_data
import pandas as pd

classifier = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
)


def semantic_feature_series(ticker: str) -> pd.Series:
    """Creates a feature column for sentiment analysis from news data."""
    news_data = fetch_news_data(ticker)

    news = [item["news"] for item in news_data]
    dates = [item["date"] for item in news_data]
    result = classifier(news)

    sentiment_series = pd.Series(result, index=pd.to_datetime(dates))
    return sentiment_series
