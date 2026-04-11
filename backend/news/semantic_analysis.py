from transformers import pipeline
from backend.news.types import NewsDict
import pandas as pd

classifier = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
)


def create_feature(news_data: list[NewsDict]) -> pd.Series:
    """Creates a feature column for sentiment analysis from news data."""
    news = [item["news"] for item in news_data]
    dates = [item["date"] for item in news_data]
    result = classifier(news)

    sentiment_col = pd.Series(result, index=pd.to_datetime(dates))
    return sentiment_col
