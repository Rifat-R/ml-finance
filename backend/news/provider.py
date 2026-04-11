from backend.news.types import NewsDict


def fetch_news_data(ticker: str) -> NewsDict:
    data: NewsDict = {"news": "example news headline", "date": "2024-01-01"}
    return data
