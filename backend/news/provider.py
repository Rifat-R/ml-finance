from typing import TypedDict


class NewsDict(TypedDict):
    news: str
    date: str


def fetch_news_data(ticker: str) -> NewsDict:
    data: NewsDict = {"news": "example news headline", "date": "2024-01-01"}
    return data
