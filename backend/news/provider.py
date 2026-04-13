from backend.news.types import NewsDict


def fetch_news_data(ticker: str) -> list[NewsDict]:
    data: list[NewsDict] = [{"news": "example news headline", "date": "2024-01-01"}]
    return data
