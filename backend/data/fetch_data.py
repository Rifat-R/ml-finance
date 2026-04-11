import pandas as pd
from backend.data import tiingo_client
from typing import TypedDict


def fetch_stock_data(
    ticker: str, start: str = "2010-01-01", end: str | None = None
) -> pd.DataFrame:
    data = tiingo_client.get_dataframe(
        ticker,
        startDate=start,
        endDate=end,
        frequency="daily",
    )

    if not isinstance(data, pd.DataFrame):
        raise RuntimeError(f"Data downloaded for {ticker} is not a DataFrame")

    return data


class NewsDict(TypedDict):
    news: str
    date: str


def fetch_news_data(ticker: str) -> NewsDict:
    data: NewsDict = {"news": "example news headline", "date": "2024-01-01"}
    return data
