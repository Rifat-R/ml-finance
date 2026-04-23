import pandas as pd

DF = pd.read_parquet("data/news_features.parquet")


def load_news_sentiment_features(ticker: str) -> pd.DataFrame:
    ticker_df = DF[DF["ticker"] == ticker]
    return ticker_df.copy()
