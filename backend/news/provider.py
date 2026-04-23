import pandas as pd

DF = pd.read_parquet("data/news_features.parquet")


def load_news_sentiment_features(ticker: str) -> pd.DataFrame:
    ticker_df = DF[DF["ticker"] == ticker]
    semantic_cols = ticker_df.columns
    semantic_cols = semantic_cols.drop(["ticker", "date"])
    ticker_df[semantic_cols] = ticker_df[semantic_cols].fillna(0)
    return ticker_df.copy()
