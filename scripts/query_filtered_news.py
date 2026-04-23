import pandas as pd

df = pd.read_parquet("backend/data/news_features.parquet")
print(df.head())
aapl_df = df[df["ticker"] == "AAPL"]

aapl_sentiment = aapl_df[["date", "mean_sentiment", "article_count", "positive_ratio"]]
print(aapl_df.head(100))
print(aapl_df.tail(100))
