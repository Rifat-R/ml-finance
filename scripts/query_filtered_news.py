import pandas as pd

df = pd.read_parquet("data/news_features.parquet")
print(df.head())
aapl_df = df[df["ticker"] == "AAPL"]

aapl_sentiment = aapl_df[["date", "mean_sentiment", "article_count", "positive_ratio"]]
print(aapl_sentiment.head(100))
print(aapl_sentiment.tail(100))
