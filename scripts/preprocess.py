import duckdb
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
)

con = duckdb.connect()

df = con.execute("""
SELECT date, text, stocks
FROM read_parquet('filtered_news.parquet')
""").df()

df["date"] = pd.to_datetime(df["date"])

texts = df["text"].fillna("").astype(str).tolist()

results = []


for i in tqdm(range(0, len(texts), 16)):
    batch_texts = texts[i : i + 16]
    batch_results = classifier(
        batch_texts,
        truncation=True,
        max_length=256,
    )
    results.extend(batch_results)


def label_score(result: dict) -> float:
    label = result["label"].lower()
    score = float(result["score"])

    if label == "positive":
        return score
    elif label == "negative":
        return -score
    else:
        return 0.0


df["sentiment"] = [label_score(r) for r in results]

df = df.explode("stocks").rename(columns={"stocks": "ticker"})
df["is_positive"] = (df["sentiment"] > 0).astype(int)

features = (
    df.groupby([df["date"].dt.date, "ticker"])
    .agg(
        mean_sentiment=("sentiment", "mean"),
        article_count=("sentiment", "count"),
        sum_sentiment=("sentiment", "sum"),
        positive_ratio=("is_positive", "mean"),
    )
    .reset_index()
    .rename(columns={"date": "date"})
)


print(features.head())
features.to_parquet("news_features.parquet", index=False)
