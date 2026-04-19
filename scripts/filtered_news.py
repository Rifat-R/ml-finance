import duckdb

con = duckdb.connect()


TICKER = "IP"

df = con.execute(f"""
SELECT *
FROM read_parquet('filtered_news.parquet')
WHERE list_contains(stocks, '{TICKER}')
ORDER BY date DESC
LIMIT 50
""").df()


for index, row in df.iterrows():
    print(f"Date: {row['date']}")
    print(f"Text: {row['text']}")
    print(f"Dataset: {row['dataset']}")
    print(f"Source: {row['source']}")
    print(f"URL: {row['url']}")
    print(f"Stocks: {row['stocks']}")
    print("-" * 40)


# get total count


total_count = con.execute(f"""
SELECT COUNT(*) AS total_count
FROM read_parquet('filtered_news.parquet')
WHERE list_contains(stocks, '{TICKER}')
""").fetchone()[0]

print(f"Total filtered news articles for {TICKER}: {total_count:,}")


df = con.execute("""
SELECT
    stock,
    COUNT(*) AS freq
FROM read_parquet('filtered_news.parquet')
CROSS JOIN UNNEST(stocks) AS s(stock)
WHERE stocks IS NOT NULL
GROUP BY stock
ORDER BY freq DESC
""").df()

print(df)
