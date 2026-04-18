import duckdb

con = duckdb.connect()


TICKER = "AAPL"

df = con.execute("""
SELECT *
FROM read_parquet('filtered_news.parquet')
WHERE list_contains(stocks, 'WMT')
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
""").fetchone()[0]

print(f"Total filtered news articles for {TICKER}: {total_count:,}")
