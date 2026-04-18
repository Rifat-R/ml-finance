import duckdb

con = duckdb.connect()


df = con.execute("""
SELECT 
    DATE(date) AS d,
    COUNT(*) AS count
FROM read_parquet('filtered_news.parquet')
GROUP BY d
ORDER BY d
""").df()


import matplotlib.pyplot as plt

plt.figure()
plt.plot(df["d"], df["count"])
plt.xlabel("Date")
plt.ylabel("Number of Records")
plt.title("Records per Day")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("records_per_day.png", dpi=150)
