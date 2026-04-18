import duckdb

con = duckdb.connect()

con.execute("""
COPY (
    SELECT *
    FROM read_parquet('filtered_news.parquet')
    WHERE NOT contains(text, 'Most Read from Bloomberg')
) TO 'filtered_news.parquet.tmp' (FORMAT PARQUET);
""")

import os

os.replace("filtered_news.parquet.tmp", "filtered_news.parquet")
