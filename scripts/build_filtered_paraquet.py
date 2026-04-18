import duckdb

# -----------------------------
# CONFIG
# -----------------------------
DATASET_DIR = "./financial-news"  # path from snapshot_download
OUTPUT_FILE = "./filtered_news.parquet"

START_DATE = "2020-06-01"
END_DATE = "2023-06-01"

# Choose which date to use:
USE_TRADING_DATE = False  # True = use extra_fields.date_trading

# -----------------------------
# SETUP
# -----------------------------
con = duckdb.connect("news.duckdb")

parquet_glob = f"{DATASET_DIR}/data/*/*.parquet"

# -----------------------------
# BUILD QUERY
# -----------------------------
if USE_TRADING_DATE:
    date_expr = """
    CAST(json_extract_string(extra_fields, '$.date_trading') AS TIMESTAMP)
    """
else:
    date_expr = "CAST(date AS TIMESTAMP)"

query = f"""
COPY (
    SELECT
        date,
        text,
        json_extract_string(extra_fields, '$.dataset') AS dataset,
        json_extract_string(extra_fields, '$.source') AS source,
        json_extract_string(extra_fields, '$.url') AS url,
        from_json(json_extract(extra_fields, '$.stocks'), '["VARCHAR"]') AS stocks
    FROM read_parquet('{parquet_glob}')
    WHERE {date_expr} BETWEEN TIMESTAMP '{START_DATE}' AND TIMESTAMP '{END_DATE}'
        AND json_extract(extra_fields, '$.stocks') IS NOT NULL
) TO '{OUTPUT_FILE}' (FORMAT PARQUET);
"""

# -----------------------------
# RUN
# -----------------------------
print("Running filter query...")
con.execute(query)

print(f"Done. Filtered data saved to: {OUTPUT_FILE}")
