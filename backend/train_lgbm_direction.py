import re
from typing import Optional

import pandas as pd
import yfinance as yf


def clean_name(name: str) -> str:
    """
    Clean a column/feature name so LightGBM doesn't complain about
    special JSON characters.

    Keeps only letters, digits, and underscores; replaces everything else with "_".
    """
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name))


def download_data(
    ticker: str, start: str = "2015-01-01", end: Optional[str] = None
) -> pd.DataFrame:
    data = yf.download(ticker, start=start, end=end)
    if data is None or data.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")

    # Ensure columns are strings and cleaned
    data.columns = [clean_name(c) for c in data.columns]
    return data


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates features for EVERY closeday in the DataFrame
    """

    df = df.copy()

    # Ensure we have a 'Close' column (from yfinance, after cleaning it's usually "Close")
    if "Close" not in df.columns:
        # Try a few common variants
        candidates = [c for c in df.columns if "Close" in c]
        if not candidates:
            raise RuntimeError(
                f"Could not find a 'Close' column in data. Columns: {df.columns.tolist()}"
            )
        close_col = candidates[0]
    else:
        close_col = "Close"

    # Daily returns from the close price
    df["return"] = df[close_col].pct_change()
    df = df.loc[df["return"].abs() > 0.002]  # remove days with < 0.2% movement

    # Simple features from past returns
    df["ret_1"] = df["return"]
    # Basically sliding window averages
    df["ret_3"] = df["return"].rolling(3).mean()
    df["ret_5"] = df["return"].rolling(5).mean()
    df["ret_10"] = df["return"].rolling(10).mean()
    df["vol_10"] = df["return"].rolling(10).std()

    # 1 if next day's return is positive, else 0
    df["target"] = (df["return"].shift(-1) > 0).astype(int)

    # Drop rows with NaNs (from rolling and shifting)
    df = df.dropna()

    cols = ["ret_1", "ret_3", "ret_5", "ret_10", "vol_10", "target"]
    df = df[cols].copy()  # type: ignore

    # Clean column names for LightGBM safety
    df.columns = [clean_name(c) for c in df.columns]

    return df
