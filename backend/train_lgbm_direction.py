"""
Toy example: train a LightGBM model to predict next-day direction (up/down)
based on simple return features.

This is JUST for educational purposes, not a trading edge.
"""

import re
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from lightgbm import LGBMClassifier
import joblib


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
    df = df.copy()

    # Ensure we have a 'Close' column (from yfinance, after cleaning it's usually "Close")
    if "Close" not in df.columns:
        # Sometimes Adj_Close is more reliable; adapt if needed
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

    # Simple features from past returns
    df["ret_1"] = df["return"]
    df["ret_3"] = df["return"].rolling(3).mean()
    df["ret_5"] = df["return"].rolling(5).mean()
    df["ret_10"] = df["return"].rolling(10).mean()
    df["vol_10"] = df["return"].rolling(10).std()

    # Target: 1 if next day's return is positive, else 0
    df["target"] = (df["return"].shift(-1) > 0).astype(int)

    # Drop rows with NaNs (from rolling and shifting)
    df = df.dropna()

    # Keep only the columns we care about for modeling
    cols = ["ret_1", "ret_3", "ret_5", "ret_10", "vol_10", "target"]
    df = df[cols].copy()

    # Clean column names for LightGBM safety
    df.columns = [clean_name(c) for c in df.columns]

    return df


def train_and_save_model(
    ticker: str = "AAPL",
    model_path: str = "lgbm_direction.pkl",
) -> None:
    print(f"Downloading data for {ticker}...")
    raw = download_data(ticker)
    print("Creating features...")
    df = create_features(raw)

    # After cleaning, feature names should still be these (already safe):
    feature_cols = ["ret_1", "ret_3", "ret_5", "ret_10", "vol_10"]
    # But to be extra safe, clean them the same way:
    feature_cols = [clean_name(c) for c in feature_cols]

    # Make sure these columns actually exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Missing expected feature columns: {missing}. Actual columns: {df.columns.tolist()}"
        )

    X = df.loc[:, feature_cols].copy()
    y = df["target"].copy()

    # Final sanity: clean X column names again
    X.columns = [clean_name(c) for c in X.columns]

    print("Data shape:", X.shape)
    print("Feature columns:", list(X.columns))

    # Simple time-based split: first 80% train, last 20% test
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    print("Training LightGBM model...")
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Very quick sanity check accuracy (NOT a proper evaluation)
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).mean()
    print(f"Test accuracy (toy, not CV): {acc:.3f}")

    artifact = {
        "model": model,
        "feature_cols": list(X.columns),
        "ticker": ticker,
    }

    joblib.dump(artifact, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train_and_save_model()
