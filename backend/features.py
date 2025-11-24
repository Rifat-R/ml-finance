import re
from typing import Sequence

import numpy as np
import pandas as pd

FEATURE_COLS = ["ret_1", "ret_3", "ret_5", "ret_10", "vol_10"]


def clean_name(name: str) -> str:
    """Clean a column/feature name so LightGBM doesn't complain about special JSON characters."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name))


def _select_close_column(df: pd.DataFrame) -> str:
    """Pick a close column, preferring 'Close' if present."""
    if "Close" in df.columns:
        return "Close"
    candidates = [c for c in df.columns if "Close" in c]
    if not candidates:
        raise RuntimeError(
            f"Could not find a 'Close' column in data. Columns: {df.columns.tolist()}"
        )
    return candidates[0]


def _features_from_returns(returns: pd.Series) -> dict[str, float]:
    """Compute feature dict from a return series."""
    return {
        "ret_1": returns.iloc[-1],
        "ret_3": returns.iloc[-3:].mean(),
        "ret_5": returns.iloc[-5:].mean(),
        "ret_10": returns.iloc[-10:].mean(),
        "vol_10": returns.iloc[-10:].std(),
    }


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build training features for whole training data.
    Outputs columns: FEATURE_COLS + target.
    """
    df = df.copy()

    close_col = _select_close_column(df)

    # Daily returns from the close price
    df["return"] = df[close_col].pct_change()

    df["ret_1"] = df["return"]
    df["ret_3"] = df["return"].rolling(3).mean()
    df["ret_5"] = df["return"].rolling(5).mean()
    df["ret_10"] = df["return"].rolling(10).mean()
    df["vol_10"] = df["return"].rolling(10).std()

    # 1 if next day's return is positive, else 0
    df["target"] = (df["return"].shift(-1) > 0).astype(int)

    df = df.dropna()

    cols = [*FEATURE_COLS, "target"]
    df = df[cols].copy()  # type: ignore

    df.columns = [clean_name(c) for c in df.columns]

    return df


def build_features_from_closes(closes: Sequence[float]) -> np.ndarray:
    """
    Convert a sequence of close prices into feature vector matching FEATURE_COLS.
    """
    closes_arr = np.array(closes, dtype=float)

    if len(closes_arr) < 11:
        raise ValueError("Need at least 11 closing prices for feature extraction.")

    returns = closes_arr[1:] / closes_arr[:-1] - 1.0
    s = pd.Series(returns)
    feature_dict = _features_from_returns(s)

    x = np.array([[feature_dict[c] for c in FEATURE_COLS]], dtype=float)
    return x
