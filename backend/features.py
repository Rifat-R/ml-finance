from typing import Sequence

import numpy as np
import pandas as pd

FEATURE_COLS = ["ret_1", "ret_3", "ret_5", "ret_10", "vol_10"]


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

    # Should use adjClose to account for dividends and stock splits
    close_col = "adjClose"

    # Daily percentage change for the close price, i.e. return = (close_t - close_{t-1}) / close_{t-1}
    df["return"] = df[close_col].pct_change()

    df["ret_1"] = df["return"]
    df["ret_3"] = df["return"].rolling(3).mean()
    df["ret_5"] = df["return"].rolling(5).mean()
    df["ret_10"] = df["return"].rolling(10).mean()
    df["vol_10"] = df["return"].rolling(10).std()

    # 1 if next day's return is positive, else 0
    df["target"] = (df["return"].shift(-1) > 0).astype(int)

    # Drop rows with NaN e.g. first 9 rows for ret_10
    df = df.dropna()

    cols = [*FEATURE_COLS, "target"]
    df = df[cols].copy()  # type: ignore

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
