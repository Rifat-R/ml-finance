from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import override
import numpy as np
import pandas as pd


class Feature(ABC):
    @abstractmethod
    def compute_from_returns(self, returns: pd.Series) -> dict[str, float]:
        """Compute feature vector from a sequence of close prices."""
        pass


class Ret1Feature(Feature):
    @override
    def compute_from_returns(self, returns: pd.Series) -> dict[str, float]:
        return {"ret_1": returns.iloc[-1]}


class Ret3Feature(Feature):
    @override
    def compute_from_returns(self, returns: pd.Series) -> dict[str, float]:
        return {"ret_3": returns.iloc[-3:].mean()}


class Ret5Feature(Feature):
    @override
    def compute_from_returns(self, returns: pd.Series) -> dict[str, float]:
        return {"ret_5": returns.iloc[-5:].mean()}


class Ret10Feature(Feature):
    @override
    def compute_from_returns(self, returns: pd.Series) -> dict[str, float]:
        return {"ret_10": returns.iloc[-10:].mean()}


class Vol10Feature(Feature):
    @override
    def compute_from_returns(self, returns: pd.Series) -> dict[str, float]:
        return {"vol_10": returns.iloc[-10:].std()}


FEATURES: list[Feature] = [
    Ret1Feature(),
    Ret3Feature(),
    Ret5Feature(),
    Ret10Feature(),
    Vol10Feature(),
]


def _features_from_returns(returns: pd.Series) -> dict[str, float]:
    feature_dict = {}
    for f in FEATURES:
        feature_dict.update(f.compute_from_returns(returns))
    return feature_dict


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
