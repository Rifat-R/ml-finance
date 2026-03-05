from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd


class Feature(ABC):
    """A feature that can be computed as a column over a full returns series."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def compute_series(self, returns: pd.Series) -> pd.Series:
        """Return a Series aligned with `returns` (same index/length)."""
        ...

    def compute_last(self, returns: pd.Series) -> float:
        """Compute feature value for the latest point (today)."""
        s = self.compute_series(returns)
        return float(s.iloc[-1])


@dataclass(frozen=True)
class RollingMeanReturn(Feature):
    window: int

    @property
    def name(self) -> str:
        return f"ret_{self.window}"

    def compute_series(self, returns: pd.Series) -> pd.Series:
        if self.window == 1:
            return returns
        return returns.rolling(self.window).mean()


@dataclass(frozen=True)
class RollingStdReturn(Feature):
    window: int

    @property
    def name(self) -> str:
        return f"vol_{self.window}"

    def compute_series(self, returns: pd.Series) -> pd.Series:
        return returns.rolling(self.window).std()


FEATURES: list[Feature] = [
    RollingMeanReturn(1),
    RollingMeanReturn(3),
    RollingMeanReturn(5),
    RollingMeanReturn(10),
    RollingStdReturn(10),
]

FEATURE_COLS: list[str] = [f.name for f in FEATURES]


def compute_feature_frame_from_returns(returns: pd.Series) -> pd.DataFrame:
    """Compute all feature columns for every timestamp in `returns`."""
    out = {}
    for f in FEATURES:
        out[f.name] = f.compute_series(returns)
    return pd.DataFrame(out, index=returns.index)


def create_features(df: pd.DataFrame, close_col: str = "adjClose") -> pd.DataFrame:
    """
    Training features over the full history.
    Outputs columns: FEATURE_COLS + target
    """
    df = df.copy()

    df["return"] = df[close_col].pct_change()

    feat_df = compute_feature_frame_from_returns(df["return"])
    df = df.join(feat_df)

    # next-day direction target
    df["target"] = (df["return"].shift(-1) > 0).astype(int)

    # drop rows where any feature/target is NaN (initial rolling + last target)
    df = df.dropna(subset=FEATURE_COLS + ["target"])

    return df[FEATURE_COLS + ["target"]].copy()


def build_features_from_closes(closes: Sequence[float]) -> np.ndarray:
    """
    Inference features for "today" from the most recent closes.
    Uses SAME feature definitions as training (no duplicated formulas).
    """
    closes_arr = np.asarray(closes, dtype=float)

    if closes_arr.size < 2:
        raise ValueError("Need at least 2 closing prices to compute returns.")

    returns = pd.Series(closes_arr[1:] / closes_arr[:-1] - 1.0)

    # Ensure we have enough history for the largest window
    max_window = 1
    for f in FEATURES:
        if isinstance(f, (RollingMeanReturn, RollingStdReturn)):
            max_window = max(max_window, f.window)

    if len(returns) < max_window:
        raise ValueError(f"Need at least {max_window + 1} closes for these features.")

    x = np.array([[f.compute_last(returns) for f in FEATURES]], dtype=float)
    return x
