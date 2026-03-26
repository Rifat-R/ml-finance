from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd


class Feature(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def compute_series(self, returns: pd.Series, closes: pd.Series) -> pd.Series:
        """Return a Series aligned with `returns` (same index/length)."""
        ...

    def compute_last(self, returns: pd.Series, closes: pd.Series) -> float:
        """Compute feature value for tee latest point (today)."""
        s = self.compute_series(returns, closes)
        return float(s.iloc[-1])


@dataclass(frozen=True)
class RollingMeanReturn(Feature):
    window: int

    @property
    def name(self) -> str:
        return f"ret_{self.window}"

    def compute_series(self, returns: pd.Series, closes: pd.Series) -> pd.Series:
        if self.window == 1:
            return returns
        return returns.rolling(self.window).mean()


@dataclass(frozen=True)
class RollingStdReturn(Feature):
    window: int

    @property
    def name(self) -> str:
        return f"vol_{self.window}"

    def compute_series(self, returns: pd.Series, closes: pd.Series) -> pd.Series:
        return returns.rolling(self.window).std()


@dataclass(frozen=True)
class CumulativeMomentum(Feature):
    window: int

    @property
    def name(self) -> str:
        return f"mom_{self.window}"

    def compute_series(self, returns: pd.Series, closes: pd.Series) -> pd.Series:
        if self.window == 1:
            return returns
        return (1.0 + returns).rolling(self.window).apply(np.prod, raw=True) - 1.0


@dataclass(frozen=True)
class PriceDistanceFromMA(Feature):
    window: int

    @property
    def name(self) -> str:
        return f"ma_dist_{self.window}"

    def compute_series(self, returns: pd.Series, closes: pd.Series) -> pd.Series:
        ma = closes.rolling(self.window).mean()
        return closes / ma - 1.0


@dataclass(frozen=True)
class RSI(Feature):
    window: int = 14

    @property
    def name(self) -> str:
        return f"rsi_{self.window}"

    def compute_series(self, returns: pd.Series, closes: pd.Series) -> pd.Series:
        delta = closes.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / self.window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.window, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))


FEATURES: list[Feature] = [
    RollingMeanReturn(1),
    RollingMeanReturn(5),
    RollingStdReturn(10),
    CumulativeMomentum(20),
    PriceDistanceFromMA(20),
    RSI(14),
]

FEATURE_COLS: list[str] = [f.name for f in FEATURES]


def compute_feature_frame_from_returns(
    returns: pd.Series,
    closes: pd.Series,
) -> pd.DataFrame:
    """Compute all feature columns for every timestamp in `returns`."""
    out = {}
    for f in FEATURES:
        out[f.name] = f.compute_series(returns, closes)
    return pd.DataFrame(out, index=returns.index)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Training features over the full history.
    Outputs columns: FEATURE_COLS + target
    """
    df = df.copy()

    df["return"] = df["adjClose"].pct_change()

    feat_df = compute_feature_frame_from_returns(df["return"], df["adjClose"])
    df = df.join(feat_df)

    # next-day direction target
    df["target"] = (df["return"].shift(-1) > 0).astype(int)

    # drop rows where any feature/target is NaN (initial rolling + last target)
    df = df.dropna(subset=FEATURE_COLS + ["target"])

    return df[FEATURE_COLS + ["target"]].copy()


def build_features_from_closes(closes: Sequence[float]) -> pd.DataFrame:
    closes_arr = np.asarray(closes, dtype=float)
    closes_series = pd.Series(closes_arr)

    max_window = max(getattr(f, "window", 1) for f in FEATURES)
    required_closes = max_window + 1

    if closes_arr.size < required_closes:
        raise ValueError(f"Need at least {required_closes} closing prices.")

    returns = closes_series.pct_change()

    row = {f.name: f.compute_last(returns, closes_series) for f in FEATURES}
    return pd.DataFrame([row], columns=FEATURE_COLS)
