from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd

from backend.data.fetch_data import fetch_stock_data

NEWS_FEATURES_PATH = "backend/data/news_features.parquet"

SENTIMENT_FEATURE_COLS = [
    "mean_sentiment",
    "article_count",
    "sum_sentiment",
    "positive_ratio",
]


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

PRICE_FEATURE_COLS: list[str] = [f.name for f in FEATURES]

FEATURE_COLS = SENTIMENT_FEATURE_COLS + PRICE_FEATURE_COLS


def build_feature_frame(ticker: str) -> pd.DataFrame:
    base_df = _build_base_frame(ticker)
    sentiment_df = _build_sentiment_feature_frame(ticker)

    start = sentiment_df.index.min()
    end = sentiment_df.index.max()
    base_df = base_df.loc[(base_df.index >= start) & (base_df.index <= end)]
    merged_df = base_df.join(sentiment_df, how="left")
    merged_df[SENTIMENT_FEATURE_COLS] = merged_df[SENTIMENT_FEATURE_COLS].shift(1)
    merged_df[SENTIMENT_FEATURE_COLS] = merged_df[SENTIMENT_FEATURE_COLS].fillna(0)

    return merged_df


def build_features_from_closes(
    closes: Sequence[float], sentiment_features: dict[str, float] | None = None
) -> pd.DataFrame:
    closes_arr = np.asarray(closes, dtype=float)
    closes_series = pd.Series(closes_arr)

    max_window = max(getattr(f, "window", 1) for f in FEATURES)
    required_closes = max_window + 1

    if closes_arr.size < required_closes:
        raise ValueError(f"Need at least {required_closes} closing prices.")

    returns = closes_series.pct_change()

    row = {f.name: f.compute_last(returns, closes_series) for f in FEATURES}

    sentiment_values = sentiment_features or {}
    for col in SENTIMENT_FEATURE_COLS:
        row[col] = float(sentiment_values.get(col, 0.0))

    ordered_row = {col: row[col] for col in FEATURE_COLS}
    return pd.DataFrame([ordered_row], columns=FEATURE_COLS)


def _compute_price_features(
    returns: pd.Series,
    closes: pd.Series,
) -> pd.DataFrame:
    """Compute all feature columns for every timestamp in `returns`."""
    out = {}
    for f in FEATURES:
        out[f.name] = f.compute_series(returns, closes)
    return pd.DataFrame(out, index=returns.index)


def _build_price_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feat_df_returns = _compute_price_features(df["return"], df["adjClose"])

    merged_df = df.join(feat_df_returns)
    merged_df["target"] = (merged_df["next_return"] > 0).astype(int)

    merged_df = merged_df.dropna(
        subset=PRICE_FEATURE_COLS + ["target", "next_return", "adjClose"]
    )
    return merged_df


def _build_sentiment_feature_frame(ticker: str) -> pd.DataFrame:
    news_features_df = pd.read_parquet(NEWS_FEATURES_PATH)
    df = (
        news_features_df[news_features_df["ticker"] == ticker]
        .drop(columns=["ticker"])
        .copy()
    )

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date").sort_index()

    return df


def _build_base_frame(ticker: str) -> pd.DataFrame:
    """Fetch stock data and compute price related features and target variable."""
    df = fetch_stock_data(ticker)
    df["return"] = df["adjClose"].pct_change()
    df["next_return"] = df["return"].shift(-1)

    df = _build_price_feature_frame(df)
    return df
