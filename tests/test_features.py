"""Tests for feature computation logic in backend.features."""

import numpy as np
import pandas as pd
import pytest

from backend.features import (
    PRICE_FEATURE_COLS,
    SENTIMENT_FEATURE_COLS,
    CumulativeMomentum,
    PriceDistanceFromMA,
    RollingMeanReturn,
    RollingStdReturn,
    RSI,
    build_features_from_closes,
    get_feature_cols,
)


@pytest.fixture
def constant_up_closes():
    """30 daily closes that go up by exactly 1% every day."""
    return [100.0 * (1.01**i) for i in range(30)]


@pytest.fixture
def constant_returns_series(constant_up_closes):
    closes = pd.Series(constant_up_closes)
    return closes.pct_change(), closes


class TestGetFeatureCols:
    def test_with_sentiment_includes_both_groups(self):
        cols = get_feature_cols(use_sentiment=True)
        assert cols == SENTIMENT_FEATURE_COLS + PRICE_FEATURE_COLS

    def test_without_sentiment_is_price_only(self):
        cols = get_feature_cols(use_sentiment=False)
        assert cols == PRICE_FEATURE_COLS
        assert all(c not in cols for c in SENTIMENT_FEATURE_COLS)

    def test_returns_a_fresh_list(self):
        # Mutating the returned list shouldn't leak into module state.
        cols = get_feature_cols(use_sentiment=False)
        cols.append("garbage")
        assert "garbage" not in get_feature_cols(use_sentiment=False)


class TestRollingMeanReturn:
    def test_window_one_returns_input(self, constant_returns_series):
        returns, closes = constant_returns_series
        out = RollingMeanReturn(1).compute_series(returns, closes)
        pd.testing.assert_series_equal(out, returns)

    def test_window_five_mean_matches_constant(self, constant_returns_series):
        returns, closes = constant_returns_series
        out = RollingMeanReturn(5).compute_series(returns, closes)
        # All daily returns are 0.01, so any rolling mean is 0.01.
        assert pytest.approx(out.iloc[-1], rel=1e-9) == 0.01


class TestRollingStdReturn:
    def test_constant_returns_have_zero_std(self, constant_returns_series):
        returns, closes = constant_returns_series
        out = RollingStdReturn(10).compute_series(returns, closes)
        assert pytest.approx(out.iloc[-1], abs=1e-12) == 0.0


class TestCumulativeMomentum:
    def test_compounds_returns(self, constant_returns_series):
        returns, closes = constant_returns_series
        out = CumulativeMomentum(20).compute_series(returns, closes)
        # 20 days of +1% should compound to ~ (1.01**20 - 1)
        assert pytest.approx(out.iloc[-1], rel=1e-9) == 1.01**20 - 1.0


class TestPriceDistanceFromMA:
    def test_uptrend_above_ma(self, constant_returns_series):
        returns, closes = constant_returns_series
        out = PriceDistanceFromMA(20).compute_series(returns, closes)
        # Price has been rising, so close > moving average → positive distance.
        assert out.iloc[-1] > 0


class TestRSI:
    def test_constant_gain_gives_rsi_close_to_100(self, constant_returns_series):
        returns, closes = constant_returns_series
        out = RSI(14).compute_series(returns, closes)
        # No losses at all → RSI saturates near 100.
        assert out.iloc[-1] > 99.0

    def test_constant_loss_gives_rsi_close_to_zero(self):
        closes = pd.Series([100.0 * (0.99**i) for i in range(30)])
        returns = closes.pct_change()
        out = RSI(14).compute_series(returns, closes)
        assert out.iloc[-1] < 1.0


class TestBuildFeaturesFromCloses:
    def test_returns_single_row_dataframe(self, constant_up_closes):
        df = build_features_from_closes(constant_up_closes, use_sentiment=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_columns_match_price_only_when_sentiment_off(self, constant_up_closes):
        df = build_features_from_closes(constant_up_closes, use_sentiment=False)
        assert list(df.columns) == PRICE_FEATURE_COLS

    def test_columns_include_sentiment_when_on(self, constant_up_closes):
        df = build_features_from_closes(
            constant_up_closes,
            sentiment_features={
                "mean_sentiment": 0.5,
                "article_count": 10,
                "sum_sentiment": 5.0,
                "positive_ratio": 0.8,
            },
            use_sentiment=True,
        )
        assert list(df.columns) == SENTIMENT_FEATURE_COLS + PRICE_FEATURE_COLS
        assert df["mean_sentiment"].iloc[0] == 0.5
        assert df["article_count"].iloc[0] == 10
        assert df["positive_ratio"].iloc[0] == 0.8

    def test_missing_sentiment_values_default_to_zero(self, constant_up_closes):
        df = build_features_from_closes(
            constant_up_closes,
            sentiment_features=None,
            use_sentiment=True,
        )
        for col in SENTIMENT_FEATURE_COLS:
            assert df[col].iloc[0] == 0.0

    def test_raises_on_too_few_closes(self):
        with pytest.raises(ValueError, match="Need at least"):
            build_features_from_closes([1.0, 2.0, 3.0], use_sentiment=False)

    def test_no_nan_values_in_output(self, constant_up_closes):
        df = build_features_from_closes(constant_up_closes, use_sentiment=False)
        assert not df.isna().any().any()

    def test_random_closes_produce_finite_features(self):
        rng = np.random.default_rng(seed=42)
        closes = list(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=60))))
        df = build_features_from_closes(closes, use_sentiment=False)
        assert np.isfinite(df.values).all()
