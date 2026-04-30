"""Tests for walk-forward evaluation and yearly backtest in backend.training."""

import numpy as np
import pandas as pd
import pytest

from backend.features import PRICE_FEATURE_COLS
from backend.training import (
    _compute_annualized_sharpe,
    walk_forward_evaluate,
    walk_forward_year_backtest,
)


def _synthetic_feature_frame(n_rows: int = 600, seed: int = 0) -> pd.DataFrame:
    """Build a small but realistic feature frame for backtest tests."""
    rng = np.random.default_rng(seed)
    features = {col: rng.normal(0.0, 1.0, size=n_rows) for col in PRICE_FEATURE_COLS}
    next_return = rng.normal(0.0005, 0.01, size=n_rows)
    df = pd.DataFrame(features)
    df["next_return"] = next_return
    df["target"] = (df["next_return"] > 0).astype(int)
    return df


class TestComputeAnnualizedSharpe:
    def test_returns_none_for_too_few_points(self):
        assert _compute_annualized_sharpe([0.01]) is None

    def test_returns_none_for_zero_volatility(self):
        # All identical returns → std = 0.
        assert _compute_annualized_sharpe([0.01, 0.01, 0.01, 0.01]) is None

    def test_positive_mean_gives_positive_sharpe(self):
        rng = np.random.default_rng(0)
        positive_returns = np.abs(rng.normal(0.001, 0.001, size=100))
        sharpe = _compute_annualized_sharpe(positive_returns)
        assert sharpe is not None
        assert sharpe > 0

    def test_negative_mean_gives_negative_sharpe(self):
        rng = np.random.default_rng(1)
        negative_returns = -np.abs(rng.normal(0.001, 0.001, size=100))
        sharpe = _compute_annualized_sharpe(negative_returns)
        assert sharpe is not None
        assert sharpe < 0


class TestWalkForwardEvaluate:
    def test_basic_run_produces_expected_fold_count(self):
        df = _synthetic_feature_frame(n_rows=500).reset_index(drop=True)
        X = df[PRICE_FEATURE_COLS]
        y = df["target"]

        # 300 train, then 50-row chunks → folds at 300-350, 350-400, 400-450, 450-500.
        result = walk_forward_evaluate(X, y, initial_train_size=300, test_size=50)

        assert len(result["folds"]) == 4
        assert all(f["test_size"] == 50 for f in result["folds"])
        # Train sizes should grow as the window expands.
        train_sizes = [f["train_size"] for f in result["folds"]]
        assert train_sizes == sorted(train_sizes)

    def test_metrics_are_in_unit_interval(self):
        df = _synthetic_feature_frame(n_rows=500).reset_index(drop=True)
        X = df[PRICE_FEATURE_COLS]
        y = df["target"]
        result = walk_forward_evaluate(X, y, initial_train_size=300, test_size=50)

        for key in ("avg_train_acc", "avg_test_acc"):
            assert 0.0 <= result[key] <= 1.0

    def test_raises_when_x_and_y_lengths_mismatch(self):
        df = _synthetic_feature_frame(n_rows=200).reset_index(drop=True)
        with pytest.raises(ValueError, match="same length"):
            walk_forward_evaluate(
                df[PRICE_FEATURE_COLS],
                df["target"].iloc[:-1],
                initial_train_size=100,
                test_size=20,
            )

    def test_raises_when_not_enough_data(self):
        df = _synthetic_feature_frame(n_rows=50).reset_index(drop=True)
        with pytest.raises(ValueError, match="Not enough data"):
            walk_forward_evaluate(
                df[PRICE_FEATURE_COLS],
                df["target"],
                initial_train_size=100,
                test_size=20,
            )

    def test_rejects_non_positive_sizes(self):
        df = _synthetic_feature_frame(n_rows=100).reset_index(drop=True)
        with pytest.raises(ValueError, match="must be positive"):
            walk_forward_evaluate(
                df[PRICE_FEATURE_COLS],
                df["target"],
                initial_train_size=0,
                test_size=10,
            )


class TestWalkForwardYearBacktest:
    @pytest.fixture
    def datetime_indexed_frame(self):
        # 4 years of daily-ish data → ~1000 trading days. Index starts 2020.
        n_rows = 1000
        dates = pd.bdate_range(start="2020-01-01", periods=n_rows)
        df = _synthetic_feature_frame(n_rows=n_rows, seed=7)
        df.index = dates
        return df

    def test_runs_for_single_year(self, datetime_indexed_frame):
        result = walk_forward_year_backtest(
            datetime_indexed_frame,
            feature_cols=PRICE_FEATURE_COLS,
            train_start_year=2020,
            start_year=2022,
            end_year=2022,
        )

        assert result["start_year"] == 2022
        assert result["end_year"] == 2022
        assert len(result["years"]) == 1
        year = result["years"][0]
        assert year["year"] == 2022
        assert year["test_size"] > 0
        assert 0.0 <= year["accuracy"] <= 1.0

    def test_runs_for_multiple_years(self, datetime_indexed_frame):
        result = walk_forward_year_backtest(
            datetime_indexed_frame,
            feature_cols=PRICE_FEATURE_COLS,
            train_start_year=2020,
            start_year=2022,
            end_year=2023,
        )
        assert len(result["years"]) == 2
        assert [y["year"] for y in result["years"]] == [2022, 2023]

    def test_overall_curve_length_matches_total_test_days(self, datetime_indexed_frame):
        result = walk_forward_year_backtest(
            datetime_indexed_frame,
            feature_cols=PRICE_FEATURE_COLS,
            train_start_year=2020,
            start_year=2022,
            end_year=2023,
        )
        total_test = sum(y["test_size"] for y in result["years"])
        assert len(result["overall"]["curve"]) == total_test

    def test_curve_starts_around_one(self, datetime_indexed_frame):
        result = walk_forward_year_backtest(
            datetime_indexed_frame,
            feature_cols=PRICE_FEATURE_COLS,
            train_start_year=2020,
            start_year=2022,
            end_year=2022,
        )
        first_point = result["years"][0]["curve"][0]
        # After day one the values are (1 + return) which is close to but not exactly 1.
        assert abs(first_point["model"] - 1.0) < 0.2
        assert abs(first_point["buy_hold"] - 1.0) < 0.2

    def test_raises_without_next_return_column(self):
        df = _synthetic_feature_frame(n_rows=100).reset_index(drop=True)
        df.index = pd.bdate_range(start="2020-01-01", periods=100)
        df = df.drop(columns=["next_return"])
        with pytest.raises(ValueError, match="next_return"):
            walk_forward_year_backtest(
                df,
                feature_cols=PRICE_FEATURE_COLS,
                train_start_year=2020,
                start_year=2020,
                end_year=2020,
            )
