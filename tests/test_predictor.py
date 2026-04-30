"""Integration-style tests for the FastAPI predictor router.

We mock out everything that would hit the network (Tiingo, NewsAPI, FinBERT)
or train a real model, and verify the route shapes and the use_sentiment flag
plumbing.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.routers import predictor as predictor_module


class FakeModel:
    """Minimal stand-in for a fitted LGBMClassifier."""

    def __init__(self, prob_up: float = 0.7):
        self.prob_up = prob_up

    def predict_proba(self, X):
        # sklearn convention: column 0 = prob(class 0 = down), column 1 = prob(up)
        return np.array([[1.0 - self.prob_up, self.prob_up]])


def _fake_artifact(use_sentiment: bool, prob_up: float = 0.7) -> dict:
    from backend.features import get_feature_cols

    return {
        "model": FakeModel(prob_up=prob_up),
        "feature_cols": get_feature_cols(use_sentiment),
        "use_sentiment": use_sentiment,
        "ticker": "AAPL",
        "accuracy": 0.55,
        "overfitting_val": 0.05,
        "worst_overfitting_val": 0.08,
        "overfitting_std": 0.02,
        "walk_forward_overall": {
            "model_return": 0.1,
            "buy_hold_return": 0.05,
            "model_sharpe": 1.2,
            "buy_hold_sharpe": 0.8,
            "curve": [{"date": "2023-01-03", "model": 1.0, "buy_hold": 1.0}],
        },
        "walk_forward_years": [
            {
                "year": 2023,
                "train_start": "2020-01-02",
                "train_end": "2022-12-30",
                "test_start": "2023-01-03",
                "test_end": "2023-12-29",
                "train_size": 750,
                "test_size": 250,
                "accuracy": 0.55,
                "model_return": 0.1,
                "buy_hold_return": 0.05,
                "model_sharpe": 1.2,
                "buy_hold_sharpe": 0.8,
                "curve": [{"date": "2023-01-03", "model": 1.0, "buy_hold": 1.0}],
            }
        ],
        "walk_forward_start_year": 2023,
        "walk_forward_end_year": 2023,
    }


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def fake_closes():
    # 30 closes is enough for the longest feature window (20).
    return [100.0 * (1.005**i) for i in range(30)]


@pytest.fixture
def patched_dependencies(monkeypatch, fake_closes):
    """Patch network/training side effects so endpoints are pure."""

    def fake_fetch_latest_closes(ticker, window):
        return fake_closes[-window:]

    def fake_load_or_train_model(ticker, *, use_sentiment=True):
        return _fake_artifact(use_sentiment=use_sentiment)

    def fake_sentiment_features(ticker):
        return {
            "mean_sentiment": 0.4,
            "article_count": 12,
            "sum_sentiment": 4.8,
            "positive_ratio": 0.75,
        }

    monkeypatch.setattr(
        predictor_module, "fetch_latest_closes", fake_fetch_latest_closes
    )
    monkeypatch.setattr(
        predictor_module, "load_or_train_model", fake_load_or_train_model
    )
    monkeypatch.setattr(
        predictor_module, "_build_live_sentiment_features", fake_sentiment_features
    )
    return None


class TestPredictInfo:
    def test_returns_feature_names(self, client):
        response = client.get("/predict-info")
        assert response.status_code == 200
        body = response.json()
        assert "features_expected" in body
        assert isinstance(body["features_expected"], list)
        assert len(body["features_expected"]) > 0


class TestPredictDirectionFromTicker:
    def test_with_sentiment_returns_full_payload(self, client, patched_dependencies):
        response = client.post(
            "/predict-direction-from-ticker",
            json={"ticker": "aapl", "use_sentiment": True},
        )
        assert response.status_code == 200
        body = response.json()

        assert body["ticker"] == "AAPL"
        assert body["direction"] in {"up", "down"}
        assert 0.0 <= body["prob_up"] <= 1.0
        assert 0.0 <= body["prob_down"] <= 1.0
        assert pytest.approx(body["prob_up"] + body["prob_down"], abs=1e-9) == 1.0
        assert body["use_sentiment"] is True
        assert body["sentiment_features"]["article_count"] == 12

    def test_without_sentiment_omits_sentiment_features(
        self, client, patched_dependencies
    ):
        response = client.post(
            "/predict-direction-from-ticker",
            json={"ticker": "AAPL", "use_sentiment": False},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["use_sentiment"] is False
        assert body["sentiment_features"] is None

    def test_default_use_sentiment_is_true(self, client, patched_dependencies):
        response = client.post(
            "/predict-direction-from-ticker", json={"ticker": "AAPL"}
        )
        assert response.status_code == 200
        assert response.json()["use_sentiment"] is True

    def test_direction_up_when_prob_up_high(self, client, monkeypatch, fake_closes):
        def fake_fetch_latest_closes(ticker, window):
            return fake_closes[-window:]

        def fake_load_or_train_model(ticker, *, use_sentiment=True):
            return _fake_artifact(use_sentiment=use_sentiment, prob_up=0.9)

        monkeypatch.setattr(
            predictor_module, "fetch_latest_closes", fake_fetch_latest_closes
        )
        monkeypatch.setattr(
            predictor_module, "load_or_train_model", fake_load_or_train_model
        )

        response = client.post(
            "/predict-direction-from-ticker",
            json={"ticker": "AAPL", "use_sentiment": False},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["direction"] == "up"
        assert body["prob_up"] > 0.5

    def test_window_below_minimum_is_rejected(self, client):
        response = client.post(
            "/predict-direction-from-ticker",
            json={"ticker": "AAPL", "window": 5},
        )
        # FastAPI/pydantic rejects this before any handler runs.
        assert response.status_code == 422


class TestBacktestWalkForward:
    def test_returns_overall_and_years(self, client, patched_dependencies):
        response = client.get("/backtest-walk-forward?ticker=AAPL")
        assert response.status_code == 200
        body = response.json()
        assert body["ticker"] == "AAPL"
        assert body["start_year"] == 2023
        assert body["end_year"] == 2023
        assert "overall" in body
        assert isinstance(body["years"], list)
        assert len(body["years"]) >= 1

    def test_empty_ticker_is_400(self, client):
        # FastAPI parses %20 → " ", route still receives a non-empty string,
        # but our handler trims and rejects whitespace-only.
        response = client.get("/backtest-walk-forward?ticker=%20")
        assert response.status_code == 400
