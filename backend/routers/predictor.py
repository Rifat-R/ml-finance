import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException
from lightgbm import LGBMClassifier
from pydantic import BaseModel, Field

from ..train_lgbm_direction import create_features, download_data

router = APIRouter()

MODEL_DIR = "models"
FEATURE_COLS = ["ret_1", "ret_3", "ret_5", "ret_10", "vol_10"]

# Cache models per ticker to avoid retraining on every request
model_cache: dict[str, dict[str, object]] = {}


class TickerRequest(BaseModel):
    ticker: str = Field(
        ...,
        description="Ticker symbol understood by yfinance, e.g. AAPL or MSFT.",
        min_length=1,
    )
    window: int = Field(
        30,
        description="Number of most recent daily closes to use (must be >= 11).",
        ge=11,
        le=300,
    )


class PredictionResponse(BaseModel):
    direction: str  # "up" or "down"
    prob_up: float
    prob_down: float


class PredictionWithCloses(PredictionResponse):
    ticker: str
    closes_used: list[float]


def build_features_from_closes(
    closes: list[float], feature_cols: list[str]
) -> np.ndarray:
    """
    Converting a sequence of close prices -> Percentage return for
    each day -> build features.
    """
    closes_arr = np.array(closes, dtype=float)

    if len(closes_arr) < 11:
        raise ValueError("Need at least 11 closing prices for feature extraction.")

    # Computing the percentage return for each day
    returns = closes_arr[1:] / closes_arr[:-1] - 1.0
    s = pd.Series(returns)

    # Create features identical to the training script
    ret_1 = s.iloc[-1]
    ret_3 = s.iloc[-3:].mean()
    ret_5 = s.iloc[-5:].mean()
    ret_10 = s.iloc[-10:].mean()
    vol_10 = s.iloc[-10:].std()

    feature_dict = {
        "ret_1": ret_1,
        "ret_3": ret_3,
        "ret_5": ret_5,
        "ret_10": ret_10,
        "vol_10": vol_10,
    }

    # Ensure correct ordering
    x = np.array([[feature_dict[c] for c in feature_cols]], dtype=float)
    return x


def _predict_from_features(
    closes: list[float], model_obj, feature_cols: list[str]
) -> PredictionResponse:
    try:
        X = build_features_from_closes(closes, feature_cols)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    proba = model_obj.predict_proba(X)[0]

    prob_down = float(proba[0])
    prob_up = float(proba[1])
    direction = "up" if prob_up >= 0.5 else "down"

    return PredictionResponse(direction=direction, prob_up=prob_up, prob_down=prob_down)


def fetch_latest_closes(ticker: str, window: int) -> list[float]:
    """
    Fetch recent daily close prices using yfinance.
    """
    data = yf.download(ticker, period="180d", interval="1d", progress=False)
    if data is None or data.empty:
        raise HTTPException(
            status_code=502, detail=f"Could not download closes for '{ticker}'."
        )

    # yfinance sometimes returns MultiIndex columns; normalise to a Series of closes
    closes_slice = None
    if isinstance(data.columns, pd.MultiIndex):
        close_columns = [
            col for col in data.columns if isinstance(col, tuple) and "Close" in col
        ]
        if close_columns:
            closes_slice = data[close_columns[0]]
    elif "Close" in data.columns:
        closes_slice = data["Close"]

    if closes_slice is None:
        raise HTTPException(
            status_code=502,
            detail=f"Downloaded data did not include close prices for '{ticker}'.",
        )

    if isinstance(closes_slice, pd.DataFrame):
        # Pick the first column if multiple tickers were returned
        closes_slice = closes_slice.iloc[:, 0]

    closes = closes_slice.dropna().tolist()
    if len(closes) < window:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough recent closes for '{ticker}'. Needed {window}, got {len(closes)}.",
        )

    return closes[-window:]


def train_model_for_ticker(ticker: str) -> dict[str, object]:
    """
    Train a new LightGBM model for the given ticker using the same pipeline as the training script.
    """
    raw = download_data(ticker)
    df = create_features(raw)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=500, detail=f"Training failed: missing features {missing}"
        )

    X = df.loc[:, FEATURE_COLS].copy()
    y = df["target"].copy()

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Using 200 trees and a lower learning rate for better generalization
    model_obj = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42,
    )
    model_obj.fit(X_train, y_train)

    artifact_local = {
        "model": model_obj,
        "feature_cols": FEATURE_COLS,
        "ticker": ticker,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"lgbm_direction_{ticker.upper()}.pkl")
    joblib.dump(artifact_local, model_path)

    return artifact_local


def load_or_train_model(ticker: str) -> dict[str, object]:
    key = ticker.upper()

    cached = model_cache.get(key)
    if cached:
        return cached

    disk_path = os.path.join(MODEL_DIR, f"lgbm_direction_{key}.pkl")
    if os.path.exists(disk_path):
        try:
            artifact_local = joblib.load(disk_path)
            model_cache[key] = {
                "model": artifact_local["model"],
                "feature_cols": artifact_local["feature_cols"],
            }
            return model_cache[key]
        except Exception:
            # Fall back to retraining if loading fails
            pass

    artifact_local = train_model_for_ticker(ticker)
    model_cache[key] = {
        "model": artifact_local["model"],
        "feature_cols": artifact_local["feature_cols"],
    }
    return model_cache[key]


@router.get("/predict-info")
def predict_info():
    return {
        "features_expected": FEATURE_COLS,
        "cached_models": list(model_cache.keys()),
    }


@router.post("/predict-direction-from-ticker", response_model=PredictionWithCloses)
def predict_direction_from_ticker(request: TickerRequest):
    """
    Fetch the latest closes for a ticker with yfinance and run the predictor.
    """
    closes = fetch_latest_closes(request.ticker, window=request.window)
    model_entry = load_or_train_model(request.ticker)
    base_prediction = _predict_from_features(
        closes, model_entry["model"], model_entry["feature_cols"]  # type: ignore
    )

    return PredictionWithCloses(
        ticker=request.ticker.upper(),
        closes_used=closes,
        **base_prediction.model_dump(),
    )
