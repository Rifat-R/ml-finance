import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Create the router
router = APIRouter()

MODEL_PATH = "lgbm_direction.pkl"

try:
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["feature_cols"]
    trained_ticker = artifact["ticker"]
except Exception as e:
    raise RuntimeError(f"Could not load model '{MODEL_PATH}'. Error: {e}")


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


def build_features_from_closes(closes: list[float]) -> np.ndarray:
    """
    Convert a sequence of close prices into feature vector:
    ret_1, ret_3, ret_5, ret_10, vol_10
    """
    closes_arr = np.array(closes, dtype=float)

    if len(closes_arr) < 11:
        raise ValueError("Need at least 11 closing prices for feature extraction.")

    # Returns
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


def _predict_from_features(closes: list[float]) -> PredictionResponse:
    try:
        X = build_features_from_closes(closes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    proba = model.predict_proba(X)[0]

    prob_down = float(proba[0])
    prob_up = float(proba[1])
    direction = "up" if prob_up >= 0.5 else "down"

    return PredictionResponse(direction=direction, prob_up=prob_up, prob_down=prob_down)


def fetch_latest_closes(ticker: str, window: int) -> list[float]:
    """
    Fetch recent daily close prices using yfinance.
    """
    data = yf.download(ticker, period="180d", interval="1d", progress=False)
    if data.empty:
        raise HTTPException(status_code=502, detail=f"Could not download closes for '{ticker}'.")

    # yfinance sometimes returns MultiIndex columns; normalise to a Series of closes
    closes_slice = None
    if isinstance(data.columns, pd.MultiIndex):
        close_columns = [col for col in data.columns if isinstance(col, tuple) and "Close" in col]
        if close_columns:
            closes_slice = data[close_columns[0]]
    elif "Close" in data.columns:
        closes_slice = data["Close"]

    if closes_slice is None:
        raise HTTPException(status_code=502, detail=f"Downloaded data did not include close prices for '{ticker}'.")

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


@router.get("/predict-info")
def predict_info():
    return {
        "trained_on_ticker": trained_ticker,
        "features_expected": feature_cols,
    }


@router.post("/predict-direction-from-ticker", response_model=PredictionWithCloses)
def predict_direction_from_ticker(request: TickerRequest):
    """
    Fetch the latest closes for a ticker with yfinance and run the predictor.
    """
    closes = fetch_latest_closes(request.ticker, window=request.window)
    base_prediction = _predict_from_features(closes)

    return PredictionWithCloses(
        ticker=request.ticker.upper(),
        closes_used=closes,
        **base_prediction.model_dump(),
    )
