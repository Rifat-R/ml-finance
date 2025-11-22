import joblib
import numpy as np
import pandas as pd
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


class PriceWindow(BaseModel):
    closes: list[float] = Field(
        ...,
        description="List of recent daily closing prices, oldest → newest.",
        min_length=11,  # need at least 11 closes to compute 10 returns
    )


class PredictionResponse(BaseModel):
    direction: str  # "up" or "down"
    prob_up: float
    prob_down: float


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


@router.get("/predict-info")
def predict_info():
    return {
        "trained_on_ticker": trained_ticker,
        "features_expected": feature_cols,
    }


@router.post("/predict-direction", response_model=PredictionResponse)
def predict_direction(data: PriceWindow):
    """Predict next-day price direction: up/down"""
    try:
        X = build_features_from_closes(data.closes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # LightGBM prediction
    proba = model.predict_proba(X)[0]

    prob_down = float(proba[0])
    prob_up = float(proba[1])
    direction = "up" if prob_up >= 0.5 else "down"

    return PredictionResponse(
        direction=direction,
        prob_up=prob_up,
        prob_down=prob_down,
    )
