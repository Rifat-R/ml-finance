import os
from datetime import date, timedelta

import joblib
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException
from newsapi import NewsApiClient
from pydantic import BaseModel, Field

from ..features import (
    FEATURE_COLS,
    SENTIMENT_FEATURE_COLS,
    build_features_from_closes,
)

from ..training import MODEL_DIR, train_model_for_ticker


router = APIRouter()

NEWS_LOOKBACK_DAYS = 1
SENTIMENT_BATCH_SIZE = 16

_news_api_client: NewsApiClient | None = None
_news_client_unavailable = False
_sentiment_pipeline = None


class TickerRequest(BaseModel):
    ticker: str = Field(
        ...,
        description="Ticker symbol understood by yfinance, e.g. AAPL or MSFT.",
        min_length=1,
    )
    window: int = Field(
        30,
        description="Number of most recent daily closes to use (must be >= 21).",
        ge=21,
        le=300,
    )


class PredictionResponse(BaseModel):
    direction: str  # "up" or "down"
    prob_up: float
    prob_down: float


class PredictionData(PredictionResponse):
    ticker: str
    closes_used: list[float]
    accuracy: float
    overfitting_val: float
    sentiment_features: dict[str, float]


class BacktestPoint(BaseModel):
    date: str
    model: float
    buy_hold: float


class BacktestYear(BaseModel):
    year: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_size: int
    test_size: int
    accuracy: float
    model_return: float
    buy_hold_return: float
    model_sharpe: float | None = None
    buy_hold_sharpe: float | None = None
    curve: list[BacktestPoint]


class BacktestOverall(BaseModel):
    model_return: float
    buy_hold_return: float
    model_sharpe: float | None = None
    buy_hold_sharpe: float | None = None
    curve: list[BacktestPoint]


class BacktestResponse(BaseModel):
    ticker: str
    start_year: int
    end_year: int
    overall: BacktestOverall
    years: list[BacktestYear]


class TickerInfoResponse(BaseModel):
    ticker: str
    name: str | None = None
    sector: str | None = None
    country: str | None = None
    website: str | None = None
    summary: str | None = None


def _zero_sentiment_features() -> dict[str, float]:
    return {col: 0.0 for col in SENTIMENT_FEATURE_COLS}


def _get_news_api_client() -> NewsApiClient | None:
    global _news_api_client

    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="News API key not configured. Set the NEWS_API_KEY environment variable to enable live sentiment features.",
        )

    _news_api_client = NewsApiClient(api_key=api_key)
    return _news_api_client


def _get_sentiment_pipeline():
    global _sentiment_pipeline

    if _sentiment_pipeline is None:
        from transformers import pipeline

        _sentiment_pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
        )

    return _sentiment_pipeline


def _label_score(result: dict[str, object]) -> float:
    label = str(result.get("label", "")).lower()
    score = float(result.get("score", 0.0))  # type: ignore

    if label == "positive":
        return score
    if label == "negative":
        return -score
    return 0.0


def extract_article_titles(response: dict[str, object]) -> list[str]:
    articles = response.get("articles")

    titles: list[str] = []
    for article in articles:
        title = article.get("title")
        title.append(title)

    return titles


def _build_live_sentiment_features(ticker: str) -> dict[str, float]:
    client = _get_news_api_client()
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Failed to initialize News API client",
        )

    start_date = (date.today() - timedelta(days=NEWS_LOOKBACK_DAYS)).isoformat()

    try:
        response = client.get_everything(
            q=ticker.strip().upper(),
            from_param=start_date,
            language="en",
        )
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Failed to fetch news articles for sentiment analysis",
        )

    titles = extract_article_titles(response)
    if not titles:
        raise HTTPException(
            status_code=404,
            detail=f"No news articles found for '{ticker}' in the last {NEWS_LOOKBACK_DAYS} days.",
        )

    classifier = _get_sentiment_pipeline()
    results: list[dict[str, object]] = []
    for i in range(0, len(titles), SENTIMENT_BATCH_SIZE):
        batch_texts = titles[i : i + SENTIMENT_BATCH_SIZE]
        batch_results = classifier(
            batch_texts,
            truncation=True,
            max_length=256,
        )
        if isinstance(batch_results, list):
            results.extend(r for r in batch_results if isinstance(r, dict))

    if not results:
        raise HTTPException(
            status_code=502,
            detail="Sentiment analysis failed to produce results",
        )

    sentiments = [_label_score(result) for result in results]
    article_count = float(len(sentiments))
    sum_sentiment = float(sum(sentiments))
    mean_sentiment = sum_sentiment / article_count
    positive_ratio = float(sum(1 for s in sentiments if s > 0.0) / article_count)

    return {
        "mean_sentiment": mean_sentiment,
        "article_count": article_count,
        "sum_sentiment": sum_sentiment,
        "positive_ratio": positive_ratio,
    }


def _get_expected_feature_cols(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    if not all(isinstance(col, str) for col in value):
        return None
    return value


def _predict_from_features(
    ticker: str,
    closes: list[float],
    model_obj,
    expected_feature_cols: list[str] | None = None,
) -> tuple[PredictionResponse, dict[str, float]]:
    sentiment_features = _build_live_sentiment_features(ticker)

    try:
        X = build_features_from_closes(closes, sentiment_features=sentiment_features)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if expected_feature_cols:
        missing_cols = [col for col in expected_feature_cols if col not in X.columns]
        for col in missing_cols:
            X[col] = 0.0
        X = X[expected_feature_cols]

    proba = model_obj.predict_proba(X)[0]

    prob_down = float(proba[0])
    prob_up = float(proba[1])
    direction = "up" if prob_up >= 0.5 else "down"

    return (
        PredictionResponse(direction=direction, prob_up=prob_up, prob_down=prob_down),
        sentiment_features,
    )


def fetch_latest_closes(ticker: str, window: int) -> list[float]:
    """
    Fetch recent daily close prices using yfinance.
    """
    data = yf.download(ticker, period="1y", interval="1d", progress=False)
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


def load_or_train_model(
    ticker: str, *, require_backtest: bool = False
) -> dict[str, object]:
    key = ticker.upper()

    disk_path = os.path.join(MODEL_DIR, f"lgbm_direction_{key}.pkl")
    if os.path.exists(disk_path):
        try:
            artifact_local = joblib.load(disk_path)
            if require_backtest and "walk_forward_overall" not in artifact_local:
                raise ValueError("Missing walk-forward backtest in artifact")
            return artifact_local
        except Exception:
            # Fall back to retraining if loading fails
            pass

    # Big API request to tiingo
    artifact_local = train_model_for_ticker(ticker)
    return artifact_local


@router.get("/predict-info")
def predict_info():
    return {
        "features_expected": FEATURE_COLS,
    }


@router.get("/backtest-walk-forward", response_model=BacktestResponse)
def backtest_walk_forward(ticker: str):
    if not ticker or not ticker.strip():
        raise HTTPException(status_code=400, detail="Ticker symbol is required.")

    model_entry = load_or_train_model(ticker, require_backtest=True)

    overall = model_entry.get("walk_forward_overall")
    years = model_entry.get("walk_forward_years")
    start_year = model_entry.get("walk_forward_start_year")
    end_year = model_entry.get("walk_forward_end_year")

    if not overall or not years:
        raise HTTPException(
            status_code=500,
            detail="Backtest results missing; retrain the model.",
        )

    return BacktestResponse(
        ticker=ticker.strip().upper(),
        start_year=int(start_year),
        end_year=int(end_year),
        overall=overall,
        years=years,
    )


def _ticker_exists(ticker: str) -> bool:
    """
    Check if a ticker exists by attempting to download minimal data.
    """
    try:
        data = yf.Ticker(ticker).history(period="1d")
        return data is not None and not data.empty
    except Exception:
        return False


@router.get("/ticker-info", response_model=TickerInfoResponse)
def ticker_info(ticker: str):
    """
    Lightweight ticker metadata using yfinance.
    """
    if not ticker or not ticker.strip():
        raise HTTPException(status_code=400, detail="Ticker symbol is required.")

    symbol = ticker.strip()
    if not _ticker_exists(symbol):
        raise HTTPException(
            status_code=404, detail=f"Ticker '{symbol}' does not exist."
        )

    try:
        t = yf.Ticker(symbol)
        info = t.get_info()
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Failed to fetch info for '{symbol}': {exc}"
        ) from exc

    if not info:
        raise HTTPException(status_code=404, detail=f"No info found for '{symbol}'.")

    return TickerInfoResponse(
        ticker=symbol.upper(),
        name=info.get("shortName") or info.get("longName"),
        sector=info.get("sector"),
        country=info.get("country"),
        website=info.get("website"),
        summary=info.get("longBusinessSummary"),
    )


@router.post("/predict-direction-from-ticker", response_model=PredictionData)
def predict_direction_from_ticker(request: TickerRequest):
    """
    Fetch the latest closes for a ticker with yfinance and run the predictor.
    """
    closes = fetch_latest_closes(request.ticker, window=request.window)
    model_entry = load_or_train_model(request.ticker)
    expected_feature_cols = _get_expected_feature_cols(model_entry.get("feature_cols"))
    base_prediction, sentiment_features = _predict_from_features(
        request.ticker,
        closes,
        model_entry["model"],
        expected_feature_cols=expected_feature_cols,
    )

    accuracy = model_entry.get("accuracy", 0.0)
    overfitting_val = model_entry.get("overfitting_val", 0.0)

    print(f"ACCURACY FROM PREDICT DIRECTION FROM TICKER: {model_entry.get('accuracy')}")

    return PredictionData(
        ticker=request.ticker.upper(),
        closes_used=closes,
        accuracy=accuracy,  # type: ignore
        overfitting_val=overfitting_val,  # type: ignore
        sentiment_features=sentiment_features,
        **base_prediction.model_dump(),
    )
