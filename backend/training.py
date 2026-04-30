import os
from typing import TypedDict
import joblib
import numpy as np
import pandas as pd

from fastapi import HTTPException
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

from .features import build_feature_frame, get_feature_cols


MODEL_DIR = "models"
TRADING_DAYS_PER_YEAR = 252

# Sentiment-enabled training window (limited by news data ~2020+)
SENTIMENT_TRAIN_START_YEAR = 2020
SENTIMENT_BACKTEST_START_YEAR = 2023
SENTIMENT_BACKTEST_END_YEAR = 2023
SENTIMENT_DATA_START = "2020-01-01"

# Price-only training window (much longer history available)
PRICE_ONLY_TRAIN_START_YEAR = 2015
PRICE_ONLY_BACKTEST_START_YEAR = 2018
PRICE_ONLY_DATA_START = "2015-01-01"


def _compute_annualized_sharpe(
    returns: pd.Series | np.ndarray,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float | None:
    values = pd.Series(returns).dropna().astype(float)
    if len(values) < 2:
        return None

    rf_per_period = risk_free_rate / periods_per_year
    excess_returns = values - rf_per_period
    volatility = float(excess_returns.std(ddof=1))
    if np.isclose(volatility, 0.0):
        return None

    sharpe = np.sqrt(periods_per_year) * float(excess_returns.mean()) / volatility
    if not np.isfinite(sharpe):
        return None
    return float(sharpe)


def _make_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=35,
        learning_rate=0.03,
        num_leaves=5,
        max_depth=2,
        min_child_samples=100,
        reg_alpha=0.5,
        reg_lambda=1.0,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        verbose=-1,
    )


class FoldMetrics(TypedDict):
    fold: int
    train_start: str | int
    train_end: str | int
    test_start: str | int
    test_end: str | int
    train_size: int
    test_size: int
    train_acc: float
    test_acc: float
    overfitting_val: float


def walk_forward_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    initial_train_size: int,
    test_size: int,
) -> dict[str, object]:
    """
    Expanding-window walk-forward validation.

    Example:
    - train on first 60%
    - test on next 10%
    - then expand training window and repeat

    Returns aggregate metrics plus per-fold metrics.
    """
    n = len(X)

    if len(y) != n:
        raise ValueError("X and y must have the same length.")

    if initial_train_size <= 0 or test_size <= 0:
        raise ValueError("initial_train_size and test_size must be positive.")

    if initial_train_size + test_size > n:
        raise ValueError("Not enough data for even one walk-forward fold.")

    folds: list[FoldMetrics] = []

    train_end = initial_train_size
    fold_num = 1

    while train_end + test_size <= n:
        test_end = train_end + test_size

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[train_end:test_end]
        y_test = y.iloc[train_end:test_end]

        model = _make_model()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        folds.append(
            {
                "fold": fold_num,
                "train_start": int(X_train.index[0])
                if np.issubdtype(type(X_train.index[0]), np.integer)
                else str(X_train.index[0]),
                "train_end": int(X_train.index[-1])
                if np.issubdtype(type(X_train.index[-1]), np.integer)
                else str(X_train.index[-1]),
                "test_start": int(X_test.index[0])
                if np.issubdtype(type(X_test.index[0]), np.integer)
                else str(X_test.index[0]),
                "test_end": int(X_test.index[-1])
                if np.issubdtype(type(X_test.index[-1]), np.integer)
                else str(X_test.index[-1]),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "train_acc": train_acc,
                "test_acc": test_acc,
                "overfitting_val": train_acc - test_acc,
            }
        )

        train_end = test_end
        fold_num += 1

    avg_train_acc = float(np.mean([f["train_acc"] for f in folds]))
    avg_test_acc = float(np.mean([f["test_acc"] for f in folds]))
    avg_overfitting = float(np.mean([f["overfitting_val"] for f in folds]))
    worst_overfitting = float(np.max([f["overfitting_val"] for f in folds]))
    overfitting_std = float(np.std([f["overfitting_val"] for f in folds]))

    return {
        "folds": folds,
        "avg_train_acc": avg_train_acc,
        "avg_test_acc": avg_test_acc,
        "avg_overfitting_val": avg_overfitting,
        "worst_overfitting_val": worst_overfitting,
        "overfitting_std": overfitting_std,
    }


def walk_forward_year_backtest(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    train_start_year: int = 2020,
    start_year: int = 2023,
    end_year: int = 2023,
) -> dict[str, object]:
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    if "next_return" not in df.columns:
        raise ValueError("Dataframe must include 'next_return' for backtesting.")

    df = df.sort_index()
    if df.empty:
        raise HTTPException(
            status_code=500, detail="No rows available for backtesting."
        )

    min_year = int(df.index.min().year)
    max_year = int(df.index.max().year)

    if min_year > train_start_year or max_year < end_year:
        raise HTTPException(
            status_code=500,
            detail=(
                "Not enough yearly history for backtest. "
                f"Need data from {train_start_year} to {end_year}, "
                f"but got {min_year} to {max_year}."
            ),
        )

    effective_start_year = start_year
    effective_end_year = end_year

    if effective_start_year > effective_end_year:
        raise HTTPException(
            status_code=500,
            detail=(
                "Invalid yearly backtest range. "
                f"start_year={effective_start_year}, end_year={effective_end_year}."
            ),
        )

    next_dates = df.index.to_series().shift(-1)
    df = df.assign(next_date=next_dates)

    model_value = 1.0
    buy_hold_value = 1.0
    overall_curve: list[dict[str, object]] = []
    overall_model_returns: list[float] = []
    overall_buy_hold_returns: list[float] = []
    years: list[dict[str, object]] = []

    for year in range(effective_start_year, effective_end_year + 1):
        train_mask = (df.index.year >= train_start_year) & (df.index.year < year)
        test_mask = (df.index.year == year) & (df["next_date"].dt.year == year)

        if not test_mask.any():
            continue

        if not train_mask.any():
            continue

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, "target"]
        X_test = df.loc[test_mask, feature_cols]
        y_test = df.loc[test_mask, "target"]
        next_returns = df.loc[test_mask, "next_return"]

        model = _make_model()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        positions = (preds == 1).astype(float)
        strat_returns = next_returns.values * positions

        year_model_returns = pd.Series(strat_returns)
        year_buy_hold_returns = next_returns.reset_index(drop=True)
        overall_model_returns.extend(year_model_returns.tolist())
        overall_buy_hold_returns.extend(year_buy_hold_returns.tolist())

        year_model_value = 1.0
        year_buy_hold_value = 1.0
        year_curve: list[dict[str, object]] = []

        for date_idx, strat_ret, bh_ret in zip(
            X_test.index,
            strat_returns,
            next_returns.values,
        ):
            year_model_value *= 1.0 + float(strat_ret)
            year_buy_hold_value *= 1.0 + float(bh_ret)
            model_value *= 1.0 + float(strat_ret)
            buy_hold_value *= 1.0 + float(bh_ret)

            date_str = date_idx.date().isoformat()
            year_curve.append(
                {
                    "date": date_str,
                    "model": year_model_value,
                    "buy_hold": year_buy_hold_value,
                }
            )
            overall_curve.append(
                {
                    "date": date_str,
                    "model": model_value,
                    "buy_hold": buy_hold_value,
                }
            )

        years.append(
            {
                "year": year,
                "train_start": df.loc[train_mask].index[0].date().isoformat(),
                "train_end": df.loc[train_mask].index[-1].date().isoformat(),
                "test_start": X_test.index[0].date().isoformat(),
                "test_end": X_test.index[-1].date().isoformat(),
                "train_size": len(X_train),
                "test_size": len(X_test),
                "accuracy": float(accuracy_score(y_test, preds)),
                "model_return": float(year_model_value - 1.0),
                "buy_hold_return": float(year_buy_hold_value - 1.0),
                "model_sharpe": _compute_annualized_sharpe(year_model_returns),
                "buy_hold_sharpe": _compute_annualized_sharpe(year_buy_hold_returns),
                "curve": year_curve,
            }
        )

    if not years:
        raise HTTPException(
            status_code=500,
            detail=(
                "No backtest years available after aligning to available data range."
            ),
        )

    overall_model_sharpe = _compute_annualized_sharpe(np.array(overall_model_returns))
    overall_buy_hold_sharpe = _compute_annualized_sharpe(
        np.array(overall_buy_hold_returns)
    )

    return {
        "start_year": effective_start_year,
        "end_year": effective_end_year,
        "overall": {
            "model_return": float(model_value - 1.0),
            "buy_hold_return": float(buy_hold_value - 1.0),
            "model_sharpe": overall_model_sharpe,
            "buy_hold_sharpe": overall_buy_hold_sharpe,
            "curve": overall_curve,
        },
        "years": years,
    }


def train_model_for_ticker(
    ticker: str, *, use_sentiment: bool = True
) -> dict[str, object]:
    """
    Train a LightGBM model for the given ticker and evaluate it with walk-forward validation.
    Also fits one final model on all available data for later inference.
    """
    feature_cols = get_feature_cols(use_sentiment)

    if use_sentiment:
        data_start = SENTIMENT_DATA_START
        train_start_year = SENTIMENT_TRAIN_START_YEAR
        backtest_start_year = SENTIMENT_BACKTEST_START_YEAR
        backtest_end_year = SENTIMENT_BACKTEST_END_YEAR
    else:
        data_start = PRICE_ONLY_DATA_START
        train_start_year = PRICE_ONLY_TRAIN_START_YEAR
        backtest_start_year = PRICE_ONLY_BACKTEST_START_YEAR
        backtest_end_year = pd.Timestamp.today().year - 1

    df = build_feature_frame(ticker, use_sentiment=use_sentiment, start=data_start)

    X = df.loc[:, feature_cols].copy()
    y = df["target"].copy()

    n = len(df)
    initial_train_size = int(n * 0.6)
    test_size = int(n * 0.1)

    if test_size == 0:
        raise HTTPException(
            status_code=500, detail="Not enough data for walk-forward validation."
        )

    wf = walk_forward_evaluate(
        X,
        y,
        initial_train_size=initial_train_size,
        test_size=test_size,
    )

    year_backtest = walk_forward_year_backtest(
        df,
        feature_cols=feature_cols,
        train_start_year=train_start_year,
        start_year=backtest_start_year,
        end_year=backtest_end_year,
    )

    print(
        f"WALK-FORWARD AVG TRAIN ACCURACY: {wf['avg_train_acc']:.4f}, "
        f"AVG TEST ACCURACY: {wf['avg_test_acc']:.4f}, "
        f"AVG OVERFITTING VAL: {wf['avg_overfitting_val']:.4f}, "
        f"WORST OVERFITTING VAL: {wf['worst_overfitting_val']:.4f}, "
        f"OVERFITTING STD: {wf['overfitting_std']:.4f}"
    )

    for fold in wf["folds"]:
        print(
            f"FOLD {fold['fold']}: "
            f"train_size={fold['train_size']}, test_size={fold['test_size']}, "
            f"train_acc={fold['train_acc']:.4f}, test_acc={fold['test_acc']:.4f}, "
            f"overfit={fold['overfitting_val']:.4f}"
        )

    # Final model for live prediction: fit on all available data
    final_model = _make_model()
    final_model.fit(X, y)

    artifact_local = {
        "model": final_model,
        "feature_cols": feature_cols,
        "use_sentiment": use_sentiment,
        "ticker": ticker,
        "accuracy": wf["avg_test_acc"],
        "overfitting_val": wf["avg_overfitting_val"],
        "worst_overfitting_val": wf["worst_overfitting_val"],
        "overfitting_std": wf["overfitting_std"],
        "walk_forward_avg_train_accuracy": wf["avg_train_acc"],
        "walk_forward_avg_test_accuracy": wf["avg_test_acc"],
        "walk_forward_avg_overfitting_val": wf["avg_overfitting_val"],
        "walk_forward_worst_overfitting_val": wf["worst_overfitting_val"],
        "walk_forward_overfitting_std": wf["overfitting_std"],
        "walk_forward_folds": wf["folds"],
        "walk_forward_years": year_backtest["years"],
        "walk_forward_overall": year_backtest["overall"],
        "walk_forward_start_year": year_backtest["start_year"],
        "walk_forward_end_year": year_backtest["end_year"],
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    suffix = "" if use_sentiment else "_price_only"
    model_path = os.path.join(MODEL_DIR, f"lgbm_direction_{ticker.upper()}{suffix}.pkl")
    joblib.dump(artifact_local, model_path)

    return artifact_local
