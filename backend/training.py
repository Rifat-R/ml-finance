import os
import joblib
from datetime import date
import numpy as np
import pandas as pd

from fastapi import HTTPException
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

from backend.data import fetch_stock_data
from .features import FEATURE_COLS, compute_feature_frame_from_returns

MODEL_DIR = "models"


def _make_model() -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=50,
        learning_rate=0.05,
        num_leaves=7,
        max_depth=3,
        min_child_samples=50,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )


def _resolve_close_col(raw: pd.DataFrame) -> str:
    if "adjClose" in raw.columns:
        return "adjClose"
    if "close" in raw.columns:
        return "close"
    if "Close" in raw.columns:
        return "Close"
    raise HTTPException(status_code=500, detail="No close price column found.")


def _build_feature_frame(raw: pd.DataFrame, close_col: str) -> pd.DataFrame:
    df = raw.copy()
    df["return"] = df[close_col].pct_change()

    # this adds a column (next_return) where each row contains the percentage change from that day to the next day
    df["next_return"] = df["return"].shift(-1)

    feat_df = compute_feature_frame_from_returns(df["return"], df[close_col])
    df = df.join(feat_df)
    df["target"] = (df["next_return"] > 0).astype(int)

    df = df.dropna(subset=FEATURE_COLS + ["target", "next_return", close_col])
    return df


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

    folds: list[dict[str, object]] = []

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

    return {
        "folds": folds,
        "avg_train_acc": avg_train_acc,
        "avg_test_acc": avg_test_acc,
        "avg_overfitting_val": avg_overfitting,
    }


def walk_forward_year_backtest(
    df: pd.DataFrame,
    *,
    start_year: int = 2018,
    end_year: int | None = None,
) -> dict[str, object]:
    if end_year is None:
        end_year = date.today().year

    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    if "next_return" not in df.columns:
        raise ValueError("Dataframe must include 'next_return' for backtesting.")

    df = df.sort_index()
    next_dates = df.index.to_series().shift(-1)
    df = df.assign(next_date=next_dates)

    model_value = 1.0
    buy_hold_value = 1.0
    overall_curve: list[dict[str, object]] = []
    years: list[dict[str, object]] = []

    for year in range(start_year, end_year + 1):
        train_mask = df.index.year < year
        test_mask = (df.index.year == year) & (df["next_date"].dt.year == year)

        if not test_mask.any():
            continue

        if not train_mask.any():
            raise HTTPException(
                status_code=500,
                detail=f"Not enough data to train before {year}.",
            )

        X_train = df.loc[train_mask, FEATURE_COLS]
        y_train = df.loc[train_mask, "target"]
        X_test = df.loc[test_mask, FEATURE_COLS]
        y_test = df.loc[test_mask, "target"]
        next_returns = df.loc[test_mask, "next_return"]

        model = _make_model()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        positions = (preds == 1).astype(float)
        strat_returns = next_returns.values * positions

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
                "curve": year_curve,
            }
        )

    if not years:
        raise HTTPException(status_code=500, detail="No backtest years available.")

    return {
        "start_year": start_year,
        "end_year": end_year,
        "overall": {
            "model_return": float(model_value - 1.0),
            "buy_hold_return": float(buy_hold_value - 1.0),
            "curve": overall_curve,
        },
        "years": years,
    }


def train_model_for_ticker(ticker: str) -> dict[str, object]:
    """
    Train a LightGBM model for the given ticker and evaluate it with walk-forward validation.
    Also fits one final model on all available data for later inference.
    """
    raw = fetch_stock_data(ticker)
    close_col = _resolve_close_col(raw)
    df = _build_feature_frame(raw, close_col)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=500, detail=f"Training failed: missing features {missing}"
        )

    X = df.loc[:, FEATURE_COLS].copy()
    y = df["target"].copy()

    # Example: first 60% train, then evaluate in 4, 10% chunks
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

    year_backtest = walk_forward_year_backtest(df, start_year=2018)

    print(
        f"WALK-FORWARD AVG TRAIN ACCURACY: {wf['avg_train_acc']:.4f}, "
        f"AVG TEST ACCURACY: {wf['avg_test_acc']:.4f}, "
        f"AVG OVERFITTING VAL: {wf['avg_overfitting_val']:.4f}"
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
        "feature_cols": FEATURE_COLS,
        "ticker": ticker,
        "accuracy": wf["avg_test_acc"],
        "overfitting_val": wf["avg_overfitting_val"],
        "walk_forward_avg_train_accuracy": wf["avg_train_acc"],
        "walk_forward_avg_test_accuracy": wf["avg_test_acc"],
        "walk_forward_avg_overfitting_val": wf["avg_overfitting_val"],
        "walk_forward_folds": wf["folds"],
        "walk_forward_years": year_backtest["years"],
        "walk_forward_overall": year_backtest["overall"],
        "walk_forward_start_year": year_backtest["start_year"],
        "walk_forward_end_year": year_backtest["end_year"],
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"lgbm_direction_{ticker.upper()}.pkl")
    joblib.dump(artifact_local, model_path)

    return artifact_local
