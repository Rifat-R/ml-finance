import os
import joblib
import numpy as np
import pandas as pd

from fastapi import HTTPException
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

from backend.data import fetch_stock_data
from .features import FEATURE_COLS, create_features

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


def train_model_for_ticker(ticker: str) -> dict[str, object]:
    """
    Train a LightGBM model for the given ticker and evaluate it with walk-forward validation.
    Also fits one final model on all available data for later inference.
    """
    raw = fetch_stock_data(ticker)
    df = create_features(raw)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=500, detail=f"Training failed: missing features {missing}"
        )

    X = df.loc[:, FEATURE_COLS].copy()
    y = df["target"].copy()

    # Example: first 60% train, then evaluate in 10% chunks
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
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, f"lgbm_direction_{ticker.upper()}.pkl")
    joblib.dump(artifact_local, model_path)

    return artifact_local
