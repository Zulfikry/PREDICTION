"""Trend classifier: predicts whether price will be up / sideways / down
`horizon` candles from now, using engineered technical features.

Deliberately starts with Random Forest, not deep learning:
- Tabular technical-indicator data with a few thousand rows rarely
  benefits from LSTMs/transformers - tree ensembles are a stronger,
  more defensible baseline and far easier to evaluate honestly.
- The point of this project is a *correctly validated* pipeline, not
  a fancy architecture that's actually overfit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.transform.indicators import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

LABEL_NAMES = {0: "down", 1: "sideways", 2: "up"}


@dataclass
class TrainResult:
    model: RandomForestClassifier
    report: str
    confusion: np.ndarray
    n_train: int
    n_test: int


def _prepare_xy(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = features.dropna(subset=[*FEATURE_COLUMNS, "trend_label"]).copy()
    X = df[FEATURE_COLUMNS]
    y = df["trend_label"].astype(int)
    return X, y


def train_chronological(features: pd.DataFrame, test_fraction: float = 0.2) -> TrainResult:
    """Train with a chronological (not random) train/test split.

    Random splits on time-series leak future information into training
    (a row's neighbors, which share near-identical rolling-window
    features, end up on both sides of the split). Splitting by time is
    the minimum bar for an honest evaluation here.
    """
    X, y = _prepare_xy(features)
    if len(X) < 50:
        raise ValueError(f"Not enough labeled rows to train ({len(X)}). Fetch more history first.")

    split_idx = int(len(X) * (1 - test_fraction))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,  # guards against overfitting to noisy candles
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)])
    confusion = confusion_matrix(y_test, y_pred)

    logger.info("Trained on %d rows, evaluated on %d rows", len(X_train), len(X_test))
    return TrainResult(model=model, report=report, confusion=confusion,
                        n_train=len(X_train), n_test=len(X_test))


def predict_one(model: RandomForestClassifier, feature_row: pd.Series) -> dict:
    X = feature_row[FEATURE_COLUMNS].to_frame().T
    proba = model.predict_proba(X)[0]
    label = int(np.argmax(proba))
    return {
        "label": label,
        "label_name": LABEL_NAMES[label],
        "confidence": float(proba[label]),
        "probabilities": {LABEL_NAMES[i]: float(p) for i, p in enumerate(proba)},
    }


def save_model(model: RandomForestClassifier, path: str | Path) -> None:
    joblib.dump(model, path)


def load_model(path: str | Path) -> RandomForestClassifier:
    return joblib.load(path)
