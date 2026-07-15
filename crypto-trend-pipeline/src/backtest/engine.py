"""Walk-forward backtest for the trend classifier.

Why walk-forward instead of a single train/test split:
A single split answers "how good was the model on one fixed period?"
Walk-forward retrains repeatedly on an expanding history and evaluates
on the immediately following block, which is much closer to how the
model would actually be used in production (retrain periodically,
predict forward) and surfaces whether performance is consistent or
just lucky on one window.

This also reports a naive baseline (predict "sideways" every time, or
predict last-known majority class) so accuracy numbers have context -
a classifier that's "70% accurate" is meaningless if the class
distribution alone gives 65%.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.transform.indicators import FEATURE_COLUMNS

logger = logging.getLogger(__name__)


@dataclass
class FoldResult:
    fold: int
    train_size: int
    test_size: int
    accuracy: float
    f1_macro: float
    baseline_accuracy: float


@dataclass
class BacktestResult:
    folds: list[FoldResult] = field(default_factory=list)

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean([f.accuracy for f in self.folds])) if self.folds else 0.0

    @property
    def mean_baseline_accuracy(self) -> float:
        return float(np.mean([f.baseline_accuracy for f in self.folds])) if self.folds else 0.0

    @property
    def edge_over_baseline(self) -> float:
        """How much better than the naive baseline, in accuracy points.
        This is the number that actually matters - not raw accuracy.
        """
        return self.mean_accuracy - self.mean_baseline_accuracy

    def summary(self) -> str:
        lines = [
            f"Folds run:            {len(self.folds)}",
            f"Mean accuracy:        {self.mean_accuracy:.3f}",
            f"Mean baseline (naive):{self.mean_baseline_accuracy:.3f}",
            f"Edge over baseline:   {self.edge_over_baseline:+.3f}",
        ]
        return "\n".join(lines)


def walk_forward_backtest(
    features: pd.DataFrame,
    n_folds: int = 5,
    min_train_size: int = 200,
    model_factory=None,
) -> BacktestResult:
    """Run an expanding-window walk-forward backtest for one symbol's features.

    features must already be sorted chronologically and contain
    trend_label (NaN rows are dropped). Call once per symbol.
    """
    if model_factory is None:
        model_factory = lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )

    df = features.dropna(subset=[*FEATURE_COLUMNS, "trend_label"]).reset_index(drop=True)
    n = len(df)
    if n < min_train_size + n_folds:
        raise ValueError(f"Not enough rows ({n}) for {n_folds} folds with min_train_size={min_train_size}")

    fold_size = (n - min_train_size) // n_folds
    result = BacktestResult()

    for fold in range(n_folds):
        train_end = min_train_size + fold * fold_size
        test_end = min(train_end + fold_size, n)
        if test_end <= train_end:
            continue

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        X_train, y_train = train_df[FEATURE_COLUMNS], train_df["trend_label"].astype(int)
        X_test, y_test = test_df[FEATURE_COLUMNS], test_df["trend_label"].astype(int)

        model = model_factory()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Naive baseline: always predict the majority class seen in training.
        majority_class = y_train.mode().iloc[0]
        baseline_pred = np.full(len(y_test), majority_class)

        result.folds.append(FoldResult(
            fold=fold,
            train_size=len(train_df),
            test_size=len(test_df),
            accuracy=accuracy_score(y_test, y_pred),
            f1_macro=f1_score(y_test, y_pred, average="macro", zero_division=0),
            baseline_accuracy=accuracy_score(y_test, baseline_pred),
        ))

    logger.info("Walk-forward backtest complete:\n%s", result.summary())
    return result
