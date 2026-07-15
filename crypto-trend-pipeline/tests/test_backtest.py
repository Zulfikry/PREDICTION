from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import walk_forward_backtest
from src.transform.indicators import build_features


def _make_trending_features(n=500, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    drift = np.linspace(0, 80, n)
    noise = rng.normal(0, 1.5, n).cumsum()
    close = np.clip(100 + drift + noise, 1, None)

    raw = pd.DataFrame({
        "open_time": times,
        "close": close,
        "high": close + rng.uniform(0, 2, n),
        "low": close - rng.uniform(0, 2, n),
        "open": close + rng.normal(0, 0.5, n),
        "volume": rng.uniform(100, 1000, n),
        "symbol": "TEST",
        "interval": "1h",
    })
    return build_features(raw, label_horizon=4)


def test_walk_forward_backtest_runs_and_produces_folds():
    features = _make_trending_features()
    result = walk_forward_backtest(features, n_folds=5, min_train_size=200)
    assert len(result.folds) == 5
    for fold in result.folds:
        assert 0.0 <= fold.accuracy <= 1.0
        assert 0.0 <= fold.baseline_accuracy <= 1.0
        assert fold.train_size > 0
        assert fold.test_size > 0


def test_walk_forward_backtest_raises_on_insufficient_data():
    tiny_features = _make_trending_features(n=50)
    with pytest.raises(ValueError):
        walk_forward_backtest(tiny_features, n_folds=5, min_train_size=200)


def test_walk_forward_train_sizes_expand_across_folds():
    """Expanding-window design: each fold's training set should be >= the previous."""
    features = _make_trending_features()
    result = walk_forward_backtest(features, n_folds=4, min_train_size=200)
    train_sizes = [f.train_size for f in result.folds]
    assert train_sizes == sorted(train_sizes)


def test_edge_over_baseline_is_well_defined():
    features = _make_trending_features()
    result = walk_forward_backtest(features, n_folds=3, min_train_size=200)
    # Should not raise, and should be a real float within a plausible range.
    edge = result.edge_over_baseline
    assert -1.0 <= edge <= 1.0
