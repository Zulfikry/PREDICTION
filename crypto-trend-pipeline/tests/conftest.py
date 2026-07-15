from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Synthetic OHLCV data with a clear upward drift, for deterministic tests."""
    rng = np.random.default_rng(42)
    n = 300
    times = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")

    drift = np.linspace(0, 50, n)
    noise = rng.normal(0, 1, n).cumsum()
    close = 100 + drift + noise
    close = np.clip(close, 1, None)

    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(100, 1000, n)

    return pd.DataFrame({
        "open_time": times,
        "close_time": times + pd.Timedelta(hours=1),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
        "num_trades": rng.integers(50, 500, n),
        "symbol": "BTCUSDT",
        "interval": "1h",
        "fetched_at": pd.Timestamp.now(tz="UTC"),
    })
