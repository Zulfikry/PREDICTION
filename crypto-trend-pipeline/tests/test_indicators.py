from __future__ import annotations

import numpy as np
import pandas as pd

from src.transform.indicators import (
    FEATURE_COLUMNS,
    add_moving_averages,
    add_price_position,
    add_rsi,
    add_volatility,
    build_features,
    label_trend,
    latest_feature_row,
)


def test_sma_matches_manual_calculation(sample_ohlcv):
    df = add_moving_averages(sample_ohlcv, windows=(7,))
    manual = sample_ohlcv["close"].rolling(7).mean()
    pd.testing.assert_series_equal(df["sma_7"], manual, check_names=False)


def test_rsi_bounded_between_0_and_100(sample_ohlcv):
    df = add_rsi(sample_ohlcv, period=14)
    valid = df["rsi_14"].dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_rsi_high_for_strictly_increasing_prices():
    n = 50
    df = pd.DataFrame({"close": np.linspace(100, 200, n)})
    df = add_rsi(df, period=14)
    # Strictly increasing prices -> no losses -> RSI should be at/near 100.
    assert df["rsi_14"].iloc[-1] > 95


def test_price_position_within_unit_range(sample_ohlcv):
    df = add_price_position(sample_ohlcv)
    assert (df["price_position"] >= 0).all() and (df["price_position"] <= 1).all()


def test_volatility_nonnegative(sample_ohlcv):
    df = add_volatility(sample_ohlcv, window=14)
    valid = df[f"volatility_14"].dropna()
    assert (valid >= 0).all()


def test_label_trend_marks_last_rows_as_nan(sample_ohlcv):
    horizon = 4
    df = label_trend(sample_ohlcv, horizon=horizon)
    assert df["trend_label"].iloc[-horizon:].isna().all()
    assert df["trend_label"].iloc[:-horizon].notna().all()


def test_label_trend_correctly_classifies_synthetic_uptrend():
    """If price rises by exactly 1% every step, horizon-ahead label must be 'up' (2)."""
    n = 20
    close = 100 * (1.01 ** np.arange(n))
    df = pd.DataFrame({"close": close})
    df = label_trend(df, horizon=3, flat_threshold=0.002)
    assert (df["trend_label"].dropna() == 2).all()


def test_label_trend_no_lookahead_leak_into_feature_columns():
    """The label uses shift(-horizon), i.e. future data - this test asserts
    that shifting is happening the correct direction (future, not past).
    """
    close = pd.Series([100, 101, 102, 90, 89])  # drop after index 2
    df = pd.DataFrame({"close": close})
    df = label_trend(df, horizon=1, flat_threshold=0.001)
    # At index 2 (close=102), next close=90 -> big drop -> label 0 (down)
    assert df["trend_label"].iloc[2] == 0


def test_build_features_produces_all_expected_columns(sample_ohlcv):
    features = build_features(sample_ohlcv, label_horizon=4)
    for col in FEATURE_COLUMNS:
        assert col in features.columns
    assert "trend_label" in features.columns


def test_build_features_groups_by_symbol_independently():
    """Rolling windows must not leak across symbols when multiple are present."""
    df_a = pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=60, freq="h", tz="UTC"),
        "close": np.linspace(100, 110, 60),
        "high": np.linspace(101, 111, 60),
        "low": np.linspace(99, 109, 60),
        "open": np.linspace(100, 110, 60),
        "volume": np.full(60, 500.0),
        "symbol": "AAA",
        "interval": "1h",
    })
    df_b = df_a.copy()
    df_b["symbol"] = "BBB"
    df_b["close"] = np.linspace(500, 400, 60)  # opposite trend

    combined = pd.concat([df_a, df_b], ignore_index=True)
    features = build_features(combined, label_horizon=4)

    row_a = latest_feature_row(features, "AAA")
    row_b = latest_feature_row(features, "BBB")
    assert row_a is not None and row_b is not None
    # AAA trending up -> RSI should be high; BBB trending down -> RSI should be low.
    assert row_a["rsi_14"] > row_b["rsi_14"]


def test_latest_feature_row_returns_none_when_insufficient_history():
    tiny = pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
        "close": [100, 101, 102],
        "high": [101, 102, 103],
        "low": [99, 100, 101],
        "open": [100, 101, 102],
        "volume": [500, 500, 500],
        "symbol": "AAA",
        "interval": "1h",
    })
    features = build_features(tiny, label_horizon=4)
    assert latest_feature_row(features, "AAA") is None
