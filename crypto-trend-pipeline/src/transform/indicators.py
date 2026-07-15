"""Transform layer: turns raw OHLCV candles into model-ready features.

Indicators implemented from scratch (no TA-Lib dependency) so the math
is transparent and auditable - important for a portfolio piece, since
"import ta_lib; magic()" doesn't demonstrate understanding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_moving_averages(df: pd.DataFrame, windows: tuple[int, ...] = (7, 21, 50)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"sma_{w}"] = df["close"].rolling(window=w, min_periods=w).mean()
        df[f"ema_{w}"] = df["close"].ewm(span=w, adjust=False, min_periods=w).mean()
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index, Wilder's smoothing method."""
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    df[f"rsi_{period}"] = df[f"rsi_{period}"].fillna(50)  # neutral when undefined
    return df


def add_volatility(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Rolling std of log returns - a standard volatility proxy."""
    df = df.copy()
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["log_return"] = log_ret
    df[f"volatility_{window}"] = log_ret.rolling(window=window, min_periods=window).std()
    return df


def add_price_position(df: pd.DataFrame) -> pd.DataFrame:
    """Where close sits within the candle's high-low range (0=low, 1=high).

    Cheap but informative feature: captures intra-candle buying/selling
    pressure that raw OHLC alone doesn't make explicit.
    """
    df = df.copy()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["price_position"] = ((df["close"] - df["low"]) / rng).fillna(0.5)
    return df


def add_volume_features(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    df["volume_sma_ratio"] = df["volume"] / df["volume"].rolling(window, min_periods=window).mean()
    return df


def label_trend(df: pd.DataFrame, horizon: int = 4, flat_threshold: float = 0.002) -> pd.DataFrame:
    """Label each row with the price direction `horizon` candles ahead.

    0 = down, 1 = sideways, 2 = up. flat_threshold is the minimum
    fractional price change (e.g. 0.002 = 0.2%) required to count as a
    directional move rather than "sideways" - without this, tiny noise
    gets classified as a strong trend, which inflates apparent accuracy
    without adding real predictive value.

    IMPORTANT: this uses future data (shift(-horizon)) and must only be
    used for training/backtesting labels, never as a live feature.
    """
    df = df.copy()
    future_close = df["close"].shift(-horizon)
    pct_change = (future_close - df["close"]) / df["close"]

    conditions = [pct_change > flat_threshold, pct_change < -flat_threshold]
    choices = [2, 0]
    df["trend_label"] = np.select(conditions, choices, default=1)

    # Rows where we don't yet know the future (tail of the series) get NaN
    # so callers can explicitly drop them instead of silently training on
    # garbage labels - a common source of data leakage bugs.
    df.loc[future_close.isna(), "trend_label"] = np.nan
    return df


FEATURE_COLUMNS = [
    "sma_7", "sma_21", "sma_50",
    "ema_7", "ema_21", "ema_50",
    "rsi_14",
    "volatility_14",
    "price_position",
    "volume_sma_ratio",
]


def build_features(raw: pd.DataFrame, label_horizon: int = 4) -> pd.DataFrame:
    """Full transform pipeline: raw OHLCV -> engineered features + label.

    Operates per-symbol so rolling windows don't leak across symbols.
    """
    parts = []
    for symbol, group in raw.groupby("symbol", sort=False):
        g = group.sort_values("open_time").reset_index(drop=True)
        g = add_moving_averages(g)
        g = add_rsi(g)
        g = add_volatility(g)
        g = add_price_position(g)
        g = add_volume_features(g)
        g = label_trend(g, horizon=label_horizon)
        parts.append(g)

    features = pd.concat(parts, ignore_index=True)
    return features


def latest_feature_row(features: pd.DataFrame, symbol: str) -> pd.Series | None:
    """Most recent row with all feature columns populated (no NaNs),
    for making a live prediction. Excludes trend_label since it's
    unknown for the latest row by definition.
    """
    sub = features[features["symbol"] == symbol].dropna(subset=FEATURE_COLUMNS)
    if sub.empty:
        return None
    return sub.sort_values("open_time").iloc[-1]
