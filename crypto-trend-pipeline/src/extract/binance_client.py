"""Extract layer: pulls raw OHLCV candles from Binance's public REST API.

Design notes:
- No API key needed for public market data.
- We keep this layer "dumb" on purpose: it only fetches and lightly shapes
  raw data. Cleaning/feature engineering happens in the transform layer.
  This separation (raw -> transform -> load) mirrors how real data
  platforms avoid mutating source-of-truth data in place.
- Retries with exponential backoff protect against transient network
  issues and Binance rate limiting (HTTP 429 / 418).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

BASE_URL = "https://api.binance.com/api/v3/klines"

# Binance kline response columns, in order.
_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


class BinanceRequestError(RuntimeError):
    """Raised when Binance returns a non-recoverable error."""


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    retry=retry_if_exception_type((requests.exceptions.RequestException, BinanceRequestError)),
)
def _fetch_klines(symbol: str, interval: str, limit: int = 500,
                   start_time_ms: int | None = None) -> list[list]:
    """Fetch a single page of candles. Retries transient failures."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_time_ms is not None:
        params["startTime"] = start_time_ms

    resp = requests.get(BASE_URL, params=params, timeout=10)

    if resp.status_code == 429 or resp.status_code == 418:
        # Rate limited / banned briefly. Let tenacity back off and retry.
        raise BinanceRequestError(f"Rate limited by Binance (status={resp.status_code})")
    if resp.status_code >= 500:
        raise BinanceRequestError(f"Binance server error (status={resp.status_code})")
    resp.raise_for_status()

    data = resp.json()
    if not isinstance(data, list):
        raise BinanceRequestError(f"Unexpected Binance response shape: {data!r}")
    return data


def fetch_ohlcv(symbol: str, interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    """Fetch recent OHLCV candles for a symbol.

    Returns a DataFrame with typed columns, one row per candle, sorted
    ascending by open_time. Raw and untouched aside from type casting -
    no indicators or labels here.
    """
    raw = _fetch_klines(symbol=symbol, interval=interval, limit=limit)
    if not raw:
        logger.warning("No candles returned for %s/%s", symbol, interval)
        return pd.DataFrame(columns=_COLUMNS)

    df = pd.DataFrame(raw, columns=_COLUMNS)

    numeric_cols = ["open", "high", "low", "close", "volume",
                     "quote_asset_volume", "taker_buy_base_volume", "taker_buy_quote_volume"]
    df[numeric_cols] = df[numeric_cols].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    df["symbol"] = symbol
    df["interval"] = interval
    df["fetched_at"] = datetime.now(timezone.utc)

    return df.drop(columns=["ignore"])


def fetch_many(symbols: list[str], interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    """Fetch OHLCV for multiple symbols and concatenate.

    Failures on one symbol are logged and skipped rather than aborting the
    whole batch - a partial pipeline run is more useful than none.
    """
    frames = []
    for symbol in symbols:
        try:
            frames.append(fetch_ohlcv(symbol, interval=interval, limit=limit))
        except Exception:
            logger.exception("Failed to fetch %s after retries, skipping", symbol)
    if not frames:
        return pd.DataFrame(columns=_COLUMNS + ["symbol", "interval", "fetched_at"])
    return pd.concat(frames, ignore_index=True)
