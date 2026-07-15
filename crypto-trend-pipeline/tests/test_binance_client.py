from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from src.extract.binance_client import BinanceRequestError, fetch_many, fetch_ohlcv

SAMPLE_KLINE = [
    1700000000000, "100.0", "105.0", "99.0", "103.0", "1000.0",
    1700003600000, "103000.0", 500, "600.0", "61800.0", "0",
]


def _mock_response(json_data, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    mock.raise_for_status = MagicMock()
    if status_code >= 400:
        mock.raise_for_status.side_effect = requests.exceptions.HTTPError(f"status {status_code}")
    return mock


@patch("src.extract.binance_client.requests.get")
def test_fetch_ohlcv_returns_typed_dataframe(mock_get):
    mock_get.return_value = _mock_response([SAMPLE_KLINE])
    df = fetch_ohlcv("BTCUSDT", interval="1h", limit=1)

    assert len(df) == 1
    assert df["close"].dtype == float
    assert df["symbol"].iloc[0] == "BTCUSDT"
    assert pd.api.types.is_datetime64_any_dtype(df["open_time"])


@patch("src.extract.binance_client.requests.get")
def test_fetch_ohlcv_empty_response_returns_empty_dataframe(mock_get):
    mock_get.return_value = _mock_response([])
    df = fetch_ohlcv("BTCUSDT")
    assert df.empty


@patch("src.extract.binance_client.requests.get")
def test_fetch_ohlcv_raises_on_malformed_response(mock_get):
    mock_get.return_value = _mock_response({"error": "bad request"})
    with pytest.raises(BinanceRequestError):
        fetch_ohlcv("BTCUSDT")


@patch("src.extract.binance_client.requests.get")
def test_fetch_many_skips_failed_symbols_without_aborting(mock_get):
    def side_effect(*args, **kwargs):
        params = kwargs.get("params", {})
        if params.get("symbol") == "BADCOIN":
            return _mock_response({"error": "unknown symbol"})
        return _mock_response([SAMPLE_KLINE])

    mock_get.side_effect = side_effect
    df = fetch_many(["BTCUSDT", "BADCOIN", "ETHUSDT"], interval="1h", limit=1)

    # BADCOIN fails all 5 retries and is skipped; the other two succeed.
    assert set(df["symbol"].unique()) == {"BTCUSDT", "ETHUSDT"}
