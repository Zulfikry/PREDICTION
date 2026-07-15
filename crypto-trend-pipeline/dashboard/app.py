"""Streamlit dashboard: price chart + latest prediction per symbol.

Run with: streamlit run dashboard/app.py
Expects the API (src/api/main.py) to be running at API_URL.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import settings
from src.db.database import load_features

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Crypto Trend Pipeline", layout="wide")
st.title("Crypto Trend Pipeline")
st.caption(
    "Technical-indicator based trend classifier. Educational project — "
    "not financial advice."
)

symbol = st.selectbox("Symbol", settings.symbols)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"{symbol} price history")
    features = load_features(symbol=symbol)
    if features.empty:
        st.warning("No data yet. Trigger a refresh via the API (`POST /refresh`) or run the pipeline flow.")
    else:
        features = features.sort_values("open_time")
        chart_df = features.set_index("open_time")[["close", "sma_7", "sma_21", "sma_50"]]
        st.line_chart(chart_df)

        with st.expander("RSI (14)"):
            st.line_chart(features.set_index("open_time")[["rsi_14"]])

with col2:
    st.subheader("Latest prediction")
    try:
        resp = requests.get(f"{API_URL}/predict", params={"symbol": symbol}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            label_color = {"up": "green", "down": "red", "sideways": "gray"}[data["label_name"]]
            st.markdown(f"### :{label_color}[{data['label_name'].upper()}]")
            st.metric("Confidence", f"{data['confidence']:.1%}")
            st.write("As of:", data["as_of"])
            st.bar_chart(pd.Series(data["probabilities"]))
            st.caption(data["disclaimer"])
        else:
            st.error(f"API error {resp.status_code}: {resp.json().get('detail', 'unknown error')}")
    except requests.exceptions.ConnectionError:
        st.error(f"Can't reach API at {API_URL}. Is it running? (`uvicorn src.api.main:app`)")

st.divider()
st.caption(
    "Pipeline: Binance API -> feature engineering -> RandomForest classifier "
    "-> walk-forward backtested -> served via FastAPI."
)
