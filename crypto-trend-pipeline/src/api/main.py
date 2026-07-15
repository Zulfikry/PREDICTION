"""FastAPI service exposing trend predictions and pipeline health.

Endpoints:
  GET  /health              - liveness check
  GET  /symbols              - which symbols the pipeline tracks
  GET  /predict?symbol=...   - latest trend prediction for a symbol
  POST /refresh               - trigger extract -> transform -> load for all symbols
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import settings
from src.db.database import init_db, load_features, save_features, save_raw_prices
from src.extract.binance_client import fetch_many
from src.models.trend_classifier import load_model, predict_one
from src.transform.indicators import build_features, latest_feature_row

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Crypto Trend Pipeline API", version="0.1.0")

MODEL_PATH = Path("models/trend_classifier.joblib")


class PredictionResponse(BaseModel):
    symbol: str
    as_of: str
    label: int
    label_name: str
    confidence: float
    probabilities: dict[str, float]
    disclaimer: str = (
        "Statistical estimate from technical indicators only. Not financial "
        "advice; markets are influenced by many factors this model doesn't see."
    )


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/symbols")
def symbols() -> dict:
    return {"symbols": settings.symbols, "interval": settings.interval}


@app.post("/refresh")
def refresh() -> dict:
    """Run extract -> transform -> load for all configured symbols."""
    raw = fetch_many(settings.symbols, interval=settings.interval, limit=500)
    if raw.empty:
        raise HTTPException(status_code=502, detail="Failed to fetch data from Binance")
    n_raw = save_raw_prices(raw)

    features = build_features(raw, label_horizon=settings.label_horizon)
    n_features = save_features(features)

    return {"raw_rows_saved": n_raw, "feature_rows_saved": n_features}


@app.get("/predict", response_model=PredictionResponse)
def predict(symbol: str) -> PredictionResponse:
    symbol = symbol.upper()
    if symbol not in settings.symbols:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} is not tracked")
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not trained yet. Run training first.")

    features = load_features(symbol=symbol)
    if features.empty:
        raise HTTPException(status_code=404, detail=f"No feature data for {symbol} yet. Call /refresh first.")

    row = latest_feature_row(features, symbol)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Not enough history to compute features for {symbol}")

    model = load_model(MODEL_PATH)
    result = predict_one(model, row)

    return PredictionResponse(
        symbol=symbol,
        as_of=str(row["open_time"]),
        label=result["label"],
        label_name=result["label_name"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
    )
