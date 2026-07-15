"""Prefect flow: extract -> transform -> load, schedulable and retryable.

Run once manually:      python flows/pipeline_flow.py
Deploy on a schedule:    prefect deployment build flows/pipeline_flow.py:pipeline_flow \
                              -n "hourly-refresh" --cron "0 * * * *"
"""
from __future__ import annotations

import logging

from prefect import flow, get_run_logger, task

from src.config import settings
from src.db.database import init_db, save_features, save_raw_prices
from src.extract.binance_client import fetch_many
from src.transform.indicators import build_features

logging.basicConfig(level=logging.INFO)


@task(retries=3, retry_delay_seconds=30)
def extract_task() -> "pd.DataFrame":  # noqa: F821
    logger = get_run_logger()
    df = fetch_many(settings.symbols, interval=settings.interval, limit=500)
    logger.info("Extracted %d raw candles across %d symbols", len(df), len(settings.symbols))
    return df


@task
def transform_task(raw) -> "pd.DataFrame":  # noqa: F821
    logger = get_run_logger()
    features = build_features(raw, label_horizon=settings.label_horizon)
    logger.info("Built %d feature rows", len(features))
    return features


@task
def load_task(raw, features) -> None:
    logger = get_run_logger()
    n_raw = save_raw_prices(raw)
    n_feat = save_features(features)
    logger.info("Loaded %d raw rows, %d feature rows", n_raw, n_feat)


@flow(name="crypto-trend-pipeline")
def pipeline_flow() -> None:
    init_db()
    raw = extract_task()
    if raw.empty:
        get_run_logger().warning("No data extracted this run; skipping transform/load.")
        return
    features = transform_task(raw)
    load_task(raw, features)


if __name__ == "__main__":
    pipeline_flow()
