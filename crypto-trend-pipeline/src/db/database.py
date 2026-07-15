"""Engine/session management plus idempotent upsert helpers.

Upserts (ON CONFLICT DO UPDATE) matter here because the pipeline runs
repeatedly on overlapping time windows - without them, reruns would
either crash on duplicate keys or silently duplicate rows.
"""
from __future__ import annotations

import logging

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings
from src.db.models import Base, Feature, RawPrice

logger = logging.getLogger(__name__)

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    Base.metadata.create_all(engine)
    logger.info("Database schema ensured.")


def _upsert(session: Session, table, rows: list[dict], conflict_cols: list[str]) -> None:
    if not rows:
        return
    stmt = pg_insert(table).values(rows)
    update_cols = {c.name: c for c in stmt.excluded if c.name not in conflict_cols and c.name != "id"}
    stmt = stmt.on_conflict_do_update(index_elements=conflict_cols, set_=update_cols)
    session.execute(stmt)


def save_raw_prices(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    cols = ["symbol", "interval", "open_time", "close_time", "open", "high",
            "low", "close", "volume", "num_trades", "fetched_at"]
    rows = df[cols].to_dict(orient="records")
    with SessionLocal() as session:
        _upsert(session, RawPrice, rows, conflict_cols=["symbol", "interval", "open_time"])
        session.commit()
    logger.info("Upserted %d raw price rows", len(rows))
    return len(rows)


def save_features(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    from src.transform.indicators import FEATURE_COLUMNS

    cols = ["symbol", "interval", "open_time", "close", *FEATURE_COLUMNS, "trend_label"]
    present = [c for c in cols if c in df.columns]
    rows = df[present].where(pd.notnull(df[present]), None).to_dict(orient="records")
    with SessionLocal() as session:
        _upsert(session, Feature, rows, conflict_cols=["symbol", "interval", "open_time"])
        session.commit()
    logger.info("Upserted %d feature rows", len(rows))
    return len(rows)


def load_features(symbol: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM features"
    params = {}
    if symbol:
        query += " WHERE symbol = :symbol"
        params["symbol"] = symbol
    return pd.read_sql(query, engine, params=params)
