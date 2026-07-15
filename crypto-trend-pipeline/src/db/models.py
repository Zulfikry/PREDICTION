"""SQLAlchemy ORM models for the three storage layers:

- RawPrice: untouched OHLCV candles, as fetched.
- Feature: engineered features + trend label, one row per candle per symbol.
- Prediction: model outputs served over time, for tracking drift later.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class RawPrice(Base):
    __tablename__ = "raw_prices"
    __table_args__ = (UniqueConstraint("symbol", "interval", "open_time", name="uq_raw_candle"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    interval: Mapped[str] = mapped_column(String(5))
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    close_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    num_trades: Mapped[int] = mapped_column(Integer)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Feature(Base):
    __tablename__ = "features"
    __table_args__ = (UniqueConstraint("symbol", "interval", "open_time", name="uq_feature_candle"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    interval: Mapped[str] = mapped_column(String(5))
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    close: Mapped[float] = mapped_column(Float)

    sma_7: Mapped[float | None] = mapped_column(Float, nullable=True)
    sma_21: Mapped[float | None] = mapped_column(Float, nullable=True)
    sma_50: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema_7: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema_21: Mapped[float | None] = mapped_column(Float, nullable=True)
    ema_50: Mapped[float | None] = mapped_column(Float, nullable=True)
    rsi_14: Mapped[float | None] = mapped_column(Float, nullable=True)
    volatility_14: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_position: Mapped[float | None] = mapped_column(Float, nullable=True)
    volume_sma_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)

    trend_label: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), index=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    predicted_label: Mapped[int] = mapped_column(Integer)
    label_name: Mapped[str] = mapped_column(String(10))
    confidence: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(50))
