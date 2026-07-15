"""Centralized configuration loaded from environment variables.

Keeping this in one place means every module (extract, transform, api,
dashboard) reads settings the same way instead of scattering os.getenv calls.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _split_symbols(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql+psycopg2://pipeline:pipeline@localhost:5432/crypto_pipeline"
        )
    )
    symbols: list[str] = field(
        default_factory=lambda: _split_symbols(os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT"))
    )
    interval: str = field(default_factory=lambda: os.getenv("INTERVAL", "1h"))
    label_horizon: int = field(default_factory=lambda: int(os.getenv("LABEL_HORIZON", "4")))
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))


settings = Settings()
