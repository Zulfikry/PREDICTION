# Crypto Trend Pipeline

An end-to-end data pipeline that pulls crypto OHLCV data from Binance, engineers
technical-indicator features, trains a trend classifier (up / sideways / down),
evaluates it honestly with **walk-forward backtesting**, and serves live
predictions through a FastAPI endpoint with a Streamlit dashboard on top.

This is a portfolio project focused less on "can I call scikit-learn" and more
on the parts that separate a toy notebook from something closer to
production: idempotent loads, chronological validation, a naive baseline to
contextualize accuracy, retry/backoff on a flaky external API, and tests that
actually check math, not just "did it run."

## Architecture

```
Binance REST API
      │
      ▼
 ┌─────────┐    raw OHLCV     ┌───────────┐   features + label   ┌────────┐
 │ extract │ ───────────────▶ │ transform │ ────────────────────▶│  load  │
 └─────────┘                  └───────────┘                       └────┬───┘
                                                                        │
                                                                        ▼
                                                                  PostgreSQL
                                                              (raw_prices, features)
                                                                        │
                              ┌─────────────────────────────────────────┤
                              ▼                                         ▼
                     scripts/train.py                            FastAPI (/predict)
                     (walk-forward backtest                             │
                      + final model fit)                                ▼
                              │                                 Streamlit dashboard
                              ▼
                      models/*.joblib
```

Orchestration is a [Prefect](https://www.prefect.io/) flow
(`flows/pipeline_flow.py`) that can run once manually or be deployed on a
schedule (e.g. hourly).

## Why these design choices

- **Raw → transform → load, not raw → transform in place.** Raw candles are
  stored untouched in `raw_prices`. If a bug is later found in feature
  engineering, features can be recomputed from raw data without re-fetching
  from Binance.
- **Chronological splits everywhere, never random.** Rolling-window features
  make adjacent rows highly correlated; a random train/test split leaks
  near-duplicate information across the split and inflates accuracy. Both
  `train_chronological` and `walk_forward_backtest` split by time only.
- **Walk-forward backtest, not a single holdout.** A single train/test split
  answers "how did it do on one window?" Walk-forward retrains on an
  expanding history and evaluates on the next block, repeated across folds,
  which is closer to how the model would actually be used and reveals
  whether performance is consistent or a fluke of one period.
- **A naive baseline is reported alongside accuracy.** "70% accurate" means
  nothing if predicting the majority class alone gets 65%. `edge_over_baseline`
  is the number that actually matters here.
- **Sideways is a real label, not just up/down.** A `flat_threshold` filters
  out noise-level moves so the model isn't rewarded for calling tiny wiggles
  a "trend."
- **Retries with exponential backoff on the extract layer**, since public
  rate-limited APIs (Binance returns 429/418) are the most common source of
  pipeline flakiness in practice.
- **Upserts, not inserts**, so re-running the pipeline over overlapping time
  windows is idempotent instead of erroring or duplicating rows.

## Honest limitations

This project is deliberately upfront about what it doesn't do, because
overclaiming is the fastest way to lose credibility with anyone technical
reviewing it:

- It predicts a **short-horizon directional label from technical indicators
  only** — no order book depth, no on-chain data, no news/sentiment, no
  macro factors. Real price movement is driven by far more than moving
  averages and RSI.
- Backtested accuracy on historical data is **not a guarantee of future
  performance**, and the backtest here doesn't model transaction costs,
  slippage, or execution latency — it measures classification accuracy, not
  trading profitability.
- The API deliberately returns a `disclaimer` field on every prediction.
  This is not investment advice.

## Project layout

```
crypto-trend-pipeline/
├── src/
│   ├── config.py              # env-driven settings
│   ├── extract/binance_client.py
│   ├── transform/indicators.py
│   ├── db/{models.py,database.py}
│   ├── models/trend_classifier.py
│   ├── backtest/engine.py
│   └── api/main.py
├── flows/pipeline_flow.py     # Prefect orchestration
├── scripts/train.py           # backtest + fit + save model
├── dashboard/app.py           # Streamlit UI
├── tests/                     # pytest, mocked HTTP, no live network needed
├── docker-compose.yml         # Postgres + API
├── Dockerfile
├── .github/workflows/ci.yml   # lint + typecheck + test on push
└── pyproject.toml
```

## Running it

### 1. Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env   # edit if needed
```

### 2. Start Postgres

```bash
docker compose up -d db
```

### 3. Run the pipeline once (extract → transform → load)

```bash
python flows/pipeline_flow.py
```

Run it a few times over a few hours/days to accumulate enough history to
train on — 200+ candles per symbol minimum, more is better.

### 4. Backtest and train

```bash
python scripts/train.py
```

This prints walk-forward fold results and the final chronological holdout
report, then saves models to `models/`.

### 5. Serve predictions

```bash
uvicorn src.api.main:app --reload
```

```bash
curl "http://localhost:8000/predict?symbol=BTCUSDT"
```

### 6. Dashboard

```bash
streamlit run dashboard/app.py
```

### Or run everything in Docker

```bash
docker compose up --build
```

## Testing

```bash
pytest --cov=src --cov-report=term-missing
```

Tests cover: indicator math against manual calculations (SMA, RSI bounds,
volatility non-negativity), trend labeling correctness (including that it
correctly looks *forward*, not backward), per-symbol feature isolation
(no cross-symbol leakage in rolling windows), the walk-forward backtest's
fold structure and baseline comparison, and the Binance client's retry/error
handling with mocked HTTP (no real network calls in CI).

## Next steps / extension ideas

- Save one model per symbol (`models/{symbol}.joblib`) instead of overwriting
  a single generic file — `scripts/train.py` already saves per-symbol files;
  wiring the API to load the matching one is a small follow-up.
- Add `mlflow` for model versioning and experiment tracking across backtest runs.
- Add a drift-monitoring job that compares live prediction distribution
  against the training distribution over time.
- Extend features with order-book imbalance or funding rate data from
  Binance's other endpoints.
