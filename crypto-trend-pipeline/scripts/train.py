"""Train and evaluate the trend classifier for every tracked symbol.

Usage:
    python scripts/train.py

For each symbol this:
  1. Loads features from the database (run the pipeline flow first).
  2. Runs a walk-forward backtest and prints per-fold + summary metrics.
  3. Fits a final model on all available history and saves it to disk,
     so the API can serve live predictions.

Note: one model per run is saved as models/trend_classifier.joblib,
trained on the last symbol processed by default. For a portfolio-quality
setup you'd save one model per symbol (models/{symbol}.joblib) - left as
a clear extension point, see README "Next steps".
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest.engine import walk_forward_backtest
from src.config import settings
from src.db.database import load_features
from src.models.trend_classifier import save_model, train_chronological

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    for symbol in settings.symbols:
        logger.info("=" * 60)
        logger.info("Symbol: %s", symbol)

        features = load_features(symbol=symbol)
        if features.empty:
            logger.warning("No features found for %s. Run the pipeline flow first. Skipping.", symbol)
            continue
        features = features.sort_values("open_time").reset_index(drop=True)

        try:
            backtest = walk_forward_backtest(features, n_folds=5, min_train_size=200)
        except ValueError as e:
            logger.warning("Skipping backtest for %s: %s", symbol, e)
            continue

        logger.info("Walk-forward backtest results for %s:\n%s", symbol, backtest.summary())

        try:
            result = train_chronological(features)
        except ValueError as e:
            logger.warning("Skipping final fit for %s: %s", symbol, e)
            continue

        logger.info("Final chronological holdout report for %s:\n%s", symbol, result.report)

        model_path = MODELS_DIR / f"{symbol}.joblib"
        save_model(result.model, model_path)
        # Also save under the generic name the API expects for whichever
        # symbol was trained last - see docstring note above.
        save_model(result.model, MODELS_DIR / "trend_classifier.joblib")
        logger.info("Saved model to %s", model_path)


if __name__ == "__main__":
    main()
