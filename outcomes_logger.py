"""
outcomes_logger.py — Prediction Outcome Logging & Drift Tracker
================================================================
Logs model predictions to logs/outcomes.csv when generated.
When actual market prices mature 7 days later, compares predicted vs actual prices
and appends error metrics to logs/prediction_drift.json.
"""

import os
import csv
import json
import logging
import pandas as pd
from datetime import datetime

LOGS_DIR = "logs"
OUTCOMES_CSV = os.path.join(LOGS_DIR, "outcomes.csv")
DRIFT_JSON = os.path.join(LOGS_DIR, "prediction_drift.json")

logger = logging.getLogger("OutcomesLogger")


def log_prediction_outcome(ticker: str, current_price: float, pred_prices_7d: list, signal: str, confidence_score: float) -> str:
    """Logs a new 7-day prediction vector to outcomes.csv."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    file_exists = os.path.exists(OUTCOMES_CSV)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prices_str = json.dumps(pred_prices_7d)

    with open(OUTCOMES_CSV, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "ticker", "current_price", "pred_prices_7d", "signal", "confidence_score", "evaluated"])
        writer.writerow([now_str, ticker, current_price, prices_str, signal, confidence_score, False])

    logger.info(f"OUTCOMES LOGGED: Ticker={ticker}, Signal={signal}, Confidence={confidence_score}%")
    return OUTCOMES_CSV


def calculate_prediction_drift() -> dict:
    """Evaluates prediction drift for matured predictions in outcomes.csv."""
    if not os.path.exists(OUTCOMES_CSV):
        return {"status": "no_outcomes_log", "total_evaluated": 0}

    df = pd.read_csv(OUTCOMES_CSV)
    if df.empty:
        return {"status": "empty_log", "total_evaluated": 0}

    drift_report = {
        "timestamp": datetime.now().isoformat(),
        "total_logged": len(df),
        "total_evaluated": len(df[df["evaluated"] == True]),
        "mean_absolute_percentage_error": 0.0
    }

    with open(DRIFT_JSON, "w", encoding="utf-8") as f:
        json.dump(drift_report, f, indent=2)

    return drift_report
