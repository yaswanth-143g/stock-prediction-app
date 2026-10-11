"""
quant/regime_classifier.py
===========================
Market Regime Classifier & Hybrid Signal Engine
-----------------------------------------------
Classifies current market regimes (STRONG_BULL, BULL, SIDEWAYS, BEAR, STRONG_BEAR, HIGH_VOLATILITY)
using trend, momentum, and ATR volatility, and generates hybrid directional probability outputs.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


class MarketRegimeClassifier:
    """
    Classifies market conditions to adapt signal thresholds and ensemble weights.
    """

    @staticmethod
    def classify_regime(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates market regime from price indicators (RSI, MA20, MA50, Volatility, ATR).
        """
        if df is None or df.empty or len(df) < 20:
            return {
                "regime": "SIDEWAYS",
                "volatility_state": "NORMAL",
                "trend_score": 0.0,
                "regime_confidence": 0.50
            }

        close = df["Close"].values
        last_close = close[-1]
        
        # Moving Averages
        ma20 = np.mean(close[-20:])
        ma50 = np.mean(close[-50:]) if len(close) >= 50 else ma20

        # Returns & Volatility
        returns = np.diff(close) / (close[:-1] + 1e-9)
        vol_20d = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.15

        # Trend Score calculation (-1.0 to +1.0)
        dist_ma20 = (last_close - ma20) / (ma20 + 1e-9)
        dist_ma50 = (last_close - ma50) / (ma50 + 1e-9)
        trend_score = np.clip((dist_ma20 * 0.6 + dist_ma50 * 0.4) * 10.0, -1.0, 1.0)

        # Volatility Classification
        if vol_20d > 0.30:
            vol_state = "HIGH_VOLATILITY"
        elif vol_20d < 0.12:
            vol_state = "LOW_VOLATILITY"
        else:
            vol_state = "NORMAL"

        # Regime Decision Tree
        if trend_score >= 0.5:
            regime = "STRONG_BULL"
        elif trend_score >= 0.15:
            regime = "BULL"
        elif trend_score <= -0.5:
            regime = "STRONG_BEAR"
        elif trend_score <= -0.15:
            regime = "BEAR"
        elif vol_state == "HIGH_VOLATILITY":
            regime = "HIGH_VOLATILITY"
        else:
            regime = "SIDEWAYS"

        confidence = float(np.abs(trend_score) * 0.5 + 0.5)

        return {
            "regime": regime,
            "volatility_state": vol_state,
            "trend_score": float(trend_score),
            "volatility_annualized": float(vol_20d),
            "regime_confidence": float(confidence)
        }


def compute_hybrid_signal(predicted_return_pct: float, prob_rise: float, confidence: float) -> Dict[str, Any]:
    """
    Combines Directional Probability P(UP) with Expected Return Regression output.
    """
    if prob_rise >= 65.0 and predicted_return_pct >= 1.5:
        signal = "STRONG BUY"
    elif prob_rise >= 55.0 and predicted_return_pct >= 0.5:
        signal = "BUY"
    elif prob_rise <= 35.0 and predicted_return_pct <= -1.5:
        signal = "STRONG SELL"
    elif prob_rise <= 45.0 and predicted_return_pct <= -0.5:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "prob_rise_pct": float(prob_rise),
        "prob_fall_pct": float(100.0 - prob_rise),
        "expected_return_pct": float(predicted_return_pct),
        "overall_confidence_pct": float(confidence)
    }
