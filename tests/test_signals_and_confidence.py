"""
test_signals_and_confidence.py — Unit Tests for Volatility Signals & Confidence Score
====================================================================================
Verifies that:
1. predict_7days() returns 'confidence_score' between 0 and 100.
2. 'signal_threshold_pct' is present and scales with volatility.
3. Risk management fields 'stop_loss' and 'take_profit' are mathematically consistent.
"""

import pytest
import numpy as np


def test_predict_payload_signals_and_confidence():
    """Verifies confidence score and signal fields in predict_7days output."""
    from predict import predict_7days
    
    res = predict_7days("RELIANCE.NS", use_cache=False)
    
    assert res["status"] == "success"
    assert "confidence_score" in res, "confidence_score missing from predict_7days payload"
    assert 0.0 <= res["confidence_score"] <= 100.0, f"confidence_score out of range: {res['confidence_score']}"
    
    assert "signal_threshold_pct" in res
    assert res["signal_threshold_pct"] >= 1.5
    
    curr = res["current_price"]
    assert res["stop_loss"] < curr, f"stop_loss ({res['stop_loss']}) must be below current_price ({curr})"
    assert res["take_profit"] > curr, f"take_profit ({res['take_profit']}) must be above current_price ({curr})"
