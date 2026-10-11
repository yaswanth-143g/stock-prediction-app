"""
tests/test_sentiment_math.py
=============================
Unit test verifying additive news sentiment return adjustment logic.
Formula: r_adj = r_raw + alpha * S * daily_vol
Guarantees that negative sentiment (S < 0) always lowers the forecasted return (r_adj < r_raw)
even when r_raw is negative.
"""

import numpy as np
import pytest


def compute_additive_sentiment_adjustment(r_raw, sentiment_score, alpha, daily_vol):
    """
    Computes additive sentiment return adjustment.
    r_adj = r_raw + alpha * sentiment_score * daily_vol
    """
    if alpha == 0.0 or sentiment_score == 0.0:
        return r_raw
    sentiment_delta = alpha * sentiment_score * daily_vol
    return r_raw + sentiment_delta


def test_negative_return_and_negative_sentiment():
    """
    Test that for negative r_raw and negative sentiment S (with alpha > 0),
    the adjusted return r_adj is strictly lower than r_raw (r_adj < r_raw).
    """
    r_raw = np.array([-0.02, -0.015, -0.03], dtype=float)
    sentiment_score = -0.50  # Negative sentiment (Bearish news)
    alpha = 0.15             # Active alpha
    daily_vol = 0.02         # 2% daily return volatility

    r_adj = compute_additive_sentiment_adjustment(r_raw, sentiment_score, alpha, daily_vol)

    # Assert that r_adj is strictly lower than r_raw elementwise
    assert np.all(r_adj < r_raw), f"Expected r_adj < r_raw, got r_adj={r_adj}, r_raw={r_raw}"

    # Check exact delta
    expected_delta = 0.15 * (-0.50) * 0.02  # -0.0015
    np.testing.assert_allclose(r_adj - r_raw, expected_delta)


def test_positive_return_and_negative_sentiment():
    """
    Test that for positive r_raw and negative sentiment S (with alpha > 0),
    the adjusted return r_adj is also strictly lower than r_raw.
    """
    r_raw = np.array([0.02, 0.015, 0.03], dtype=float)
    sentiment_score = -0.50
    alpha = 0.15
    daily_vol = 0.02

    r_adj = compute_additive_sentiment_adjustment(r_raw, sentiment_score, alpha, daily_vol)
    assert np.all(r_adj < r_raw)


def test_alpha_zero_off_by_default():
    """
    Test that when alpha=0.0 (off by default), r_adj is identical to r_raw.
    """
    r_raw = np.array([-0.02, 0.01], dtype=float)
    sentiment_score = -0.80
    alpha = 0.0
    daily_vol = 0.02

    r_adj = compute_additive_sentiment_adjustment(r_raw, sentiment_score, alpha, daily_vol)
    np.testing.assert_allclose(r_adj, r_raw)
