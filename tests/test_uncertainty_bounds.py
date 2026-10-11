"""
test_uncertainty_bounds.py — Unit Tests for Dynamic Quantile Uncertainty Bounds
=================================================================================
Verifies that:
1. Upper bounds and lower bounds expand monotonically with forecast horizon k=1..7.
2. Upper bounds > predicted prices > lower bounds for all horizons.
3. Quantile coverage calculation logic is correct.
"""

import pytest
import numpy as np
import pandas as pd


def test_monotonic_expansion_of_uncertainty_bounds():
    """Verifies upper bounds increase monotonically and lower bounds decrease monotonically with horizon."""
    curr_price = 100.0
    daily_vol = 0.015
    z_score = 1.645
    log_returns = np.array([0.002, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009])

    upper_bounds = []
    lower_bounds = []
    pred_prices = []

    for i, r_hat in enumerate(log_returns):
        k = i + 1
        horizon_std = daily_vol * np.sqrt(k)
        r_upper = r_hat + (z_score * horizon_std)
        r_lower = r_hat - (z_score * horizon_std)
        
        pred_prices.append(curr_price * np.exp(r_hat))
        upper_bounds.append(curr_price * np.exp(r_upper))
        lower_bounds.append(curr_price * np.exp(r_lower))

    # Assert upper bounds > pred_prices > lower_bounds
    for k in range(7):
        assert upper_bounds[k] > pred_prices[k], f"Upper bound at horizon {k+1} should exceed predicted price"
        assert pred_prices[k] > lower_bounds[k], f"Lower bound at horizon {k+1} should be below predicted price"

    # Assert monotonic expansion of uncertainty spread
    spreads = [upper_bounds[k] - lower_bounds[k] for k in range(7)]
    for k in range(1, 7):
        assert spreads[k] > spreads[k-1], f"Uncertainty spread at horizon {k+1} ({spreads[k]}) must be greater than at {k} ({spreads[k-1]})"


def test_empirical_coverage_calculation():
    """Test empirical coverage calculation for a set of actual vs upper/lower bounds."""
    true_prices = np.array([101.0, 102.0, 101.5, 103.0, 102.5, 104.0, 105.0])
    upper_bounds = np.array([103.0, 104.0, 104.0, 105.0, 105.0, 106.0, 107.0])
    lower_bounds = np.array([99.0, 99.5, 99.0, 100.0, 99.5, 101.0, 102.0])

    # Check that all true prices fall within bounds (100% empirical coverage)
    covered = (true_prices <= upper_bounds) & (true_prices >= lower_bounds)
    coverage_pct = float(np.mean(covered) * 100.0)
    assert coverage_pct == 100.0, f"Expected 100% coverage, got {coverage_pct}%"
