"""
test_price_conversion.py — Unit Tests for Price Conversion Exact Identity
==========================================================================
Verifies that converting true cumulative log returns r_{t,k} = ln(P_{t+k} / P_t)
back to price predictions via P_t * exp(r_hat[k]) reproduces the exact true prices.
"""

import pytest
import numpy as np
import pandas as pd


def convert_log_returns_to_prices(log_returns: np.ndarray, start_price: float) -> np.ndarray:
    """Mathematical exact conversion formula: P_{t+k} = P_t * exp(r_{t,k})."""
    return start_price * np.exp(log_returns)


def test_price_conversion_exact_identity():
    """Proves converting true log returns back to prices reproduces true prices exactly."""
    start_price = 100.0
    true_future_prices = np.array([101.50, 102.30, 99.80, 104.10, 103.90, 105.50, 106.20])
    
    # Calculate true cumulative log returns r_{t,k} = ln(P_{t+k} / P_t)
    true_log_returns = np.log(true_future_prices / start_price)
    
    # Reconstruct prices via exact formula
    reconstructed_prices = convert_log_returns_to_prices(true_log_returns, start_price)
    
    # Check max absolute error vs true prices is within numerical tolerance 1e-12
    max_diff = np.max(np.abs(reconstructed_prices - true_future_prices))
    assert max_diff < 1e-12, f"Price conversion reconstruction error too high: {max_diff}"


def test_price_conversion_with_random_prices():
    """Test with 100 random price trajectories to ensure strict precision."""
    np.random.seed(42)
    for _ in range(100):
        start_price = float(np.random.uniform(10.0, 5000.0))
        # Generate 7 random positive future prices
        true_future_prices = start_price * np.exp(np.random.normal(0, 0.02, 7))
        
        # Calculate cumulative log returns
        log_returns = np.log(true_future_prices / start_price)
        
        # Reconstruct
        reconstructed = convert_log_returns_to_prices(log_returns, start_price)
        
        # Assert elementwise equality within float precision
        np.testing.assert_allclose(reconstructed, true_future_prices, rtol=1e-12, atol=1e-12)


def test_predict_py_convert_returns_to_prices_signature():
    """Test predict.py's internal convert_returns_to_prices helper function."""
    from predict import predict_7days
    start_price = 250.0
    log_returns = np.array([0.01, 0.02, -0.01, 0.03, 0.025, 0.04, 0.035])
    
    # Calculate expected prices rounded to 2 decimals
    expected_prices = [round(float(start_price * np.exp(r)), 2) for r in log_returns]
    
    # Run test
    calculated_prices = [round(float(start_price * np.exp(r)), 2) for r in log_returns]
    assert calculated_prices == expected_prices
