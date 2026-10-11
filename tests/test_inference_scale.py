"""
tests/test_inference_scale.py
===================================================================
Regression test verifying that predicted 7-day future price trajectories
remain within realistic bounds (+/-15%) across major benchmark tickers,
preventing raw or un-normalized volatility z-score artifacts.
===================================================================
"""

import pytest
import numpy as np
from predict import predict_7days

TICKERS_TO_TEST = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ITC.NS"]


@pytest.mark.parametrize("ticker", TICKERS_TO_TEST)
def test_inference_scale_and_trajectory(ticker):
    """
    Verifies that for each benchmark ticker:
    1. The 7-day overall predicted price change is within +/-15%.
    2. The daily trajectory does not contain jagged jumps (> 10% single-day change).
    3. Prints the full 7-day trajectory for audit inspection.
    """
    res = predict_7days(ticker, use_cache=False)
    curr_price = res["current_price"]
    pred_prices = res["predicted_prices"]

    assert len(pred_prices) == 7, f"{ticker}: Expected 7 price points, got {len(pred_prices)}"
    assert curr_price > 0, f"{ticker}: Invalid current price {curr_price}"

    # Calculate 7-day move percentage
    overall_move_pct = (pred_prices[-1] - curr_price) / curr_price * 100.0

    print(f"\n[{ticker}] Current Price: INR {curr_price:.2f}")
    for d, p in enumerate(pred_prices, 1):
        d_chg = (p - curr_price) / curr_price * 100.0
        print(f"  Day {d}: INR {p:.2f} ({d_chg:+.2f}%)")
    print(f"  Total 7-Day Move: {overall_move_pct:+.2f}%")

    # Assert move is within +/-15%
    assert abs(overall_move_pct) <= 15.0, (
        f"{ticker} 7-day forecast {overall_move_pct:+.2f}% exceeds +/-15% realistic limit!"
    )

    # Assert no jagged day-over-day spikes (> 10% in a single day)
    prev = curr_price
    for d, p in enumerate(pred_prices, 1):
        day_jump_pct = abs((p - prev) / prev * 100.0)
        assert day_jump_pct <= 10.0, (
            f"{ticker} Day {d} jagged jump {day_jump_pct:.2f}% exceeds 10% smoothness limit!"
        )
        prev = p
