"""
tests/test_stock_comparison.py — Automated Unit & Integration Test Suite
========================================================================
Validates the Stock Comparison Service, financial metrics calculation engines,
trading-date alignment, and isolated REST API endpoints in app.py.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add project root directory to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from services.stock_comparison_service import (
    validate_comparison_inputs,
    compute_stock_metrics,
    align_stock_histories,
    compare_stocks,
    compare_predictions
)
from app import app


class TestStockComparisonService(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_input_validation(self):
        """Test stock ticker list validation logic."""
        # Valid cases
        is_valid, err, valid_symbols, warning = validate_comparison_inputs(["RELIANCE.NS", "TCS.NS"])
        self.assertTrue(is_valid)
        self.assertEqual(valid_symbols, ["RELIANCE.NS", "TCS.NS"])

        # Case sensitivity & cleanup
        is_valid, err, valid_symbols, warning = validate_comparison_inputs([" reliance.ns ", "tcs.ns "])
        self.assertTrue(is_valid)
        self.assertEqual(valid_symbols, ["RELIANCE.NS", "TCS.NS"])

        # Deduplication
        is_valid, err, valid_symbols, warning = validate_comparison_inputs(["RELIANCE.NS", "RELIANCE.NS", "TCS.NS"])
        self.assertTrue(is_valid)
        self.assertEqual(valid_symbols, ["RELIANCE.NS", "TCS.NS"])

        # Too few stocks (< 2)
        is_valid, err, valid_symbols, warning = validate_comparison_inputs(["RELIANCE.NS"])
        self.assertFalse(is_valid)
        self.assertIn("at least 2", err)

        # Too many stocks (> 5 truncated with warning)
        is_valid, err, valid_symbols, warning = validate_comparison_inputs(["S1.NS", "S2.NS", "S3.NS", "S4.NS", "S5.NS", "S6.NS"])
        self.assertFalse(is_valid)  # Returns False when > 5 with clean_syms up to 5
        self.assertEqual(len(valid_symbols), 5)

    def test_financial_metrics_calculation(self):
        """Test accuracy of financial metrics: Return, Volatility, Drawdown."""
        # Simulated price series starting at 100 ending at 120 (20% return)
        prices = pd.Series([100.0, 105.0, 110.0, 102.0, 115.0, 120.0])
        dates = pd.Series(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05", "2025-01-06"])
        metrics = compute_stock_metrics(prices, dates)

        self.assertEqual(metrics['starting_price'], 100.0)
        self.assertEqual(metrics['latest_price'], 120.0)
        self.assertEqual(metrics['absolute_change'], 20.0)
        self.assertEqual(metrics['percentage_return'], 20.0)
        self.assertGreater(metrics['annualized_volatility'], 0)
        self.assertLessEqual(metrics['max_drawdown'], 0)  # Peak 110 to Trough 102 decline
        self.assertEqual(metrics['normalized_series'][0], 100.0)

    def test_alignment_logic(self):
        """Test inner join trading-date alignment."""
        dates1 = pd.date_range(start="2025-01-01", periods=5, freq="D").strftime("%Y-%m-%d")
        dates2 = pd.date_range(start="2025-01-03", periods=5, freq="D").strftime("%Y-%m-%d")

        dict_df = {
            "S1": pd.DataFrame({"Date": dates1, "Close": [10, 11, 12, 13, 14]}),
            "S2": pd.DataFrame({"Date": dates2, "Close": [20, 21, 22, 23, 24]})
        }

        aligned_df, meta = align_stock_histories(dict_df)

        # Overlapping dates: Jan 3, Jan 4, Jan 5 (3 days)
        self.assertEqual(len(aligned_df), 3)
        self.assertIn("S1", aligned_df.columns)
        self.assertIn("S2", aligned_df.columns)

    def test_compare_stocks_main(self):
        """Test full compare_stocks pipeline."""
        res = compare_stocks(["RELIANCE.NS", "TCS.NS"], period="1m")
        self.assertEqual(res["status"], "success")
        self.assertIn("metrics_table", res)
        self.assertIn("charts_data", res)

    def test_api_stock_comparison_symbols(self):
        """Test GET /api/stock-comparison/symbols route."""
        response = self.client.get('/api/stock-comparison/symbols')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIsInstance(data['symbols'], list)
        self.assertGreater(len(data['symbols']), 0)

    def test_api_stock_comparison_history(self):
        """Test GET /api/stock-comparison/history endpoint."""
        response = self.client.get('/api/stock-comparison/history?symbols=RELIANCE.NS&symbols=TCS.NS&period=1m')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('charts_data', data)
        self.assertIn('metrics_table', data)
        self.assertIn('metadata', data)

    def test_api_stock_comparison_predictions(self):
        """Test POST /api/stock-comparison/predictions endpoint."""
        response = self.client.post('/api/stock-comparison/predictions', json={
            'symbols': ['RELIANCE.NS', 'TCS.NS']
        })
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('predictions', data)
        self.assertIsInstance(data['predictions'], list)

    def test_stock_comparison_view_route(self):
        """Test GET /stock-comparison frontend view route."""
        response = self.client.get('/stock-comparison')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Multi-Stock Performance', response.data)


if __name__ == '__main__':
    unittest.main()

