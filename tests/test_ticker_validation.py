"""
tests/test_ticker_validation.py — Verification Test Suite
============================================================
Tests automatic failed stock ticker detection and replacement logic,
validation reports, JSON mapping files, and prediction pipeline integration.
"""

import os
import sys
import json
import pandas as pd

# Add project root to sys.path if running directly from tests/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ticker_validator import validate_all_tickers, validate_single_ticker
from predict import predict_stock

TEST_STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "ZOMATO.NS",
    "PAYTM.NS",
    "INVALID123.NS",
    "NONEXISTENT_TICKER.NS",
    "ABC999.NS",
    "FAKETICKER.NS",
    "TESTFAIL.NS"
]


import unittest

class TestTickerValidation(unittest.TestCase):
    def test_validation_system(self):
        print("\n" + "=" * 65)
        print("      RUNNING AUTOMATED TICKER VALIDATION TEST SUITE")
        print("=" * 65)

        # 1. Run full batch validation
        validated_tickers, mapping, failed_list = validate_all_tickers(TEST_STOCKS, period="10y", save_reports=True)

        print("\n[TEST 1] Checking saved artifacts...")
        mapping_path = os.path.join("data", "ticker_validation", "ticker_mapping.json")
        report_path = os.path.join("data", "ticker_validation", "ticker_validation_report.csv")

        self.assertTrue(os.path.exists(mapping_path), f"Missing {mapping_path}")
        self.assertTrue(os.path.exists(report_path), f"Missing {report_path}")
        print("  [OK] Artifact files verified!")

        # 2. Check ZOMATO.NS replacement
        print("\n[TEST 2] Verifying ZOMATO.NS replacement to ETERNAL.NS...")
        zomato_info = mapping.get("ZOMATO.NS", {})
        self.assertEqual(zomato_info.get("status"), "REPLACED", f"ZOMATO.NS status is {zomato_info.get('status')}")
        self.assertEqual(zomato_info.get("final_ticker"), "ETERNAL.NS", f"ZOMATO.NS replacement is {zomato_info.get('final_ticker')}")
        print("  [OK] ZOMATO.NS successfully replaced with ETERNAL.NS!")

        # 3. Check invalid tickers marked as FAILED
        print("\n[TEST 3] Verifying invalid tickers marked as FAILED...")
        invalid_sample = ["INVALID123.NS", "NONEXISTENT_TICKER.NS", "ABC999.NS", "FAKETICKER.NS", "TESTFAIL.NS"]
        for inv in invalid_sample:
            info = mapping.get(inv, {})
            self.assertEqual(info.get("status"), "FAILED", f"{inv} status is {info.get('status')}")
        print("  [OK] All 5 invalid tickers correctly marked as FAILED!")

        # 4. Predict Integration Test (ZOMATO.NS -> ETERNAL.NS)
        print("\n[TEST 4] Testing predict_stock('ZOMATO.NS')...")
        res_zomato = predict_stock("ZOMATO.NS")
        self.assertTrue(res_zomato["is_replaced"], "ZOMATO.NS was not marked as replaced in predict_stock")
        self.assertEqual(res_zomato["final_ticker"], "ETERNAL.NS", f"Expected ETERNAL.NS, got {res_zomato['final_ticker']}")
        self.assertEqual(res_zomato["status"], "success", f"Prediction status: {res_zomato['status']}")
        print(f"  [OK] predict_stock('ZOMATO.NS') resolved to {res_zomato['final_ticker']} and returned predictions successfully!")

        # 5. Predict Integration Test (Invalid Ticker Error Handling)
        print("\n[TEST 5] Testing predict_stock('INVALID123.NS') error handling...")
        with self.assertRaises(ValueError):
            predict_stock("INVALID123.NS")
        print("  [OK] predict_stock('INVALID123.NS') correctly raised ValueError!")

        print("\n" + "=" * 65)
        print("          ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
        print("=" * 65 + "\n")


if __name__ == "__main__":
    unittest.main()
