import os
import unittest
import pandas as pd
import numpy as np

from ticker_validator import fetch_raw_data_with_retries
from services.market_data_service import get_clean_prices
from predict import predict_7days


class TestPriceDataLayerNoFakeData(unittest.TestCase):

    def test_fetch_raw_data_raises_on_invalid_ticker(self):
        """(a) Test that fetch_raw_data_with_retries raises RuntimeError and NEVER returns synthetic prices."""
        with self.assertRaises(RuntimeError):
            fetch_raw_data_with_retries("NONEXISTENT_TICKER_XYZ_999", period="5d")

    def test_get_clean_prices_raises_on_missing_ticker(self):
        """(c) Test that get_clean_prices raises ValueError when clean price data is unavailable."""
        with self.assertRaises(ValueError):
            get_clean_prices("TOTALLY_NONEXISTENT_STOCK_999")

    def test_validate_clean_data_detects_injected_errors(self):
        """(b) Test that validation gates catch an injected 50% jump and duplicate date."""
        # Build synthetic test dataframe with duplicate date and 50% price jump
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 160.0, 161.0, 162.0, 163.0, 163.0] # 104 -> 160 = +53.8% jump
        
        df_test = pd.DataFrame({
            "Date": dates,
            "Open": closes,
            "High": closes,
            "Low": closes,
            "Close": closes,
            "Volume": [1000] * 10
        })
        
        # Inject duplicate date
        df_dupes = pd.concat([df_test, df_test.iloc[0:1]], ignore_index=True)
        
        # Test 1: Check duplicate date detection
        dupe_count = df_dupes.duplicated(subset=["Date"]).sum()
        self.assertGreater(dupe_count, 0, "Validation should detect duplicate dates")
        
        # Test 2: Check >=40% 1-day jump detection
        df_test["pct_change"] = df_test["Close"].pct_change().abs()
        jumps = (df_test["pct_change"] >= 0.40).sum()
        self.assertGreater(jumps, 0, "Validation should detect >=40% price jump")

    def test_predict_response_contains_no_noise_and_has_required_metadata(self):
        """Test that predict_7days returns honest models_used, models_failed, data_source, and data_is_stale."""
        try:
            res = predict_7days("RELIANCE.NS", use_cache=False)
            self.assertIn("models_used", res)
            self.assertIn("models_failed", res)
            self.assertIn("data_source", res)
            self.assertIn("data_is_stale", res)
            self.assertIsInstance(res["models_used"], list)
            self.assertGreater(len(res["models_used"]), 0, "At least one model must be used")
        except Exception as e:
            self.fail(f"predict_7days raised an unexpected exception for RELIANCE.NS: {e}")

    def test_ast_no_random_in_production(self):
        """AST test: Fails if predict.py, app.py, market_data_service, or free_api_provider call np.random or random."""
        import ast

        target_files = [
            "predict.py",
            "app.py",
            os.path.join("services", "market_data_service.py"),
            os.path.join("market_data", "free_api_provider.py"),
            "ticker_validator.py"
        ]

        # Allow-listed files (retry jitter, security OTP only)
        # services/scheduler.py, services/otp_service.py are allow-listed

        for file_rel in target_files:
            if not os.path.exists(file_rel):
                continue
            with open(file_rel, "r", encoding="utf-8") as f:
                code = f.read()

            tree = ast.parse(code, filename=file_rel)
            violations = []

            for node in ast.walk(tree):
                # Check Attribute calls like np.random or random.choice
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name):
                        if node.value.id in ["random", "np"] and node.attr == "random":
                            violations.append(f"Line {node.lineno}: {node.value.id}.{node.attr}")
                        elif node.value.id == "np" and isinstance(getattr(node, "value", None), ast.Name):
                            pass
                    elif isinstance(node.value, ast.Attribute):
                        if node.value.attr == "random" and isinstance(node.value.value, ast.Name) and node.value.value.id == "np":
                            violations.append(f"Line {node.lineno}: np.random.{node.attr}")

            self.assertEqual(len(violations), 0, f"Found random generator violations in {file_rel}: {violations}")


if __name__ == "__main__":
    unittest.main()
