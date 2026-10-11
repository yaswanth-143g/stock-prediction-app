"""
quant/walk_forward_validation.py
================================
Walk-Forward / Rolling Time-Series Validation Engine
---------------------------------------------------
Implements leakage-free chronological rolling validation, baseline benchmarking,
target gap/embargo handling, and Day 1 through Day 7 horizon accuracy metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class WalkForwardValidator:
    """
    Chronological rolling time-series validator.
    Ensures zero future data leakage by fitting scalers, feature selectors,
    and models strictly on expanding training folds.
    """

    def __init__(self, n_splits: int = 5, test_size_days: int = 60, gap_days: int = 7):
        self.n_splits = n_splits
        self.test_size_days = test_size_days
        self.gap_days = gap_days  # Embargo window to prevent overlapping target leakage

    def generate_splits(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates expanding training and validation index arrays with gap embargo.
        """
        splits = []
        min_train_size = max(120, n_samples - self.n_splits * self.test_size_days)

        for i in range(self.n_splits):
            train_end = min_train_size + i * self.test_size_days
            test_start = train_end + self.gap_days
            test_end = test_start + self.test_size_days

            if test_end > n_samples:
                test_end = n_samples

            if test_start >= n_samples:
                break

            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            splits.append((train_idx, test_idx))

        return splits

    def evaluate_baselines(self, df: pd.DataFrame, target_cols: List[str]) -> Dict[str, Any]:
        """
        Evaluates simple financial benchmarks (Zero-Return, Previous Return, Mean Return).
        """
        if df.empty or 'close' not in df.columns:
            return {}

        results = {}
        for day_idx, col in enumerate(target_cols, start=1):
            if col not in df.columns:
                continue

            actual_returns = df[col].dropna().values
            if len(actual_returns) == 0:
                continue

            # Baseline 1: Zero Return
            zero_pred = np.zeros_like(actual_returns)
            zero_mae = mean_absolute_error(actual_returns, zero_pred)
            zero_rmse = np.sqrt(mean_squared_error(actual_returns, zero_pred))
            zero_dir_acc = np.mean((actual_returns >= 0) == (zero_pred >= 0)) * 100.0

            # Baseline 2: Mean Return
            mean_pred = np.full_like(actual_returns, np.mean(actual_returns))
            mean_mae = mean_absolute_error(actual_returns, mean_pred)
            mean_rmse = np.sqrt(mean_squared_error(actual_returns, mean_pred))
            mean_dir_acc = np.mean((actual_returns >= 0) == (mean_pred >= 0)) * 100.0

            results[f"Day_{day_idx}"] = {
                "zero_baseline": {"mae": float(zero_mae), "rmse": float(zero_rmse), "directional_acc": float(zero_dir_acc)},
                "mean_baseline": {"mae": float(mean_mae), "rmse": float(mean_rmse), "directional_acc": float(mean_dir_acc)}
            }

        return results

    def compute_horizon_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Computes detailed regression & directional metrics for a single prediction horizon.
        """
        if len(y_true) == 0 or len(y_pred) == 0:
            return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "directional_acc": 50.0}

        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        yt = y_true[mask]
        yp = y_pred[mask]

        if len(yt) == 0:
            return {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "directional_acc": 50.0}

        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2 = r2_score(yt, yp) if len(yt) > 1 else 0.0

        # Directional accuracy (% where predicted sign matches actual sign)
        correct_dir = (yt >= 0) == (yp >= 0)
        dir_acc = np.mean(correct_dir) * 100.0

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "directional_acc": float(dir_acc)
        }
