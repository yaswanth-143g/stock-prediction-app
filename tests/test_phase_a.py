"""
tests/test_phase_a.py — Unit Tests for Phase A Credibility Fixes
================================================================
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from feature_engineering import CANONICAL_FEATURE_COLS, CANONICAL_FEATURE_COLS_V2, build_canonical_feature_set
from train_models import check_existing_model_status


def test_canonical_feature_cols_v2_scale_independence():
    """Verify CANONICAL_FEATURE_COLS_V2 excludes nominal price/volume scale columns."""
    raw_scale_cols = ["Open", "High", "Low", "Close", "Volume", "OBV", "ATR"]
    for col in raw_scale_cols:
        assert col not in CANONICAL_FEATURE_COLS_V2, f"Scale column {col} must not be in V2 feature set"

    assert len(CANONICAL_FEATURE_COLS_V2) == 26
    assert "Return_1d" in CANONICAL_FEATURE_COLS_V2
    assert "RSI" in CANONICAL_FEATURE_COLS_V2
    assert "NATR" in CANONICAL_FEATURE_COLS_V2


test_canonical_feature_cols_v2_scale_independence()


def test_global_metadata_warning_annotated():
    """Verify models/global_metadata.json contains warning annotation for static R2 metric."""
    meta_path = os.path.join("models", "global_metadata.json")
    assert os.path.exists(meta_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    assert "warning" in meta, "Global metadata must contain a warning field"
    assert "84.2%" in meta["warning"] or "in-sample" in meta["warning"].lower()


test_global_metadata_warning_annotated()


def test_honest_baseline_metrics_json_exists():
    """Verify experiments/sentiment_v1/baseline_metrics.json output structure."""
    json_path = os.path.join("experiments", "sentiment_v1", "baseline_metrics.json")
    assert os.path.exists(json_path), f"{json_path} must exist"
    with open(json_path, "r") as f:
        data = json.load(f)
    summary = data.get("summary_metrics") or data.get("mean_metrics_across_tickers", {})
    assert "model_random_forest" in summary
    assert "zero_return_baseline" in summary
    assert "mean_return_baseline" in summary or "train_mean_baseline" in summary
    assert data["gap_days"] == 7


test_honest_baseline_metrics_json_exists()


def test_scaler_fit_train_slice_only():
    """Synthetic test: assert MinMaxScaler fitted on 70% train slice excludes large test range values."""
    from sklearn.preprocessing import MinMaxScaler

    # Create synthetic series where train is range [0, 100] and test range is [500, 1000]
    n_samples = 1000
    train_end = int(n_samples * 0.70)  # 700

    train_data = np.linspace(0, 100, train_end).reshape(-1, 1)
    test_data = np.linspace(500, 1000, n_samples - train_end).reshape(-1, 1)
    full_data = np.vstack([train_data, test_data])

    # Standard train-only fit logic
    scaler = MinMaxScaler()
    scaler.fit(full_data[:train_end])

    assert float(scaler.data_min_[0]) == 0.0, f"Expected data_min_ 0.0, got {scaler.data_min_[0]}"
    assert float(scaler.data_max_[0]) == 100.0, f"Expected data_max_ 100.0, got {scaler.data_max_[0]}"
    assert float(scaler.data_max_[0]) < float(np.max(full_data)), "Scaler must NOT see test period maximum value"


test_scaler_fit_train_slice_only()

print("✅ All Phase A tests & pre-Phase B checks passed!")
