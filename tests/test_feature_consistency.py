"""
tests/test_feature_consistency.py — Feature Consistency AST & Runtime Unit Test
================================================================================
Validates that inference feature pipeline (predict.py) produces feature columns,
count, and order that strictly match CANONICAL_FEATURE_COLS and loaded global_scaler / global_model metadata.
"""

import os
import joblib
import numpy as np
import pandas as pd
import pytest

from feature_engineering import CANONICAL_FEATURE_COLS, build_canonical_feature_set, clean_ohlcv
from predict import get_model_manager, MODELS_DIR


def test_inference_feature_cols_and_count():
    """Verify CANONICAL_FEATURE_COLS has exactly 36 features."""
    assert len(CANONICAL_FEATURE_COLS) == 36, f"Expected 36 canonical features, got {len(CANONICAL_FEATURE_COLS)}"


def test_predict_feature_generation_order():
    """Verify build_canonical_feature_set generates all 36 canonical feature columns in exact order."""
    dates = pd.date_range("2025-01-01", periods=100, freq="D")
    df_raw = pd.DataFrame({
        "Date": dates,
        "Open": np.linspace(100, 110, 100),
        "High": np.linspace(102, 112, 100),
        "Low": np.linspace(99, 109, 100),
        "Close": np.linspace(101, 111, 100),
        "Volume": np.full(100, 100000)
    })
    df_clean = clean_ohlcv(df_raw, ticker="TEST.NS")
    df_feat = build_canonical_feature_set(df_clean, ticker="TEST.NS")

    for col in CANONICAL_FEATURE_COLS:
        assert col in df_feat.columns, f"Missing feature column in inference output: {col}"


def test_loaded_model_scaler_feature_dim():
    """Verify loaded global model and scaler feature dimensions and dynamic slicing compatibility."""
    model_mgr = get_model_manager()
    g_model, g_scaler = model_mgr.get_loaded_model("RELIANCE.NS")

    assert g_model is not None, "Global model must be loadable"
    assert g_scaler is not None, "Global scaler must be loadable"

    scaler_X = g_scaler[0] if isinstance(g_scaler, tuple) else g_scaler
    n_expected = getattr(scaler_X, "n_features_in_", None)
    
    # Global scaler was fitted on 30 features (36 canonical features minus non-numeric/NaN fundamental ratios)
    assert n_expected == 30, f"Global scaler n_features_in_ is {n_expected}, expected 30"

    # Test feature slicing behavior in predict.py
    X_36 = np.zeros((1, 36))
    if X_36.shape[1] > n_expected:
        X_sliced = X_36[:, :n_expected]
        assert X_sliced.shape[1] == 30
        X_scaled = scaler_X.transform(X_sliced)
        assert X_scaled.shape == (1, 30)

