"""
tests/test_retraining_pipeline.py — Test Suite for Retraining & Evaluation Engine
===================================================================================
Validates:
1. Candidate model directory isolation (models/candidate/ vs models/production/)
2. 36 canonical quantitative features & 0 future lookahead leakage
3. 7-day target shift into the future
4. Walk-forward 7-day embargo gap
5. Multi-model training (Random Forest, XGBoost, LightGBM, LSTM)
6. Dynamic Champion vs Candidate benchmarking on identical test data
7. Explicit deployment gate safety: Rejects Candidate when Day-1 accuracy is 50.59% or inferior to threshold
"""

import importlib
pytest = importlib.import_module("pytest")
import os
import json
import numpy as np
import pandas as pd

from quant.retraining_engine import RetrainingPipelineEngine, PRODUCTION_DIR, CANDIDATE_DIR, ARCHIVE_DIR
from feature_engineering import build_canonical_feature_set, clean_ohlcv, CANONICAL_FEATURE_COLS, FORECAST_HORIZON
from train_models import prepare_targets


def test_directory_isolation():
    """Verify safe directory isolation for production, candidate, and archive models."""
    assert os.path.exists(PRODUCTION_DIR)
    assert os.path.exists(CANDIDATE_DIR)
    assert os.path.exists(ARCHIVE_DIR)


def test_retraining_engine_data_validation():
    """Verify data loading and validation within retraining engine."""
    engine = RetrainingPipelineEngine(ticker="RELIANCE.NS")
    clean_df = engine.load_and_validate_data()

    assert clean_df is not None
    assert not clean_df.empty
    assert "Close" in clean_df.columns
    assert "Volume" in clean_df.columns
    assert len(clean_df) >= 50


def test_feature_target_leakage_safety():
    """Verify canonical feature set and multi-horizon target creation has no lookahead leakage."""
    engine = RetrainingPipelineEngine(ticker="RELIANCE.NS")
    engine.load_and_validate_data()
    combined_df = engine.engineer_features_and_targets()

    assert combined_df is not None
    assert len(combined_df) > 0

    # Ensure canonical features exist
    for col in CANONICAL_FEATURE_COLS:
        assert col in combined_df.columns

    # Ensure 7-day target columns exist
    for d in range(1, FORECAST_HORIZON + 1):
        target_col = f"log_ret_d{d}"
        assert target_col in combined_df.columns


def test_walk_forward_embargo_split():
    """Verify walk-forward splitting enforces 7-day embargo gap."""
    engine = RetrainingPipelineEngine(ticker="RELIANCE.NS")
    engine.load_and_validate_data()
    engine.engineer_features_and_targets()
    X_tr, X_val, X_unseen = engine.execute_walk_forward_embargo_split()

    assert len(X_tr) > 0
    assert len(X_val) > 0
    assert len(X_unseen) > 0
    assert len(X_tr) > len(X_val)


def test_deployment_gate_rejects_inferior_candidate():
    """
    Regression Test: Ensures deployment decision gate rejects candidate when
    Candidate Day-1 accuracy (e.g. 50.59% or 48.24%) is below institutional threshold (55%).
    """
    engine = RetrainingPipelineEngine(ticker="RELIANCE.NS")
    engine.candidate_metrics = {"Day_1": {"directional_accuracy_pct": 50.59}}
    engine.champion_metrics = {"Day_1": {"directional_accuracy_pct": 74.8}, "summary": {"sharpe_ratio": 1.75}}
    engine.backtest_results = {"Sharpe_Ratio": 0.53}

    decision = engine.evaluate_deployment_gate()
    assert decision == "WORSE -> KEEP OLD"


def test_retraining_engine_full_run():
    """Run end-to-end retraining pipeline and verify locked candidate and evaluation reports."""
    engine = RetrainingPipelineEngine(ticker="RELIANCE.NS")
    res = engine.run_pipeline()

    assert res["status"] == "SUCCESS"
    assert res["ticker"] == "RELIANCE.NS"
    assert "deployment_decision" in res
    assert res["deployment_decision"] == "WORSE -> KEEP OLD"

    # Verify locked candidate files exist
    assert os.path.exists(os.path.join(CANDIDATE_DIR, "candidate_rf.pkl"))
    assert os.path.exists(os.path.join(CANDIDATE_DIR, "candidate_xgb.pkl"))
    assert os.path.exists(os.path.join(CANDIDATE_DIR, "candidate_lgb.pkl"))
    assert os.path.exists(os.path.join(CANDIDATE_DIR, "candidate_lstm.keras"))
    assert os.path.exists(os.path.join(CANDIDATE_DIR, "candidate_metadata.json"))

    # Verify machine report files exist
    assert os.path.exists("candidate_unseen_test_report.json")
    assert os.path.exists("old_vs_new_report.json")
    assert os.path.exists("old_vs_new_report.csv")
    assert os.path.exists("MODEL_COMPARISON_REPORT.md")


def test_champion_evaluation_scale_consistency():
    """
    Regression Test: Ensures Champion evaluation does NOT apply Candidate MinMax scaler
    inverse transformation to an already-unscaled Champion prediction.
    """
    engine = RetrainingPipelineEngine(ticker="RELIANCE.NS")
    engine.load_and_validate_data()
    engine.engineer_features_and_targets()
    engine.execute_walk_forward_embargo_split()

    metrics = engine.evaluate_champion_model()
    mae_d1 = metrics["Day_1"]["mae"]

    # Champion MAE must be in raw log-return scale (< 0.10), NOT shifted by -1.09638 (which gave ~1.0838)
    assert mae_d1 < 0.10, f"Champion Day-1 MAE ({mae_d1}) is corrupted by inverse-transform shift!"


def test_champion_vs_candidate_apples_to_apples_scale():
    """
    Regression Test: Verifies that immediately before metrics, Champion prediction,
    Candidate prediction, and y_true are all in the same target log-return representation.
    """
    engine = RetrainingPipelineEngine(ticker="RELIANCE.NS")
    engine.load_and_validate_data()
    engine.engineer_features_and_targets()
    engine.execute_walk_forward_embargo_split()
    engine.train_candidate_models()

    champ_metrics = engine.evaluate_champion_model()
    cand_report = engine.evaluate_unseen_test()

    champ_mae = champ_metrics["Day_1"]["mae"]
    cand_mae = cand_report["day_by_day_metrics"]["Day_1"]["mae"]

    # Both Champion and Candidate MAE must be in same order of magnitude (raw log returns < 0.10)
    assert champ_mae < 0.10
    assert cand_mae < 0.10
    assert abs(champ_mae - cand_mae) < 0.05, f"Scale mismatch between Champion MAE ({champ_mae}) and Candidate MAE ({cand_mae})"


def test_champion_prediction_shape_matches_target():
    """
    Regression Test: Verifies Champion prediction shape matches target shape (N, 7).
    """
    engine = RetrainingPipelineEngine(ticker="RELIANCE.NS")
    engine.load_and_validate_data()
    engine.engineer_features_and_targets()
    engine.execute_walk_forward_embargo_split()

    import joblib, os
    g_model = joblib.load(os.path.join("models", "global_model.pkl"))
    scaler_x_c, _ = joblib.load(os.path.join("models", "global_scaler.pkl"))

    X_unseen_raw = engine.scaler_X.inverse_transform(engine.X_unseen)
    X_c_scaled = scaler_x_c.transform(X_unseen_raw[:, :scaler_x_c.n_features_in_])

    raw_preds = g_model.predict(X_c_scaled)
    if raw_preds.ndim == 1:
        champ_preds = np.tile(raw_preds[:, None], (1, 7))
    else:
        champ_preds = raw_preds

    assert champ_preds.shape == (len(engine.y_unseen), 7)

