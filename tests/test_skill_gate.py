import os
import json
import pytest
from predict import predict_7days

def test_skill_gate_enforces_hold_when_flag_false():
    """Verify that when beats_zero_baseline is False, predict_7days always outputs HOLD with disclaimer."""
    # Run prediction for TCS.NS (uses global model)
    res = predict_7days("TCS.NS", use_cache=True)
    
    assert res["status"] == "success"
    assert res["beats_zero_baseline"] is False
    assert res["signal"] == "HOLD 🟡"
    assert res["signal_disclaimer"] is not None
    assert "no demonstrated edge" in res["signal_disclaimer"]

def test_skill_gate_allows_signal_when_flag_true(tmp_path, monkeypatch):
    """Verify that if beats_zero_baseline is True, normal directional signals can be emitted."""
    meta_path = os.path.join("models", "global_metadata.json")
    
    with open(meta_path, "r") as f:
        orig_meta = json.load(f)
        
    try:
        # Temporarily flip flag to True
        modified_meta = orig_meta.copy()
        modified_meta["beats_zero_baseline"] = True
        with open(meta_path, "w") as f:
            json.dump(modified_meta, f, indent=4)
            
        res = predict_7days("TCS.NS", use_cache=False)
        assert res["status"] == "success"
        assert res["beats_zero_baseline"] is True
        assert res["signal_disclaimer"] is None
        # Signal is allowed to reflect total_return_pct (e.g. BUY, SELL, or HOLD based on return threshold)
        assert res["signal"] in ["BUY 🟢", "SELL 🔴", "HOLD 🟡"]
    finally:
        # Restore original metadata strictly
        with open(meta_path, "w") as f:
            json.dump(orig_meta, f, indent=4)
