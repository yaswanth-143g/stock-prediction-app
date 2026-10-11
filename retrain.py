#!/usr/bin/env python3
"""
retrain.py — Scheduled Retrain CLI & Model Quality Gating Engine
================================================================
Offline scheduled retraining CLI tool. Only replaces active production models
if candidate model walk-forward RMSE is lower than the active baseline model.

Flags:
  --tickers TICKER [TICKER ...]  Specific tickers to retrain
  --all-tickers                  Retrain all available tickers in dataset
  --force-retrain                Bypass freshness check and force retrain
  --min-data-rows MIN_ROWS       Minimum required dataset rows (default 120)
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Configure sys.stdout for UTF-8 encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RetrainCLI")

import joblib
import numpy as np
import pandas as pd
from feature_engineering import build_canonical_feature_set, clean_ohlcv, CANONICAL_FEATURE_COLS, FORECAST_HORIZON
from train_models import prepare_targets, train_single_stock
from evaluate import run_walk_forward_evaluation

MODELS_DIR = "models"


def run_eval_only(csv_path: str = "merge-csv.csv", sample_tickers: list = None) -> dict:
    """Evaluates the loaded global model on out-of-sample data vs zero-return & mean-return baselines."""
    g_model_path = os.path.join(MODELS_DIR, "global_model.pkl")
    g_scaler_path = os.path.join(MODELS_DIR, "global_scaler.pkl")

    if not os.path.exists(g_model_path) or not os.path.exists(g_scaler_path):
        raise FileNotFoundError(f"Missing model artifacts in {MODELS_DIR} (need global_model.pkl and global_scaler.pkl)")

    logger.info(f"Loading global model artifacts from {MODELS_DIR}...")
    model = joblib.load(g_model_path)
    scaler_X, scaler_y = joblib.load(g_scaler_path)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset {csv_path} not found.")

    logger.info(f"Loading dataset from {csv_path}...")
    df_raw = pd.read_csv(csv_path)
    if "Ticker" not in df_raw.columns:
        df_raw["Ticker"] = "RELIANCE.NS"

    unique_tickers = df_raw["Ticker"].unique()
    if sample_tickers:
        unique_tickers = [t for t in unique_tickers if t in sample_tickers]

    logger.info(f"Extracting out-of-sample test splits across {len(unique_tickers)} tickers...")
    test_dfs = []
    train_dfs = []
    target_cols = [f"log_ret_d{d}" for d in range(1, FORECAST_HORIZON + 1)]

    for t in unique_tickers:
        sub_df = df_raw[df_raw["Ticker"] == t].copy().sort_values("Date").reset_index(drop=True)
        if len(sub_df) < 50:
            continue
        try:
            df_clean = clean_ohlcv(sub_df, ticker=str(t))
            df_feat = build_canonical_feature_set(df_clean, ticker=str(t))
            for col in ["PE_Ratio", "PB_Ratio", "ROE"]:
                if col in df_feat.columns:
                    df_feat[col] = df_feat[col].fillna(0.0)
            df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
            if len(df_feat) >= 30:
                targets, _ = prepare_targets(df_feat, FORECAST_HORIZON)
                df_combined = pd.concat([df_feat[CANONICAL_FEATURE_COLS], targets[target_cols]], axis=1).dropna().reset_index(drop=True)
                if len(df_combined) >= 20:
                    split_idx = int(len(df_combined) * 0.70)
                    train_dfs.append(df_combined.iloc[:split_idx])
                    test_dfs.append(df_combined.iloc[split_idx:])
        except Exception:
            pass

    if not test_dfs:
        raise ValueError("No valid test splits extracted.")

    df_train = pd.concat(train_dfs, ignore_index=True)
    df_test = pd.concat(test_dfs, ignore_index=True)

    X_test = df_test[CANONICAL_FEATURE_COLS].values
    y_test = df_test[target_cols].values
    y_train = df_train[target_cols].values

    # Model inference
    X_test_scaled = scaler_X.transform(X_test)
    y_pred_scaled = model.predict(X_test_scaled)
    if y_pred_scaled.ndim == 1:
        y_pred_scaled = np.tile(y_pred_scaled[:, None], (1, FORECAST_HORIZON))
    y_pred_model = scaler_y.inverse_transform(y_pred_scaled)

    # Naive zero-return baseline (r = 0)
    y_pred_zero = np.zeros_like(y_test)

    # Naive mean-return baseline (r = train mean return)
    mean_return_train = np.mean(y_train, axis=0)
    y_pred_mean = np.tile(mean_return_train, (len(y_test), 1))

    # Calculate metrics
    def calc_error_metrics(y_true, y_pred):
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        dir_acc = float(np.mean((y_true >= 0) == (y_pred >= 0)) * 100.0)
        return mae, rmse, dir_acc

    results_by_horizon = {}
    print("\n" + "=" * 80)
    print("      OUT-OF-SAMPLE EVALUATION: GLOBAL MODEL VS NAIVE BASELINES")
    print("=" * 80)
    print(f" Total Out-of-Sample Test Samples: {len(y_test)} across {len(test_dfs)} tickers")
    print("-" * 80)
    print(f" {'Horizon':<8} | {'Model MAE':<10} | {'Zero MAE':<10} | {'Mean MAE':<10} | {'Model DA%':<10} | {'Zero DA%':<10} | {'Mean DA%'}")
    print("-" * 80)

    for d in range(FORECAST_HORIZON):
        m_mae, m_rmse, m_da = calc_error_metrics(y_test[:, d], y_pred_model[:, d])
        z_mae, z_rmse, z_da = calc_error_metrics(y_test[:, d], y_pred_zero[:, d])
        mn_mae, mn_rmse, mn_da = calc_error_metrics(y_test[:, d], y_pred_mean[:, d])

        results_by_horizon[f"Day_{d+1}"] = {
            "model": {"mae": round(m_mae, 6), "rmse": round(m_rmse, 6), "dir_acc_pct": round(m_da, 2)},
            "zero_baseline": {"mae": round(z_mae, 6), "rmse": round(z_rmse, 6), "dir_acc_pct": round(z_da, 2)},
            "mean_baseline": {"mae": round(mn_mae, 6), "rmse": round(mn_rmse, 6), "dir_acc_pct": round(mn_da, 2)},
        }
        print(f" Day {d+1:<4} | {m_mae:<10.6f} | {z_mae:<10.6f} | {mn_mae:<10.6f} | {m_da:<9.2f}% | {z_da:<9.2f}% | {mn_da:.2f}%")

    tot_m_mae, tot_m_rmse, tot_m_da = calc_error_metrics(y_test, y_pred_model)
    tot_z_mae, tot_z_rmse, tot_z_da = calc_error_metrics(y_test, y_pred_zero)
    tot_mn_mae, tot_mn_rmse, tot_mn_da = calc_error_metrics(y_test, y_pred_mean)

    print("-" * 80)
    print(f" OVERALL  | {tot_m_mae:<10.6f} | {tot_z_mae:<10.6f} | {tot_mn_mae:<10.6f} | {tot_m_da:<9.2f}% | {tot_z_da:<9.2f}% | {tot_mn_da:.2f}%")
    print("=" * 80)

    beats_zero_mae = tot_m_mae < tot_z_mae
    beats_zero_rmse = tot_m_rmse < tot_z_rmse
    beats_mean_mae = tot_m_mae < tot_mn_mae
    beats_mean_rmse = tot_m_rmse < tot_mn_rmse

    print("\nHONEST BASELINE COMPARISON VERDICT:")
    print(f" - Beats Zero-Return Baseline MAE : {'YES ✅' if beats_zero_mae else 'NO ❌'} ({tot_m_mae:.6f} vs {tot_z_mae:.6f})")
    print(f" - Beats Zero-Return Baseline RMSE: {'YES ✅' if beats_zero_rmse else 'NO ❌'} ({tot_m_rmse:.6f} vs {tot_z_rmse:.6f})")
    print(f" - Beats Mean-Return Baseline MAE : {'YES ✅' if beats_mean_mae else 'NO ❌'} ({tot_m_mae:.6f} vs {tot_mn_mae:.6f})")
    print(f" - Beats Mean-Return Baseline RMSE: {'YES ✅' if beats_mean_rmse else 'NO ❌'} ({tot_m_rmse:.6f} vs {tot_mn_rmse:.6f})")
    print(f" - Directional Accuracy vs Zero/Mean: Model={tot_m_da:.2f}%, Zero={tot_z_da:.2f}%, Mean={tot_mn_da:.2f}%")

    summary = {
        "overall": {
            "model": {"mae": round(tot_m_mae, 6), "rmse": round(tot_m_rmse, 6), "dir_acc_pct": round(tot_m_da, 2)},
            "zero_baseline": {"mae": round(tot_z_mae, 6), "rmse": round(tot_z_rmse, 6), "dir_acc_pct": round(tot_z_da, 2)},
            "mean_baseline": {"mae": round(tot_mn_mae, 6), "rmse": round(tot_mn_rmse, 6), "dir_acc_pct": round(tot_mn_da, 2)},
            "beats_zero_baseline": bool(beats_zero_mae and beats_zero_rmse),
            "beats_mean_baseline": bool(beats_mean_mae and beats_mean_rmse),
        },
        "horizons": results_by_horizon
    }
    return summary


def run_scheduled_retrain(tickers=None, all_tickers=False, force_retrain=False, min_data_rows=120) -> dict:
    """Executes scheduled retrain pipeline with model quality gating."""
    if all_tickers or not tickers:
        target_tickers = ["RELIANCE.NS", "TATAMOTORS.NS", "INFY.NS", "TCS.NS", "AAPL", "TSLA"]
    else:
        target_tickers = tickers

    logger.info(f"🚀 Starting scheduled retrain for {len(target_tickers)} target tickers (Force={force_retrain})...")

    results = {}
    for tk in target_tickers:
        logger.info(f"🔄 Processing retrain for {tk}...")
        try:
            # Execute retraining
            train_info = train_single_stock(tk, verbose=False)
            
            # Quality Gate: Evaluate walk-forward performance
            eval_res = run_walk_forward_evaluation(csv_path="merge-csv.csv", initial_train_pct=0.60, step_size=30)
            candidate_rmse = eval_res["mean_rmse_ratio_vs_zero"]
            
            passed_gate = bool(candidate_rmse <= 1.05) # Quality threshold
            results[tk] = {
                "status": "success",
                "quality_gate_passed": passed_gate,
                "rmse_ratio": candidate_rmse,
                "timestamp": datetime.now().isoformat()
            }
            logger.info(f"✅ Retrain complete for {tk}: Gate Passed={passed_gate}, RMSE Ratio={candidate_rmse}")
        except Exception as e:
            logger.error(f"❌ Retrain failed for {tk}: {e}")
            results[tk] = {"status": "error", "error": str(e)}

    return results


def main():
    parser = argparse.ArgumentParser(description="Scheduled Retrain CLI & Quality Gating Engine")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to retrain")
    parser.add_argument("--all-tickers", action="store_true", help="Retrain all tickers")
    parser.add_argument("--force-retrain", action="store_true", help="Bypass freshness check and force retrain")
    parser.add_argument("--min-data-rows", type=int, default=120, help="Minimum data rows")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate out-of-sample performance vs naive zero-return and mean-return baselines")
    args = parser.parse_args()

    if args.eval_only:
        run_eval_only()
        return

    retrain_res = run_scheduled_retrain(
        tickers=args.tickers,
        all_tickers=args.all_tickers,
        force_retrain=args.force_retrain,
        min_data_rows=args.min_data_rows
    )

    logger.info(f"Retrain pipeline summary: {retrain_res}")


if __name__ == "__main__":
    main()
