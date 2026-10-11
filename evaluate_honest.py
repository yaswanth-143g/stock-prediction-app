#!/usr/bin/env python3
"""
evaluate_honest.py — Honest Out-of-Sample Walk-Forward Evaluator
================================================================
Runs quantitative walk-forward validation (gap_days=7) using scale-independent
features (CANONICAL_FEATURE_COLS_V2) and evaluates out-of-sample MAE, RMSE,
R2, and Directional Accuracy on RETURNS across tickers.

Outputs results to: experiments/sentiment_v1/baseline_metrics.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure sys.stdout for UTF-8 encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from feature_engineering import clean_ohlcv, build_canonical_feature_set, CANONICAL_FEATURE_COLS_V2, FORECAST_HORIZON
from train_models import prepare_targets
from quant.walk_forward_validation import WalkForwardValidator

EXPERIMENT_DIR = os.path.join("experiments", "sentiment_v1")
OUTPUT_JSON = os.path.join(EXPERIMENT_DIR, "baseline_metrics.json")
STOCK_DATA_DIR = "Stock_Data"


def evaluate_stock_walk_forward(df_clean: pd.DataFrame, ticker: str, validator: WalkForwardValidator) -> dict:
    """Evaluates a single stock using chronological walk-forward splits with gap embargo."""
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor

    df_clean = clean_ohlcv(df_clean, ticker=ticker)
    df_feat = build_canonical_feature_set(df_clean, ticker=ticker)

    targets, _ = prepare_targets(df_feat, FORECAST_HORIZON, normalize_volatility=False)
    target_cols = [f"log_ret_d{d}" for d in range(1, FORECAST_HORIZON + 1)]

    feature_cols = [col for col in CANONICAL_FEATURE_COLS_V2 if col in df_feat.columns]
    df_combined = pd.concat([df_feat[feature_cols], targets[target_cols]], axis=1)
    df_combined = df_combined.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    if len(df_combined) < 100:
        return None

    X_all = df_combined[feature_cols].values
    y_all = df_combined[target_cols].values

    splits = validator.generate_splits(len(df_combined))
    if not splits:
        return None

    fold_metrics = []
    all_y_true = []
    all_y_pred_model = []
    all_y_pred_zero = []
    all_y_pred_mean = []

    fold_model_maes, fold_model_rmses, fold_model_r2s, fold_model_dir_accs = [], [], [], []
    fold_zero_maes, fold_zero_rmses, fold_zero_r2s, fold_zero_dir_accs = [], [], [], []
    fold_mean_maes, fold_mean_rmses, fold_mean_r2s, fold_mean_dir_accs = [], [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(splits, start=1):
        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)

        X_test_scaled = scaler_X.transform(X_test)

        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1))
        model.fit(X_train_scaled, y_train_scaled)

        y_pred_scaled = model.predict(X_test_scaled)
        y_pred_m = scaler_y.inverse_transform(y_pred_scaled)

        # Baseline (a): Zero Return
        y_pred_z = np.zeros_like(y_test)

        # Baseline (b): Training Mean Return
        mean_ret_train = np.mean(y_train, axis=0)
        y_pred_mn = np.tile(mean_ret_train, (len(y_test), 1))

        all_y_true.append(y_test)
        all_y_pred_model.append(y_pred_m)
        all_y_pred_zero.append(y_pred_z)
        all_y_pred_mean.append(y_pred_mn)

        m_model = validator.compute_horizon_metrics(y_test.flatten(), y_pred_m.flatten())
        m_zero = validator.compute_horizon_metrics(y_test.flatten(), y_pred_z.flatten())
        m_mean = validator.compute_horizon_metrics(y_test.flatten(), y_pred_mn.flatten())

        fold_model_maes.append(m_model["mae"])
        fold_model_rmses.append(m_model["rmse"])
        fold_model_r2s.append(m_model["r2"])
        fold_model_dir_accs.append(m_model["directional_acc"])

        fold_zero_maes.append(m_zero["mae"])
        fold_zero_rmses.append(m_zero["rmse"])
        fold_zero_r2s.append(m_zero["r2"])
        fold_zero_dir_accs.append(m_zero["directional_acc"])

        fold_mean_maes.append(m_mean["mae"])
        fold_mean_rmses.append(m_mean["rmse"])
        fold_mean_r2s.append(m_mean["r2"])
        fold_mean_dir_accs.append(m_mean["directional_acc"])

        fold_metrics.append({
            "fold": fold_idx,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "model_metrics": m_model,
            "zero_baseline_metrics": m_zero,
            "mean_baseline_metrics": m_mean
        })

    y_true_concat = np.vstack(all_y_true).flatten()
    y_pred_m_concat = np.vstack(all_y_pred_model).flatten()
    y_pred_z_concat = np.vstack(all_y_pred_zero).flatten()
    y_pred_mn_concat = np.vstack(all_y_pred_mean).flatten()

    overall_model = validator.compute_horizon_metrics(y_true_concat, y_pred_m_concat)
    overall_zero = validator.compute_horizon_metrics(y_true_concat, y_pred_z_concat)
    overall_mean = validator.compute_horizon_metrics(y_true_concat, y_pred_mn_concat)

    return {
        "ticker": ticker,
        "total_samples": len(df_combined),
        "feature_set": "CANONICAL_FEATURE_COLS_V2",
        "num_folds": len(fold_metrics),
        "overall_metrics": {
            "model_rf": overall_model,
            "zero_baseline": overall_zero,
            "mean_baseline": overall_mean
        },
        "std_across_folds": {
            "model_rf": {
                "mae_std": float(np.std(fold_model_maes)),
                "rmse_std": float(np.std(fold_model_rmses)),
                "r2_std": float(np.std(fold_model_r2s)),
                "directional_acc_std": float(np.std(fold_model_dir_accs))
            },
            "zero_baseline": {
                "mae_std": float(np.std(fold_zero_maes)),
                "rmse_std": float(np.std(fold_zero_rmses)),
                "r2_std": float(np.std(fold_zero_r2s)),
                "directional_acc_std": float(np.std(fold_zero_dir_accs))
            },
            "mean_baseline": {
                "mae_std": float(np.std(fold_mean_maes)),
                "rmse_std": float(np.std(fold_mean_rmses)),
                "r2_std": float(np.std(fold_mean_r2s)),
                "directional_acc_std": float(np.std(fold_mean_dir_accs))
            }
        },
        "folds": fold_metrics
    }


def main():
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    validator = WalkForwardValidator(n_splits=5, test_size_days=60, gap_days=7)

    # Core sample tickers for honest baseline evaluation
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "TATAMOTORS.NS", "ICICIBANK.NS", "AAPL", "MSFT"]
    evaluated_results = []

    print(f"Running Honest Walk-Forward Evaluation on {len(tickers)} sample tickers with Naive Baselines...")

    for ticker in tickers:
        csv_path = os.path.join(STOCK_DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(csv_path):
            continue
        try:
            df_raw = pd.read_csv(csv_path)
            res = evaluate_stock_walk_forward(df_raw, ticker, validator)
            if res:
                evaluated_results.append(res)
                ov_m = res["overall_metrics"]["model_rf"]
                ov_z = res["overall_metrics"]["zero_baseline"]
                ov_mn = res["overall_metrics"]["mean_baseline"]
                print(f"  [{ticker:<12}] Model RF MAE: {ov_m['mae']:.4f} (R2: {ov_m['r2']:.4f}, DirAcc: {ov_m['directional_acc']:.2f}%) | Zero MAE: {ov_z['mae']:.4f} | Mean MAE: {ov_mn['mae']:.4f}")
        except Exception as e:
            import traceback
            print(f"  ❌ Error evaluating {ticker}: {e}")
            traceback.print_exc()

    if not evaluated_results:
        print("❌ No tickers evaluated successfully.")
        return

    # Calculate global average & std metrics across evaluated tickers
    def get_avg_and_std(metric_key, model_key):
        vals = [r["overall_metrics"][model_key][metric_key] for r in evaluated_results]
        return float(np.mean(vals)), float(np.std(vals))

    model_mae_avg, model_mae_std = get_avg_and_std("mae", "model_rf")
    model_rmse_avg, model_rmse_std = get_avg_and_std("rmse", "model_rf")
    model_r2_avg, model_r2_std = get_avg_and_std("r2", "model_rf")
    model_dir_avg, model_dir_std = get_avg_and_std("directional_acc", "model_rf")

    zero_mae_avg, zero_mae_std = get_avg_and_std("mae", "zero_baseline")
    zero_rmse_avg, zero_rmse_std = get_avg_and_std("rmse", "zero_baseline")
    zero_r2_avg, zero_r2_std = get_avg_and_std("r2", "zero_baseline")
    zero_dir_avg, zero_dir_std = get_avg_and_std("directional_acc", "zero_baseline")

    mean_mae_avg, mean_mae_std = get_avg_and_std("mae", "mean_baseline")
    mean_rmse_avg, mean_rmse_std = get_avg_and_std("rmse", "mean_baseline")
    mean_r2_avg, mean_r2_std = get_avg_and_std("r2", "mean_baseline")
    mean_dir_avg, mean_dir_std = get_avg_and_std("directional_acc", "mean_baseline")

    summary_payload = {
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "validation_method": "Chronological Walk-Forward Validation with 7-Day Target Gap Embargo",
        "feature_set": "CANONICAL_FEATURE_COLS_V2 (Scale-Independent Ratios & Oscillators)",
        "target_type": "Log-Returns (Not Price Levels)",
        "gap_days": 7,
        "total_tickers_evaluated": len(evaluated_results),
        "mean_metrics_across_tickers": {
            "model_random_forest": {
                "mae": round(model_mae_avg, 5), "mae_std": round(model_mae_std, 5),
                "rmse": round(model_rmse_avg, 5), "rmse_std": round(model_rmse_std, 5),
                "r2": round(model_r2_avg, 5), "r2_std": round(model_r2_std, 5),
                "directional_accuracy_pct": round(model_dir_avg, 2), "directional_accuracy_std": round(model_dir_std, 2)
            },
            "zero_return_baseline": {
                "mae": round(zero_mae_avg, 5), "mae_std": round(zero_mae_std, 5),
                "rmse": round(zero_rmse_avg, 5), "rmse_std": round(zero_rmse_std, 5),
                "r2": round(zero_r2_avg, 5), "r2_std": round(zero_r2_std, 5),
                "directional_accuracy_pct": round(zero_dir_avg, 2), "directional_accuracy_std": round(zero_dir_std, 2)
            },
            "mean_return_baseline": {
                "mae": round(mean_mae_avg, 5), "mae_std": round(mean_mae_std, 5),
                "rmse": round(mean_rmse_avg, 5), "rmse_std": round(mean_rmse_std, 5),
                "r2": round(mean_r2_avg, 5), "r2_std": round(mean_r2_std, 5),
                "directional_accuracy_pct": round(mean_dir_avg, 2), "directional_accuracy_std": round(mean_dir_std, 2)
            }
        },
        "per_ticker_results": evaluated_results
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary_payload, f, indent=4)

    print(f"\n✅ Baseline honest metrics successfully saved to {OUTPUT_JSON}")
    print(f"   Model RF MAE: {model_mae_avg:.4f} (std: {model_mae_std:.4f}) | Zero MAE: {zero_mae_avg:.4f} | Mean MAE: {mean_mae_avg:.4f}")


if __name__ == "__main__":
    main()
