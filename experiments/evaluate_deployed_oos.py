"""
experiments/evaluate_deployed_oos.py — True Out-of-Sample Evaluation Engine
=============================================================================
Evaluates the deployed global_model.pkl (WITHOUT retraining) on the unseen
out-of-sample window from 2026-01-01 to 2026-10-09 across the 187 clean universe.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering import build_canonical_feature_set, clean_ohlcv, CANONICAL_FEATURE_COLS, FORECAST_HORIZON
from predict import get_model_manager

def main():
    print("Loading eligible universe tickers...")
    with open(os.path.join("price_data_clean", "eligible_tickers.json"), "r") as f:
        uni_data = json.load(f)
    tickers = uni_data.get("eligible_tickers", [])
    print(f"Eligible universe count: {len(tickers)}")

    # Load deployed global model & scaler
    model_mgr = get_model_manager()
    g_model, g_scaler = model_mgr.get_loaded_model()
    scaler_X = g_scaler[0] if isinstance(g_scaler, tuple) else g_scaler
    n_expected = getattr(scaler_X, "n_features_in_", 30)

    start_date = "2026-01-01"
    end_date = "2026-10-09"

    records = []

    print(f"Processing OOS predictions for window {start_date} to {end_date}...")
    for t in tickers:
        csv_candidates = [
            os.path.join("price_data_clean", f"{t}.csv"),
            os.path.join("price_data_clean", f"{t}"),
            os.path.join("price_data_clean", f"{t.replace('.NS', '')}.csv")
        ]
        csv_path = next((c for c in csv_candidates if os.path.exists(c)), None)
        if not csv_path:
            continue

        try:
            df_raw = pd.read_csv(csv_path)
            df_clean = clean_ohlcv(df_raw, ticker=t)
            if df_clean.empty or len(df_clean) < 50:
                continue

            # Calculate 7-day forward return target
            close = df_clean["Close"]
            df_clean["y_true_abs"] = np.log(close.shift(-7) / (close + 1e-9))
            
            # Feature engineering
            df_feat = build_canonical_feature_set(df_clean, ticker=t)
            
            # Filter rows within OOS date window
            df_feat["Date_str"] = df_feat["Date"].astype(str).str[:10]
            mask = (df_feat["Date_str"] >= start_date) & (df_feat["Date_str"] <= end_date) & df_feat["y_true_abs"].notna()
            df_oos = df_feat[mask].copy()

            if df_oos.empty:
                continue

            feature_cols = [c for c in CANONICAL_FEATURE_COLS if c in df_oos.columns]
            X_raw = df_oos[feature_cols].values

            # Apply scaler slicing logic identical to predict.py
            if X_raw.shape[1] > n_expected:
                X_input = X_raw[:, :n_expected]
            elif X_raw.shape[1] < n_expected:
                pad_cols = n_expected - X_raw.shape[1]
                X_input = np.pad(X_raw, ((0, 0), (0, pad_cols)), mode="constant")
            else:
                X_input = X_raw

            X_scaled = scaler_X.transform(X_input)
            
            # Predict multi-output log returns (d1..d7). We take d7 prediction for 7-day horizon evaluation.
            preds_all = g_model.predict(X_scaled)
            if preds_all.ndim == 2 and preds_all.shape[1] >= 7:
                y_pred_7d = preds_all[:, 6]  # 7-day forecast
            elif preds_all.ndim == 2:
                y_pred_7d = preds_all[:, -1]
            else:
                y_pred_7d = preds_all

            df_oos["y_pred_7d"] = y_pred_7d
            
            for idx, row in df_oos.iterrows():
                records.append({
                    "ticker": t,
                    "date": row["Date_str"],
                    "y_true_abs": float(row["y_true_abs"]),
                    "y_pred_7d": float(row["y_pred_7d"])
                })

        except Exception as e:
            continue

    df_res = pd.DataFrame(records)
    print(f"Total OOS evaluation instances collected: {len(df_res)}")

    if df_res.empty:
        raise ValueError("No OOS rows collected.")

    # Calculate Market-Relative target (y_true_mkt = y_true_abs - cross_sectional_mean_per_date)
    date_means = df_res.groupby("date")["y_true_abs"].transform("mean")
    df_res["y_true_mkt"] = df_res["y_true_abs"] - date_means

    # 1. MAE Metrics
    mae_model = float(np.mean(np.abs(df_res["y_pred_7d"] - df_res["y_true_abs"])))
    mae_zero = float(np.mean(np.abs(df_res["y_true_abs"])))

    # 2. Directional Accuracy Metrics
    y_true_sign = np.sign(df_res["y_true_abs"])
    y_pred_sign = np.sign(df_res["y_pred_7d"])
    
    dir_acc_model = float(np.mean((y_pred_sign == y_true_sign).astype(float)) * 100.0)
    dir_acc_always_up = float(np.mean((y_true_sign >= 0).astype(float)) * 100.0)
    dir_acc_always_down = float(np.mean((y_true_sign < 0).astype(float)) * 100.0)

    # 3. Information Coefficient (IC) sampled every 7th date to avoid overlapping autocorrelation
    unique_dates = sorted(df_res["date"].unique())
    sampled_dates = unique_dates[::7]
    df_sampled = df_res[df_res["date"].isin(sampled_dates)].copy()

    ic_abs, _ = stats.spearmanr(df_sampled["y_pred_7d"], df_sampled["y_true_abs"])
    ic_mkt, _ = stats.spearmanr(df_sampled["y_pred_7d"], df_sampled["y_true_mkt"])
    
    ic_abs = float(ic_abs) if not np.isnan(ic_abs) else 0.0
    ic_mkt = float(ic_mkt) if not np.isnan(ic_mkt) else 0.0

    # 4. Date-Level Block Bootstrap 95% CI for IC (Absolute & Market-Relative)
    np.random.seed(42)
    n_boot = 1000
    boot_ics_abs = []
    boot_ics_mkt = []

    for _ in range(n_boot):
        sample_d = np.random.choice(sampled_dates, size=len(sampled_dates), replace=True)
        boot_df = df_sampled[df_sampled["date"].isin(sample_d)]
        if len(boot_df) > 10:
            r_abs, _ = stats.spearmanr(boot_df["y_pred_7d"], boot_df["y_true_abs"])
            r_mkt, _ = stats.spearmanr(boot_df["y_pred_7d"], boot_df["y_true_mkt"])
            if not np.isnan(r_abs): boot_ics_abs.append(r_abs)
            if not np.isnan(r_mkt): boot_ics_mkt.append(r_mkt)

    ci_abs_low, ci_abs_high = float(np.percentile(boot_ics_abs, 2.5)), float(np.percentile(boot_ics_abs, 97.5))
    ci_mkt_low, ci_mkt_high = float(np.percentile(boot_ics_mkt, 2.5)), float(np.percentile(boot_ics_mkt, 97.5))

    oos_results = {
        "evaluation_period": f"{start_date} to {end_date}",
        "total_evaluated_rows": len(df_res),
        "total_trading_dates": len(unique_dates),
        "sampled_dates_step_7": len(sampled_dates),
        "mae": {
            "model_mae": round(mae_model, 6),
            "zero_baseline_mae": round(mae_zero, 6),
            "mae_delta": round(mae_model - mae_zero, 6)
        },
        "directional_accuracy_pct": {
            "model": round(dir_acc_model, 2),
            "always_up_baseline": round(dir_acc_always_up, 2),
            "always_down_baseline": round(dir_acc_always_down, 2)
        },
        "spearman_ic_sampled_step_7": {
            "absolute_ic": round(ic_abs, 6),
            "absolute_ic_95_ci": [round(ci_abs_low, 6), round(ci_abs_high, 6)],
            "market_relative_ic": round(ic_mkt, 6),
            "market_relative_ic_95_ci": [round(ci_mkt_low, 6), round(ci_mkt_high, 6)]
        }
    }

    out_path = os.path.join("experiments", "sentiment_v1", "deployed_oos.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(oos_results, f, indent=4)

    print("\n=== DEPLOYED MODEL OOS EVALUATION RESULTS ===")
    print(json.dumps(oos_results, indent=2))
    print(f"\nSaved OOS results cleanly to {out_path}!")

if __name__ == "__main__":
    main()
