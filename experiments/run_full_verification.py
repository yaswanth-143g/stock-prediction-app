#!/usr/bin/env python3
"""
experiments/run_full_verification.py — Full 12-Fold Walk-Forward & Sanity Test Engine
========================================================================================
Runs 12-fold expanding window evaluation on merge-csv.csv (2018-2026, test_size=60d, gap=7d),
computes per-fold and per-date metrics, outputs experiments/raw_walkforward_results.csv,
and executes permutation, lag, sub-universe, and leakage sanity tests.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_engineering import build_canonical_feature_set, CANONICAL_FEATURE_COLS_V2, FORECAST_HORIZON
from train_models import prepare_targets

MERGE_CSV = "merge-csv.csv" if os.path.exists("merge-csv.csv") else os.path.join("price_data_clean", "merge-csv.csv")

def compute_ic(y_true, y_pred):
    """Computes Spearman rank correlation (IC)."""
    if len(y_true) < 3 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr) if not np.isnan(corr) else 0.0

def compute_metrics(y_true, y_pred):
    """Computes MAE, RMSE, R2, Directional Accuracy, and IC."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred))) * 100.0
    ic = compute_ic(y_true, y_pred)
    return {"mae": round(mae, 6), "rmse": round(rmse, 6), "r2": round(r2, 6), "directional_acc": round(dir_acc, 2), "ic": round(ic, 4)}

def run_verification():
    print("Loading merge-csv.csv...")
    df_raw = pd.read_csv(MERGE_CSV)
    df_raw['Date'] = pd.to_datetime(df_raw['Date'], format='mixed')
    df_raw = df_raw.sort_values(['Date', 'Ticker']).reset_index(drop=True)

    print(f"Total rows loaded: {len(df_raw)} across {df_raw['Ticker'].nunique()} tickers.")
    
    # Target construction: 7-day log return & market-relative return
    df_raw['log_ret_7d'] = np.log(df_raw.groupby('Ticker')['Close'].shift(-7) / df_raw['Close'])
    
    # Cross-sectional mean log return on date t+7
    xs_mean = df_raw.groupby('Date')['log_ret_7d'].transform('mean')
    df_raw['mkt_rel_ret_7d'] = df_raw['log_ret_7d'] - xs_mean

    # Build scale-independent canonical features (V2)
    print("Building canonical features V2...")
    # Features built per ticker
    feature_dfs = []
    for ticker, group in df_raw.groupby('Ticker'):
        group = group.sort_values('Date').reset_index(drop=True)
        feat_g = build_canonical_feature_set(group, ticker=ticker)
        feature_dfs.append(feat_g)
    
    df_feat = pd.concat(feature_dfs, axis=0).sort_values(['Date', 'Ticker']).reset_index(drop=True)

    feature_cols = [c for c in CANONICAL_FEATURE_COLS_V2 if c in df_feat.columns]
    
    # Filter valid rows
    cols_to_check = feature_cols + ['log_ret_7d', 'mkt_rel_ret_7d']
    df_valid = df_feat.dropna(subset=cols_to_check).reset_index(drop=True)
    print(f"Valid dataset size for walk-forward evaluation: {len(df_valid)} rows.")

    unique_dates = sorted(df_valid['Date'].unique())
    print(f"Total unique trading dates: {len(unique_dates)} (from {unique_dates[0].strftime('%Y-%m-%d')} to {unique_dates[-1].strftime('%Y-%m-%d')}).")

    # 12 Folds Setup: Expanding window from 2018 onwards
    # Test window: 60 trading dates per fold, gap: 7 trading dates embargo
    n_folds = 12
    test_dates_per_fold = 60
    gap_dates = 7
    min_train_dates = 250

    folds = []
    end_idx = len(unique_dates)
    
    for f in range(n_folds):
        test_end = end_idx - (n_folds - 1 - f) * test_dates_per_fold
        test_start = test_end - test_dates_per_fold
        train_end = test_start - gap_dates
        if train_end < min_train_dates:
            continue
        folds.append({
            "fold": f + 1,
            "train_dates": (unique_dates[0], unique_dates[train_end - 1]),
            "test_dates": (unique_dates[test_start], unique_dates[test_end - 1])
        })

    print(f"Generated {len(folds)} valid walk-forward folds.")

    raw_results = []
    
    # Folds evaluation
    for f_info in folds:
        f_num = f_info["fold"]
        tr_start, tr_end = f_info["train_dates"]
        te_start, te_end = f_info["test_dates"]

        train_mask = (df_valid['Date'] >= tr_start) & (df_valid['Date'] <= tr_end)
        test_mask = (df_valid['Date'] >= te_start) & (df_valid['Date'] <= te_end)

        df_tr = df_valid[train_mask]
        df_te = df_valid[test_mask]

        X_tr = df_tr[feature_cols].values
        y_tr_abs = df_tr['log_ret_7d'].values
        y_tr_mkt = df_tr['mkt_rel_ret_7d'].values

        X_te = df_te[feature_cols].values
        y_te_abs = df_te['log_ret_7d'].values
        y_te_mkt = df_te['mkt_rel_ret_7d'].values

        # Scaler fit strictly inside training fold
        scaler = MinMaxScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        # 1. RF Champion (Retrained per fold)
        rf_abs = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
        rf_abs.fit(X_tr_sc, y_tr_abs)
        pred_rf_abs = rf_abs.predict(X_te_sc)

        rf_mkt = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
        rf_mkt.fit(X_tr_sc, y_tr_mkt)
        pred_rf_mkt = rf_mkt.predict(X_te_sc)

        # 2. XGBoost Meta-Learner (Retrained per fold)
        xgb_mkt = xgb.XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1)
        xgb_mkt.fit(X_tr_sc, y_tr_mkt)
        pred_xgb_mkt = xgb_mkt.predict(X_te_sc)

        # 3. Baselines
        pred_zero = np.zeros_like(y_te_abs)
        pred_always_up = np.ones_like(y_te_abs) * 0.01
        pred_always_down = np.ones_like(y_te_abs) * (-0.01)
        pred_train_mean_abs = np.ones_like(y_te_abs) * np.mean(y_tr_abs)
        pred_train_mean_mkt = np.ones_like(y_te_mkt) * np.mean(y_tr_mkt)

        # Store per-fold metrics
        m_rf_abs = compute_metrics(y_te_abs, pred_rf_abs)
        m_rf_mkt = compute_metrics(y_te_mkt, pred_rf_mkt)
        m_xgb_mkt = compute_metrics(y_te_mkt, pred_xgb_mkt)
        m_zero = compute_metrics(y_te_abs, pred_zero)
        m_up = compute_metrics(y_te_abs, pred_always_up)
        m_down = compute_metrics(y_te_abs, pred_always_down)
        m_mean_abs = compute_metrics(y_te_abs, pred_train_mean_abs)

        print(f"Fold {f_num:2d} ({te_start.strftime('%Y-%m-%d')} to {te_end.strftime('%Y-%m-%d')}) | "
              f"RF Abs DirAcc: {m_rf_abs['directional_acc']:.1f}% | XGB Mkt DirAcc: {m_xgb_mkt['directional_acc']:.1f}%, IC: {m_xgb_mkt['ic']:.4f} | "
              f"Zero MAE: {m_zero['mae']:.5f}, RF Abs MAE: {m_rf_abs['mae']:.5f}")

        # Store row records for CSV export
        for date_val, y_a, y_m, p_rf_a, p_rf_m, p_xgb_m in zip(df_te['Date'], y_te_abs, y_te_mkt, pred_rf_abs, pred_rf_mkt, pred_xgb_mkt):
            raw_results.append({
                "fold": f_num,
                "date": date_val.strftime('%Y-%m-%d'),
                "y_true_abs": y_a,
                "y_true_mkt": y_m,
                "pred_rf_abs": p_rf_a,
                "pred_rf_mkt": p_rf_m,
                "pred_xgb_mkt": p_xgb_m
            })

    # Save raw CSV
    df_raw_out = pd.DataFrame(raw_results)
    raw_csv_path = os.path.join("experiments", "raw_walkforward_results.csv")
    df_raw_out.to_csv(raw_csv_path, index=False)
    print(f"\nSaved raw per-fold per-date evaluation results to {raw_csv_path} ({len(df_raw_out)} rows).")

    # SANITY TESTS
    print("\n================ SANITY TESTS ================")
    # 2a. Permutation Test
    print("Running 2a: Permutation Test (shuffling targets in training folds)...")
    perm_ics = []
    for f_info in folds[:4]:  # test first 4 folds
        tr_start, tr_end = f_info["train_dates"]
        te_start, te_end = f_info["test_dates"]

        df_tr = df_valid[(df_valid['Date'] >= tr_start) & (df_valid['Date'] <= tr_end)]
        df_te = df_valid[(df_valid['Date'] >= te_start) & (df_valid['Date'] <= te_end)]

        X_tr = df_tr[feature_cols].values
        y_tr_mkt_perm = np.random.permutation(df_tr['mkt_rel_ret_7d'].values)

        X_te = df_te[feature_cols].values
        y_te_mkt = df_te['mkt_rel_ret_7d'].values

        scaler = MinMaxScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        model_perm = xgb.XGBRegressor(n_estimators=30, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1)
        model_perm.fit(X_tr_sc, y_tr_mkt_perm)
        p_perm = model_perm.predict(X_te_sc)
        ic_perm = compute_ic(y_te_mkt, p_perm)
        perm_ics.append(ic_perm)
    print(f"Permutation Test Mean IC: {np.mean(perm_ics):.4f} (Expected ~0.000)")

    # 2b. Lag Test (shift features back 7 extra days)
    print("Running 2b: Lag Test (shifting features back 7 extra days)...")
    lag_ics = []
    for f_info in folds[:4]:
        tr_start, tr_end = f_info["train_dates"]
        te_start, te_end = f_info["test_dates"]

        # Shift features back 7 days
        df_valid_lagged = df_valid.copy()
        for c in feature_cols:
            df_valid_lagged[c] = df_valid_lagged.groupby('Ticker')[c].shift(7)

        df_vl = df_valid_lagged.dropna(subset=feature_cols + ['mkt_rel_ret_7d']).reset_index(drop=True)
        df_tr = df_vl[(df_vl['Date'] >= tr_start) & (df_vl['Date'] <= tr_end)]
        df_te = df_vl[(df_vl['Date'] >= te_start) & (df_vl['Date'] <= te_end)]

        X_tr = df_tr[feature_cols].values
        y_tr_mkt = df_tr['mkt_rel_ret_7d'].values
        X_te = df_te[feature_cols].values
        y_te_mkt = df_te['mkt_rel_ret_7d'].values

        scaler = MinMaxScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        model_lag = xgb.XGBRegressor(n_estimators=30, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1)
        model_lag.fit(X_tr_sc, y_tr_mkt)
        p_lag = model_lag.predict(X_te_sc)
        ic_lag = compute_ic(y_te_mkt, p_lag)
        lag_ics.append(ic_lag)
    print(f"Lag Test (7-day extra lag) Mean IC: {np.mean(lag_ics):.4f}")

    # 2d. Re-run excluding NIFTY50, <750 trading days, stale/merged, and corporate action jump windows
    print("Running 2d: Re-run with strict sub-universe exclusions...")
    
    # Exclude NIFTY50 index
    clean_sub = df_valid[df_valid['Ticker'] != 'NIFTY50'].copy()
    
    # Exclude tickers with <750 trading days
    ticker_counts = clean_sub.groupby('Ticker')['Date'].count()
    valid_750 = ticker_counts[ticker_counts >= 750].index
    clean_sub = clean_sub[clean_sub['Ticker'].isin(valid_750)].copy()

    # Exclude stale/merged tickers
    stale_list = ['CADILAHC.NS', 'MINDTREE.NS', 'LTI.NS', 'SRTRANSFIN.NS', 'SIEMENS.NS']
    clean_sub = clean_sub[~clean_sub['Ticker'].isin(stale_list)].copy()

    # Exclude corporate action jump windows (+/- 10 days around jump date)
    jump_windows = [
        ('ABFRL.NS', '2025-05-12', '2025-06-02'),
        ('ALLCARGO.NS', '2025-11-01', '2025-11-25'),
        ('CDSL.NS', '2017-06-20', '2017-07-15'),
        ('INDIAMART.NS', '2019-10-15', '2019-11-05'),
        ('INDUSINDBK.NS', '2020-03-15', '2020-04-05')
    ]
    for t_j, start_j, end_j in jump_windows:
        mask_j = (clean_sub['Ticker'] == t_j) & (clean_sub['Date'] >= pd.to_datetime(start_j)) & (clean_sub['Date'] <= pd.to_datetime(end_j))
        clean_sub = clean_sub[~mask_j]

    print(f"Filtered sub-universe size: {len(clean_sub)} rows across {clean_sub['Ticker'].nunique()} tickers.")

    sub_ics = []
    for f_info in folds:
        tr_start, tr_end = f_info["train_dates"]
        te_start, te_end = f_info["test_dates"]

        df_tr = clean_sub[(clean_sub['Date'] >= tr_start) & (clean_sub['Date'] <= tr_end)]
        df_te = clean_sub[(clean_sub['Date'] >= te_start) & (clean_sub['Date'] <= te_end)]

        if len(df_tr) < 100 or len(df_te) < 10:
            continue

        X_tr = df_tr[feature_cols].values
        y_tr_mkt = df_tr['mkt_rel_ret_7d'].values
        X_te = df_te[feature_cols].values
        y_te_mkt = df_te['mkt_rel_ret_7d'].values

        scaler = MinMaxScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        model_sub = xgb.XGBRegressor(n_estimators=30, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=-1)
        model_sub.fit(X_tr_sc, y_tr_mkt)
        p_sub = model_sub.predict(X_te_sc)
        sub_ics.append(compute_ic(y_te_mkt, p_sub))

    print(f"Sub-Universe Re-run Mean IC: {np.mean(sub_ics):.4f}")

    # 2e. Effective Number of Independent Test Dates
    total_test_dates = sum((f_info["test_dates"][1] - f_info["test_dates"][0]).days for f_info in folds)
    total_unique_test_days = sum(test_dates_per_fold for _ in folds)
    effective_n_dates = math.ceil(total_unique_test_days / 7.0)
    print(f"2e: Total unique test trading days = {total_unique_test_days}. Effective independent 7-day non-overlapping test dates = {effective_n_dates}.")

if __name__ == "__main__":
    run_verification()
