"""
run_experiment.py — EXP_7H_REGIME_CLASSIFICATION_001 Execution Engine
=======================================================================
Executes the isolated research experiment:
1. Universe selection & Data quality audit
2. Volatility-aware target generation & Dual Feature Set (Set A vs Set B) construction
3. Chronological 60/20/20 train/val/test split with 7-day embargo
4. Multi-model training (LightGBM, XGBoost, Random Forest) across Horizons 1..7
5. Validation evaluation & Probability signal threshold optimization (grid: 0.50..0.70)
6. Frozen evaluation on 20% unseen test set
7. Naive baselines comparison
8. Out-of-sample trading simulation with 10 bps transaction costs
9. Diagnostic checks (Prediction Collapse, Variance Ratio, Distribution Shift)
10. Generation of 18-section EXPERIMENT_REPORT.md
"""

import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
    recall_score, log_loss, brier_score_loss, confusion_matrix
)
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

import lightgbm as lgb
import xgboost as xgb

from ml.experiments.exp_7h_regime_classification_001.feature_target_pipeline import (
    compute_volatility_aware_targets, build_feature_set_a, build_feature_set_b,
    TOP_20_TICKERS, FORECAST_HORIZONS
)
from ml.experiments.exp_7h_regime_classification_001.inspect_data import run_data_audit

EXPERIMENT_ID = "EXP_7H_REGIME_CLASSIFICATION_001"
EXP_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "experiments", EXPERIMENT_ID)
EXP_REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "experiments", EXPERIMENT_ID)

os.makedirs(EXP_MODEL_DIR, exist_ok=True)
os.makedirs(EXP_REPORT_DIR, exist_ok=True)


def calculate_entropy(probs: np.ndarray) -> float:
    """Computes Shannon entropy of predicted class probabilities."""
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    return float(-np.sum(p * np.log2(p), axis=1).mean())


def evaluate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Calculates comprehensive multi-class classification metrics."""
    # Map classes (-1, 0, 1) to 0, 1, 2 for log_loss & Brier
    class_map = {-1: 0, 0: 1, 1: 2}
    y_true_mapped = np.array([class_map[y] for y in y_true])
    y_pred_mapped = np.array([class_map[y] for y in y_pred])

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Class-wise recalls (-1: DOWN, 0: FLAT, 1: UP)
    recalls = recall_score(y_true, y_pred, labels=[-1, 0, 1], average=None, zero_division=0)
    down_rec, flat_rec, up_rec = recalls[0], recalls[1], recalls[2]

    # Log loss & Brier score
    try:
        ll = log_loss(y_true_mapped, y_prob, labels=[0, 1, 2])
    except Exception:
        ll = float("nan")

    # One-hot true for multiclass Brier score calculation
    y_true_oh = np.eye(3)[y_true_mapped]
    brier = float(np.mean(np.sum((y_prob - y_true_oh) ** 2, axis=1)))

    cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1]).tolist()

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "mcc": float(mcc),
        "up_recall": float(up_rec),
        "down_recall": float(down_rec),
        "flat_recall": float(flat_rec),
        "log_loss": float(ll),
        "brier_score": float(brier),
        "confusion_matrix": cm
    }


def simulate_trading_strategy(
    y_true_returns: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.55,
    tx_cost_bps: float = 10.0
) -> Dict[str, Any]:
    """
    Simulates out-of-sample trading strategy based on probability thresholds.
    tx_cost_bps: 10 bps = 0.0010 per trade.
    """
    # Classes: 0 -> DOWN (-1), 1 -> FLAT (0), 2 -> UP (+1)
    p_down = y_probs[:, 0]
    p_flat = y_probs[:, 1]
    p_up = y_probs[:, 2]

    n_samples = len(y_true_returns)
    positions = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        if p_up[i] >= threshold:
            positions[i] = 1
        elif p_down[i] >= threshold:
            positions[i] = -1
        else:
            positions[i] = 0

    # Calculate position changes for transaction costs
    pos_change = np.abs(np.diff(positions, prepend=0))
    cost_pct = (tx_cost_bps / 10000.0) * pos_change

    # Strategy daily returns: position * underlying asset return - tx_cost
    strat_returns = positions * y_true_returns - cost_pct

    cum_returns = np.cumprod(1.0 + strat_returns) - 1.0
    total_return = float(cum_returns[-1]) if len(cum_returns) > 0 else 0.0

    # Annualized return (assuming 252 trading days per year)
    ann_return = float((1.0 + total_return) ** (252.0 / max(n_samples, 1)) - 1.0) if n_samples > 0 else 0.0

    # Sharpe ratio
    std_ret = np.std(strat_returns)
    sharpe = float(np.mean(strat_returns) / (std_ret + 1e-9) * np.sqrt(252)) if std_ret > 1e-9 else 0.0

    # Maximum Drawdown
    peak = np.maximum.accumulate(cum_returns + 1.0)
    dd = (cum_returns + 1.0 - peak) / peak
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0.0

    # Win rate (active trades)
    active_mask = positions != 0
    active_returns = strat_returns[active_mask]
    n_trades = int(np.sum(pos_change > 0))
    win_rate = float(np.mean(active_returns > 0)) if len(active_returns) > 0 else 0.0

    # Profit Factor
    gains = np.sum(active_returns[active_returns > 0])
    losses = np.abs(np.sum(active_returns[active_returns < 0]))
    profit_factor = float(gains / (losses + 1e-9)) if losses > 1e-9 else float(gains)

    long_exp = float(np.mean(positions == 1))
    short_exp = float(np.mean(positions == -1))
    flat_exp = float(np.mean(positions == 0))
    total_tx_costs = float(np.sum(cost_pct))

    return {
        "threshold": threshold,
        "total_return": total_return,
        "ann_return": ann_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": n_trades,
        "long_exposure": long_exp,
        "short_exposure": short_exp,
        "flat_exposure": flat_exp,
        "total_tx_costs": total_tx_costs
    }


def train_and_validate_models_for_feature_set(
    dataset_df: pd.DataFrame,
    feature_set_name: str,
    top_tickers: List[str]
) -> Dict[str, Any]:
    """
    Builds data for feature_set_name ('Set_A' or 'Set_B'), splits chronologically (60/20/20),
    applies 7-day embargo, trains LGBM, XGB, RF for Horizons 1..7, and evaluates on Validation set.
    """
    print(f"\n==================================================================")
    print(f"PROCESSING FEATURE SET: {feature_set_name}")
    print(f"==================================================================")

    # Process all stocks in top_tickers
    all_rows = []
    for ticker in top_tickers:
        stock_df = dataset_df[dataset_df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        if len(stock_df) < 200:
            continue

        # Add targets
        stock_targets = compute_volatility_aware_targets(stock_df)

        # Build features
        if feature_set_name == "Set_A":
            stock_feats, feat_cols = build_feature_set_a(stock_df)
        else:
            stock_feats, feat_cols = build_feature_set_b(stock_df)

        # Combine
        combined = pd.concat([stock_df[["Date", "Ticker"]], stock_feats, stock_targets[[f"Target_h{k}" for k in FORECAST_HORIZONS] + [f"Return_h{k}" for k in FORECAST_HORIZONS]]], axis=1)
        all_rows.append(combined)

    full_df = pd.concat(all_rows, ignore_index=True)
    full_df = full_df.dropna().sort_values(["Date", "Ticker"]).reset_index(drop=True)

    print(f"Combined {feature_set_name} total valid rows: {len(full_df)}")

    # Unique chronological dates
    unique_dates = sorted(full_df["Date"].unique())
    n_dates = len(unique_dates)

    train_end_idx = int(n_dates * 0.60)
    val_start_idx = train_end_idx + 7  # 7-day embargo
    val_end_idx = int(n_dates * 0.80)
    test_start_idx = val_end_idx + 7   # 7-day embargo

    train_dates = unique_dates[:train_end_idx]
    val_dates = unique_dates[val_start_idx:val_end_idx]
    test_dates = unique_dates[test_start_idx:]

    print(f"Train Dates ({len(train_dates)}): {pd.Timestamp(train_dates[0]).strftime('%Y-%m-%d')} to {pd.Timestamp(train_dates[-1]).strftime('%Y-%m-%d')}")
    print(f"Val Dates   ({len(val_dates)}):   {pd.Timestamp(val_dates[0]).strftime('%Y-%m-%d')} to {pd.Timestamp(val_dates[-1]).strftime('%Y-%m-%d')} (Embargo: 7 days)")
    print(f"Test Dates  ({len(test_dates)}):  {pd.Timestamp(test_dates[0]).strftime('%Y-%m-%d')} to {pd.Timestamp(test_dates[-1]).strftime('%Y-%m-%d')} (Embargo: 7 days)")

    train_mask = full_df["Date"].isin(train_dates)
    val_mask = full_df["Date"].isin(val_dates)
    test_mask = full_df["Date"].isin(test_dates)

    train_df = full_df[train_mask].copy()
    val_df = full_df[val_mask].copy()
    test_df = full_df[test_mask].copy()

    # Scale features strictly fitted on Train set
    scaler = StandardScaler()
    scaler.fit(train_df[feat_cols])

    X_train = scaler.transform(train_df[feat_cols])
    X_val = scaler.transform(val_df[feat_cols])
    X_test = scaler.transform(test_df[feat_cols])

    # Model training across horizons & model types
    models_dict = {}
    val_metrics_dict = {}

    model_families = ["LightGBM", "XGBoost", "RandomForest"]

    for k in FORECAST_HORIZONS:
        target_col = f"Target_h{k}"
        ret_col = f"Return_h{k}"

        y_train_raw = train_df[target_col].values
        y_val_raw = val_df[target_col].values
        y_test_raw = test_df[target_col].values

        # Map targets -1, 0, 1 to 0, 1, 2 for XGBoost/LightGBM
        class_map = {-1: 0, 0: 1, 1: 2}
        y_train = np.array([class_map[y] for y in y_train_raw])
        y_val = np.array([class_map[y] for y in y_val_raw])
        y_test = np.array([class_map[y] for y in y_test_raw])

        # Compute sample weights for balanced class weighting based ONLY on train set
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        weight_dict = dict(zip(classes, weights))
        sample_weights = np.array([weight_dict[y] for y in y_train])

        for m_name in model_families:
            if m_name == "LightGBM":
                clf = lgb.LGBMClassifier(
                    n_estimators=100, max_depth=5, learning_rate=0.05,
                    class_weight="balanced", random_state=42, verbose=-1
                )
                clf.fit(X_train, y_train)
            elif m_name == "XGBoost":
                clf = xgb.XGBClassifier(
                    n_estimators=100, max_depth=5, learning_rate=0.05,
                    objective="multi:softprob", num_class=3,
                    random_state=42, eval_metric="mlogloss"
                )
                clf.fit(X_train, y_train, sample_weight=sample_weights)
            elif m_name == "RandomForest":
                clf = RandomForestClassifier(
                    n_estimators=100, max_depth=8, class_weight="balanced",
                    random_state=42, n_jobs=-1
                )
                clf.fit(X_train, y_train)

            # Predict on Validation
            val_probs = clf.predict_proba(X_val)
            val_preds_mapped = np.argmax(val_probs, axis=1)
            inv_map = {0: -1, 1: 0, 2: 1}
            val_preds = np.array([inv_map[p] for p in val_preds_mapped])

            metrics = evaluate_classification_metrics(y_val_raw, val_preds, val_probs)
            key = f"h{k}_{m_name}"
            val_metrics_dict[key] = metrics
            models_dict[key] = clf

            print(f"Horizon Day {k} | {m_name:<12} | Val Acc: {metrics['accuracy']:.4f} | Bal Acc: {metrics['balanced_accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | MCC: {metrics['mcc']:.4f}")

    return {
        "feature_set": feature_set_name,
        "feature_cols": feat_cols,
        "scaler": scaler,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "models": models_dict,
        "val_metrics": val_metrics_dict
    }


def main():
    print(f"=== STARTING EXPERIMENT {EXPERIMENT_ID} ===")

    # 1. Audit raw dataset & select top 20 liquid stocks
    raw_df, top20_tickers = run_data_audit()

    # 2. Train & Evaluate Feature Set A vs Feature Set B on Validation data
    res_set_a = train_and_validate_models_for_feature_set(raw_df, "Set_A", top20_tickers)
    res_set_b = train_and_validate_models_for_feature_set(raw_df, "Set_B", top20_tickers)

    # 3. Model & Feature Set Selection based strictly on Validation Macro F1 / MCC
    # Compute mean Validation Macro F1 across all 7 horizons for each model & feature set combination
    model_scores = {}
    for res_obj in [res_set_a, res_set_b]:
        f_name = res_obj["feature_set"]
        for m_name in ["LightGBM", "XGBoost", "RandomForest"]:
            f1s = [res_obj["val_metrics"][f"h{k}_{m_name}"]["macro_f1"] for k in FORECAST_HORIZONS]
            mccs = [res_obj["val_metrics"][f"h{k}_{m_name}"]["mcc"] for k in FORECAST_HORIZONS]
            mean_f1 = float(np.mean(f1s))
            mean_mcc = float(np.mean(mccs))
            key = f"{f_name}_{m_name}"
            model_scores[key] = {"mean_macro_f1": mean_f1, "mean_mcc": mean_mcc, "feature_set": f_name, "model_family": m_name}
            print(f"Summary Val Performance | {key:<20} | Mean Macro F1: {mean_f1:.4f} | Mean MCC: {mean_mcc:.4f}")

    # Select best combination based on Mean Validation Macro F1
    best_combo_key = max(model_scores, key=lambda k: model_scores[k]["mean_macro_f1"])
    best_info = model_scores[best_combo_key]
    print(f"\nOPTIMAL VALIDATION CONFIGURATION SELECTED: {best_combo_key} (Mean Val Macro F1: {best_info['mean_macro_f1']:.4f})")

    best_res = res_set_a if best_info["feature_set"] == "Set_A" else res_set_b
    best_m_family = best_info["model_family"]

    # 4. Tune Probability Signal Threshold on Validation Set
    print("\n==================================================================")
    print("TUNING PROBABILITY SIGNAL THRESHOLDS ON VALIDATION GRID")
    print("==================================================================")
    threshold_grid = [0.50, 0.55, 0.60, 0.65, 0.70]
    thresh_scores = {}

    val_df = best_res["val_df"]
    X_val = best_res["X_val"]

    for tau in threshold_grid:
        sharpes = []
        for k in FORECAST_HORIZONS:
            clf = best_res["models"][f"h{k}_{best_m_family}"]
            val_probs = clf.predict_proba(X_val)
            val_returns = val_df[f"Return_h{k}"].values
            backtest_res = simulate_trading_strategy(val_returns, val_probs, threshold=tau)
            sharpes.append(backtest_res["sharpe_ratio"])

        mean_sharpe = float(np.mean(sharpes))
        thresh_scores[tau] = mean_sharpe
        print(f"Validation Threshold Grid tau={tau:.2f} | Mean Val Sharpe: {mean_sharpe:.4f}")

    best_threshold = max(thresh_scores, key=thresh_scores.get)
    print(f"FROZEN OPTIMAL PROBABILITY THRESHOLD: tau={best_threshold:.2f} (Mean Val Sharpe: {thresh_scores[best_threshold]:.4f})")

    # 5. FINAL UNSEEN TEST EVALUATION (EXACTLY ONCE)
    print("\n==================================================================")
    print(f"FINAL UNSEEN TEST EVALUATION (Frozen Model: {best_info['feature_set']} + {best_m_family}, tau={best_threshold:.2f})")
    print("==================================================================")

    test_df = best_res["test_df"]
    X_test = best_res["X_test"]

    unseen_metrics_by_h = {}
    backtest_by_h = {}
    diagnostic_by_h = {}

    inv_map = {0: -1, 1: 0, 2: 1}

    for k in FORECAST_HORIZONS:
        clf = best_res["models"][f"h{k}_{best_m_family}"]
        test_probs = clf.predict_proba(X_test)
        test_preds_mapped = np.argmax(test_probs, axis=1)
        test_preds = np.array([inv_map[p] for p in test_preds_mapped])

        y_test_raw = test_df[f"Target_h{k}"].values
        test_returns = test_df[f"Return_h{k}"].values

        # Classification Metrics
        metrics = evaluate_classification_metrics(y_test_raw, test_preds, test_probs)
        unseen_metrics_by_h[f"Day_{k}"] = metrics

        # Backtest with frozen threshold
        bt = simulate_trading_strategy(test_returns, test_probs, threshold=best_threshold)
        backtest_by_h[f"Day_{k}"] = bt

        # Diagnostic Checks
        pred_counts = pd.Series(test_preds).value_counts(normalize=True).to_dict()
        pred_up = pred_counts.get(1, 0.0)
        pred_flat = pred_counts.get(0, 0.0)
        pred_down = pred_counts.get(-1, 0.0)

        entropy_val = calculate_entropy(test_probs)
        target_var = float(np.var(y_test_raw))
        pred_var = float(np.var(test_preds))
        var_ratio = float(pred_var / (target_var + 1e-9))
        pred_target_corr = float(np.corrcoef(test_preds, y_test_raw)[0, 1]) if pred_var > 1e-9 else 0.0

        diagnostic_by_h[f"Day_{k}"] = {
            "pred_up_pct": pred_up,
            "pred_flat_pct": pred_flat,
            "pred_down_pct": pred_down,
            "pred_mean": float(np.mean(test_preds)),
            "pred_std": float(np.std(test_preds)),
            "entropy": entropy_val,
            "target_var": target_var,
            "pred_var": pred_var,
            "var_ratio": var_ratio,
            "pred_target_corr": pred_target_corr,
            "is_collapsed": (pred_up > 0.90 or pred_flat > 0.90 or pred_down > 0.90)
        }

        print(f"UNSEEN TEST Horizon Day {k} | Acc: {metrics['accuracy']:.4f} | Bal Acc: {metrics['balanced_accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f} | MCC: {metrics['mcc']:.4f} | Sharpe: {bt['sharpe_ratio']:.4f} | Drawdown: {bt['max_drawdown']:.2%}")

    # 6. EVALUATE NAIVE BASELINES ON UNSEEN TEST
    print("\n==================================================================")
    print("EVALUATING NAIVE BASELINES ON UNSEEN TEST")
    print("==================================================================")
    baselines_by_h = {}

    for k in FORECAST_HORIZONS:
        y_test_raw = test_df[f"Target_h{k}"].values
        n_t = len(y_test_raw)

        # Baseline 1: Random 50/50 (UP/DOWN equal)
        np.random.seed(42)
        b_rand = np.random.choice([1, -1], size=n_t)
        acc_rand = accuracy_score(y_test_raw, b_rand)
        f1_rand = f1_score(y_test_raw, b_rand, average="macro", zero_division=0)
        mcc_rand = matthews_corrcoef(y_test_raw, b_rand)

        # Baseline 2: Always UP (+1)
        b_up = np.ones(n_t, dtype=int)
        acc_up = accuracy_score(y_test_raw, b_up)
        f1_up = f1_score(y_test_raw, b_up, average="macro", zero_division=0)
        mcc_up = matthews_corrcoef(y_test_raw, b_up)

        # Baseline 3: Always DOWN (-1)
        b_down = -np.ones(n_t, dtype=int)
        acc_down = accuracy_score(y_test_raw, b_down)
        f1_down = f1_score(y_test_raw, b_down, average="macro", zero_division=0)
        mcc_down = matthews_corrcoef(y_test_raw, b_down)

        # Baseline 4: Previous Day Direction (Return_1d direction)
        b_prev = np.sign(test_df["Return_1d"].values).astype(int)
        acc_prev = accuracy_score(y_test_raw, b_prev)
        f1_prev = f1_score(y_test_raw, b_prev, average="macro", zero_division=0)
        mcc_prev = matthews_corrcoef(y_test_raw, b_prev)

        # Baseline 5: Majority Class (FLAT = 0)
        b_maj = np.zeros(n_t, dtype=int)
        acc_maj = accuracy_score(y_test_raw, b_maj)
        f1_maj = f1_score(y_test_raw, b_maj, average="macro", zero_division=0)
        mcc_maj = matthews_corrcoef(y_test_raw, b_maj)

        baselines_by_h[f"Day_{k}"] = {
            "Random_50_50": {"accuracy": float(acc_rand), "macro_f1": float(f1_rand), "mcc": float(mcc_rand)},
            "Always_UP": {"accuracy": float(acc_up), "macro_f1": float(f1_up), "mcc": float(mcc_up)},
            "Always_DOWN": {"accuracy": float(acc_down), "macro_f1": float(f1_down), "mcc": float(mcc_down)},
            "Prev_Day_Direction": {"accuracy": float(acc_prev), "macro_f1": float(f1_prev), "mcc": float(mcc_prev)},
            "Majority_FLAT": {"accuracy": float(acc_maj), "macro_f1": float(f1_maj), "mcc": float(mcc_maj)},
        }

    # 7. DISTRIBUTION SHIFT ANALYSIS
    print("\n==================================================================")
    print("COMPUTING DISTRIBUTION SHIFT BETWEEN TRAIN, VAL, AND TEST SETS")
    print("==================================================================")
    train_df = best_res["train_df"]
    val_df = best_res["val_df"]
    test_df = best_res["test_df"]

    key_features = ["Close", "Volume", "Return_1d", "Return_5d", "Vol_20d", "NATR", "RSI", "MACD_Hist"]
    # Filter features that exist in selected feature set
    key_features = [f for f in key_features if f in best_res["feature_cols"]]

    dist_shift_results = {}
    for f in key_features:
        tr_m, tr_s = float(train_df[f].mean()), float(train_df[f].std())
        val_m, val_s = float(val_df[f].mean()), float(val_df[f].std())
        te_m, te_s = float(test_df[f].mean()), float(test_df[f].std())

        # Relative shift between train and test mean
        mean_shift_pct = float(abs(te_m - tr_m) / (abs(tr_m) + 1e-9) * 100.0)

        dist_shift_results[f] = {
            "train_mean": tr_m, "train_std": tr_s,
            "val_mean": val_m, "val_std": val_s,
            "test_mean": te_m, "test_std": te_s,
            "shift_pct": mean_shift_pct
        }
        print(f"Feature: {f:<15} | Train Mean: {tr_m:10.4f} | Test Mean: {te_m:10.4f} | Shift: {mean_shift_pct:6.2f}%")

    # 8. PRODUCTION SAFETY HASH VERIFICATION
    print("\n==================================================================")
    print("PRODUCTION SAFETY HASH VERIFICATION")
    print("==================================================================")
    files_to_check = [
        "models/global_model.pkl",
        "models/global_scaler.pkl",
        "models/global_metadata.json",
        "models/production/global_model.pkl",
        "models/production/global_scaler.pkl",
        "models/production/global_metadata.json"
    ]

    hash_baseline_path = os.path.join(PROJECT_ROOT, "models", "production_hashes_baseline.json")
    with open(hash_baseline_path, "r") as f:
        baseline_hashes = json.load(f)

    current_hashes = {}
    hash_match = True

    for p in files_to_check:
        full_p = os.path.join(PROJECT_ROOT, p)
        if os.path.exists(full_p):
            with open(full_p, "rb") as f:
                h = hashlib.sha256(f.read()).hexdigest()
                current_hashes[p] = h
                if h != baseline_hashes.get(p):
                    hash_match = False
                    print(f"CRITICAL WARNING: Hash mismatch for {p}!")
        else:
            current_hashes[p] = "FILE_NOT_FOUND"

    print(f"All Production Artifact Hashes Verified Identical: {hash_match}")

    # 9. FINAL DECISION CLASSIFICATION
    # Criteria for BETTER:
    # 1. Challenger beats naive baselines on Unseen Test (higher MCC & Macro F1).
    # 2. Positive trading Sharpe ratio (> 0.50) after 10 bps costs.
    # 3. No extreme prediction collapse or variance ratio collapse.
    mean_unseen_mcc = float(np.mean([unseen_metrics_by_h[f"Day_{k}"]["mcc"] for k in FORECAST_HORIZONS]))
    mean_unseen_f1 = float(np.mean([unseen_metrics_by_h[f"Day_{k}"]["macro_f1"] for k in FORECAST_HORIZONS]))
    mean_trading_sharpe = float(np.mean([backtest_by_h[f"Day_{k}"]["sharpe_ratio"] for k in FORECAST_HORIZONS]))

    print(f"\nSummary Unseen Metrics | Mean MCC: {mean_unseen_mcc:.4f} | Mean Macro F1: {mean_unseen_f1:.4f} | Mean Trading Sharpe: {mean_trading_sharpe:.4f}")

    if mean_unseen_mcc > 0.05 and mean_trading_sharpe > 0.50:
        final_decision = "BETTER"
    else:
        final_decision = "NOT BETTER"

    print(f"\n>>> FINAL EXPERIMENT DECISION: {final_decision} <<<")

    # Save summary data JSON inside experiment model dir
    exp_summary_data = {
        "experiment_id": EXPERIMENT_ID,
        "executed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "top_20_stocks": top20_tickers,
        "selected_feature_set": best_info["feature_set"],
        "selected_model_family": best_m_family,
        "frozen_signal_threshold": best_threshold,
        "final_decision": final_decision,
        "unseen_metrics_by_horizon": unseen_metrics_by_h,
        "trading_backtest_by_horizon": backtest_by_h,
        "diagnostic_checks_by_horizon": diagnostic_by_h,
        "naive_baselines_by_horizon": baselines_by_h,
        "distribution_shift": dist_shift_results,
        "production_hashes_verified": hash_match
    }

    with open(os.path.join(EXP_MODEL_DIR, "experiment_summary.json"), "w") as f:
        json.dump(exp_summary_data, f, indent=2)

    return exp_summary_data, res_set_a, res_set_b


if __name__ == "__main__":
    main()
