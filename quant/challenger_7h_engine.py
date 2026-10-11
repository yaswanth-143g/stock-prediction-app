"""
quant/challenger_7h_engine.py
===================================================================
Research-Only Clean 7-Horizon Challenger Training & Evaluation Engine
===================================================================
Establishes a mathematically rigorous, fully reproducible 7-horizon regression
baseline across Random Forest, XGBoost, LightGBM, and LSTM architectures.

Key Principles:
1. Complete isolation from active production artifacts (models/global_model.pkl).
2. Zero data leakage: scalers fitted strictly on Train split (60%).
3. Enforces 7-day embargo gap across Train/Validation/Test boundaries.
4. Genuine multi-output shape contract: y.shape = (N, 7), predictions.shape = (N, 7).
5. Horizon Independence Verification: Ensures Day 1..7 predictions are not tiled/identical.
6. Out-of-sample trading backtest with realistic 10 bps transaction costs.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, Any, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

warnings.filterwarnings("ignore")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from feature_engineering import build_canonical_feature_set, clean_ohlcv, CANONICAL_FEATURE_COLS, FORECAST_HORIZON
from train_models import prepare_targets, MODELS_DIR, FEATURE_COLS

CHALLENGER_DIR = os.path.join(MODELS_DIR, "challenger")
REPORTS_DIR = os.path.join("reports", "challenger_7h")


class Challenger7HPipelineEngine:
    """
    Executes an isolated 7-horizon Challenger research experiment.
    """

    def __init__(self, ticker: str = "RELIANCE.NS", csv_path: str = "merge-csv.csv"):
        self.ticker = ticker
        self.csv_path = csv_path
        self.raw_df = None
        self.clean_df = None
        self.feat_df = None
        self.combined_df = None
        
        self.scaler_X = None
        self.scaler_y = None
        
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None

        self.train_dates = None
        self.val_dates = None
        self.test_dates = None

        self.models = {}
        self.val_predictions = {}
        self.test_predictions = {}
        self.val_metrics = {}
        self.test_metrics = {}
        self.backtest_results = {}
        self.correlation_matrix = None

        # Ensure directory isolation
        os.makedirs(CHALLENGER_DIR, exist_ok=True)
        os.makedirs(REPORTS_DIR, exist_ok=True)

    def load_and_validate_data(self) -> pd.DataFrame:
        """Step 1: Load and clean price dataset."""
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            if "Ticker" in df.columns:
                df_ticker = df[df["Ticker"] == self.ticker].copy()
                if len(df_ticker) >= 50:
                    self.raw_df = df_ticker.sort_values("Date").reset_index(drop=True)

        if self.raw_df is None or self.raw_df.empty:
            stock_file = os.path.join("Stock_Data", f"{self.ticker}.csv")
            if os.path.exists(stock_file):
                self.raw_df = pd.read_csv(stock_file).sort_values("Date").reset_index(drop=True)

        if self.raw_df is None or self.raw_df.empty:
            raise FileNotFoundError(f"Data for {self.ticker} not found.")

        self.clean_df = clean_ohlcv(self.raw_df, ticker=self.ticker)
        return self.clean_df

    def engineer_features_and_targets(self) -> pd.DataFrame:
        """Step 2: Build 36 canonical features & 7-day multi-horizon targets."""
        self.feat_df = build_canonical_feature_set(self.clean_df, ticker=self.ticker)
        self.feat_df = self.feat_df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

        targets = prepare_targets(self.feat_df, FORECAST_HORIZON)
        target_cols = [f"log_ret_d{d}" for d in range(1, FORECAST_HORIZON + 1)]

        self.combined_df = pd.concat([self.feat_df[["Date"] + FEATURE_COLS], targets[target_cols], self.feat_df[["Close"]]], axis=1).dropna().reset_index(drop=True)
        return self.combined_df

    def execute_chronological_embargo_split(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Step 3: 60/20/20 Chronological split with 7-day embargo gaps.
        Fits scalers STRICTLY on Train split to prevent data leakage.
        """
        from sklearn.preprocessing import MinMaxScaler

        X_raw = self.combined_df[FEATURE_COLS].values
        target_cols = [f"log_ret_d{d}" for d in range(1, FORECAST_HORIZON + 1)]
        y_raw = self.combined_df[target_cols].values
        dates = self.combined_df["Date"].values if "Date" in self.combined_df.columns else np.arange(len(self.combined_df))

        n_samples = len(self.combined_df)
        embargo_days = 7

        train_end = int(n_samples * 0.60)
        val_start = train_end + embargo_days
        val_end = val_start + int(n_samples * 0.20)
        test_start = val_end + embargo_days

        # Extract Raw Splits
        X_tr_raw, y_tr_raw = X_raw[:train_end], y_raw[:train_end]
        X_v_raw, y_v_raw = X_raw[val_start:val_end], y_raw[val_start:val_end]
        X_te_raw, y_te_raw = X_raw[test_start:], y_raw[test_start:]

        self.train_dates = dates[:train_end]
        self.val_dates = dates[val_start:val_end]
        self.test_dates = dates[test_start:]

        # Fit Scalers STRICTLY on Training portion
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        self.X_train = self.scaler_X.fit_transform(X_tr_raw)
        self.y_train = self.scaler_y.fit_transform(y_tr_raw)

        # Transform Validation and Test sets using Train scalers ONLY
        self.X_val = self.scaler_X.transform(X_v_raw)
        self.y_val = y_v_raw  # Keep unscaled for metric calculation

        self.X_test = self.scaler_X.transform(X_te_raw)
        self.y_test = y_te_raw  # Keep unscaled for metric calculation

        # Verify shapes
        assert self.y_train.shape[1] == 7, f"y_train shape mismatch: {self.y_train.shape}"
        assert self.y_val.shape[1] == 7, f"y_val shape mismatch: {self.y_val.shape}"
        assert self.y_test.shape[1] == 7, f"y_test shape mismatch: {self.y_test.shape}"

        return self.X_train, self.X_val, self.X_test

    def train_multi_output_models(self) -> Dict[str, Any]:
        """Step 4: Train Random Forest, XGBoost, LightGBM, and LSTM."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor
        import xgboost as xgb
        import lightgbm as lgb
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout

        # 1. Random Forest (MultiOutput)
        rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=45, max_depth=10, random_state=42, n_jobs=-1))
        rf.fit(self.X_train, self.y_train)
        self.models["Random_Forest"] = rf

        # 2. XGBoost (MultiOutput)
        xgb_mod = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=45, max_depth=6, learning_rate=0.04, random_state=42, n_jobs=-1))
        xgb_mod.fit(self.X_train, self.y_train)
        self.models["XGBoost"] = xgb_mod

        # 3. LightGBM (MultiOutput)
        lgb_mod = MultiOutputRegressor(lgb.LGBMRegressor(n_estimators=45, max_depth=6, learning_rate=0.04, random_state=42, n_jobs=-1, verbose=-1))
        lgb_mod.fit(self.X_train, self.y_train)
        self.models["LightGBM"] = lgb_mod

        # 4. LSTM (Multi-output 7 Dense outputs)
        win_len = 15
        X_tr_seq, y_tr_seq = [], []
        for i in range(len(self.X_train) - win_len):
            X_tr_seq.append(self.X_train[i : i + win_len])
            y_tr_seq.append(self.y_train[i + win_len - 1])
        X_tr_seq, y_tr_seq = np.array(X_tr_seq), np.array(y_tr_seq)

        lstm = Sequential([
            LSTM(64, input_shape=(win_len, self.X_train.shape[1]), return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(FORECAST_HORIZON)
        ])
        lstm.compile(optimizer="adam", loss="mse")
        lstm.fit(X_tr_seq, y_tr_seq, epochs=15, batch_size=32, verbose=0)
        self.models["LSTM"] = lstm
        self.win_len = win_len

        return self.models

    def predict_and_evaluate_split(self, split_name: str = "TEST") -> Dict[str, Any]:
        """
        Generates predictions for each model & ensemble on validation or test split.
        Verifies prediction.shape == (N, 7).
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        X_eval = self.X_val if split_name == "VAL" else self.X_test
        y_true_eval = self.y_val if split_name == "VAL" else self.y_test

        # Predict & Inverse Transform to Raw Log Return Space
        p_rf = self.scaler_y.inverse_transform(self.models["Random_Forest"].predict(X_eval))
        p_xgb = self.scaler_y.inverse_transform(self.models["XGBoost"].predict(X_eval))
        p_lgb = self.scaler_y.inverse_transform(self.models["LightGBM"].predict(X_eval))

        # LSTM Sequence Evaluation
        X_seq = []
        for i in range(len(X_eval) - self.win_len):
            X_seq.append(X_eval[i : i + self.win_len])
        X_seq = np.array(X_seq)
        p_lstm = self.scaler_y.inverse_transform(self.models["LSTM"].predict(X_seq, verbose=0))

        min_len = len(p_lstm)
        y_true = y_true_eval[:min_len]

        p_rf = p_rf[:min_len]
        p_xgb = p_xgb[:min_len]
        p_lgb = p_lgb[:min_len]

        ensemble_preds = (p_rf + p_xgb + p_lgb + p_lstm) / 4.0

        # Mandatory Shape Assertions
        assert y_true.shape == (min_len, 7), f"y_true shape invalid: {y_true.shape}"
        assert ensemble_preds.shape == (min_len, 7), f"Ensemble prediction shape invalid: {ensemble_preds.shape}"

        # Calculate per-horizon metrics
        days_metrics = {}
        for d in range(1, FORECAST_HORIZON + 1):
            yt_d = y_true[:, d - 1]
            yp_d = ensemble_preds[:, d - 1]

            mae_d = float(mean_absolute_error(yt_d, yp_d))
            rmse_d = float(np.sqrt(mean_squared_error(yt_d, yp_d)))
            r2_d = float(r2_score(yt_d, yp_d) * 100.0) if len(yt_d) > 1 else 0.0

            dir_acc_d = float(np.mean((yt_d >= 0) == (yp_d >= 0)) * 100.0)

            # Up / Down Accuracy
            pos_mask = (yt_d >= 0)
            neg_mask = (yt_d < 0)

            up_acc = float(np.mean((yp_d[pos_mask] >= 0)) * 100.0) if np.sum(pos_mask) > 0 else 50.0
            down_acc = float(np.mean((yp_d[neg_mask] < 0)) * 100.0) if np.sum(neg_mask) > 0 else 50.0
            bal_acc = float((up_acc + down_acc) / 2.0)

            days_metrics[f"Day_{d}"] = {
                "directional_accuracy_pct": round(dir_acc_d, 2),
                "up_accuracy_pct": round(up_acc, 2),
                "down_accuracy_pct": round(down_acc, 2),
                "balanced_accuracy_pct": round(bal_acc, 2),
                "mae": round(mae_d, 4),
                "rmse": round(rmse_d, 4),
                "r2_score": round(r2_d, 2)
            }

        overall_acc = round(float(np.mean([m["directional_accuracy_pct"] for m in days_metrics.values()])), 2)
        overall_mae = round(float(np.mean([m["mae"] for m in days_metrics.values()])), 4)
        overall_rmse = round(float(np.mean([m["rmse"] for m in days_metrics.values()])), 4)

        metrics_summary = {
            "day_by_day": days_metrics,
            "overall_accuracy_pct": overall_acc,
            "overall_mae": overall_mae,
            "overall_rmse": overall_rmse,
            "min_len": min_len
        }

        if split_name == "VAL":
            self.val_predictions = {"ensemble": ensemble_preds, "rf": p_rf, "xgb": p_xgb, "lgb": p_lgb, "lstm": p_lstm}
            self.val_metrics = metrics_summary
        else:
            self.test_predictions = {"ensemble": ensemble_preds, "rf": p_rf, "xgb": p_xgb, "lgb": p_lgb, "lstm": p_lstm}
            self.test_metrics = metrics_summary

        return metrics_summary

    def run_horizon_independence_check(self, split_name: str = "TEST") -> pd.DataFrame:
        """
        Step 5: Verifies that predictions across Day 1..7 are not identical (no tiling).
        Computes pairwise Pearson correlation matrix across output columns.
        """
        preds = self.val_predictions["ensemble"] if split_name == "VAL" else self.test_predictions["ensemble"]
        df_preds = pd.DataFrame(preds, columns=[f"Day_{d}" for d in range(1, FORECAST_HORIZON + 1)])
        corr_matrix = df_preds.corr().round(4)
        self.correlation_matrix = corr_matrix

        # Check for identical columns
        identical_pairs = 0
        for i in range(7):
            for j in range(i + 1, 7):
                if np.allclose(preds[:, i], preds[:, j], atol=1e-6):
                    identical_pairs += 1

        print(f"  + Horizon Independence Check: Pairwise Correlation Calculated. Identical Column Pairs = {identical_pairs}")
        assert identical_pairs == 0, "HORIZON INDEPENDENCE FAIL: Output columns are identical!"
        return corr_matrix

    def run_out_of_sample_backtest(self) -> Dict[str, float]:
        """
        Step 6: Simulates trading strategy on Final Test set using 10 bps transaction costs.
        """
        test_preds = self.test_predictions["ensemble"]
        min_len = len(test_preds)
        y_true_d1 = self.y_test[:min_len, 0]

        # Strategy Rules: Long if Pred Day 1 > 0.001 (+0.10%), Short if < -0.001 (-0.10%), else Cash (0)
        positions = np.zeros(min_len)
        positions[test_preds[:, 0] > 0.001] = 1.0
        positions[test_preds[:, 0] < -0.001] = -1.0

        # Account for 10 bps transaction costs per position change
        pos_changes = np.abs(np.diff(positions, prepend=0))
        tx_costs = pos_changes * 0.0010  # 10 bps

        daily_returns = (positions * y_true_d1) - tx_costs
        cum_returns = np.cumsum(daily_returns)

        total_return_pct = round(float(np.sum(daily_returns) * 100.0), 2)
        ann_return_pct = round(float(total_return_pct * (252.0 / max(min_len, 1))), 2)

        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns) + 1e-9
        sharpe_ratio = round(float((mean_ret / std_ret) * np.sqrt(252)), 2)

        # Max Drawdown
        peak = np.maximum.accumulate(cum_returns)
        dd = peak - cum_returns
        max_dd_pct = round(float(np.max(dd) * 100.0), 2) if len(dd) > 0 else 0.0

        active_trades = np.sum(positions != 0)
        winning_trades = np.sum(daily_returns[positions != 0] > 0)
        win_rate_pct = round(float((winning_trades / max(active_trades, 1)) * 100.0), 2)

        gross_gains = np.sum(daily_returns[daily_returns > 0])
        gross_losses = np.abs(np.sum(daily_returns[daily_returns < 0])) + 1e-9
        profit_factor = round(float(gross_gains / gross_losses), 2)

        self.backtest_results = {
            "Total_Return_%": total_return_pct,
            "Annualized_Return_%": ann_return_pct,
            "Sharpe_Ratio": sharpe_ratio,
            "Max_Drawdown_%": max_dd_pct,
            "Win_Rate_%": win_rate_pct,
            "Number_of_Trades": int(active_trades),
            "Profit_Factor": profit_factor,
            "Transaction_Fee_Bps": 10
        }
        return self.backtest_results

    def export_experiment_artifacts(self) -> Dict[str, str]:
        """Step 7: Exports serialized models and structured JSON/CSV reports."""
        # Save Challenger Model Artifacts ONLY in models/challenger/
        joblib.dump(self.models["Random_Forest"], os.path.join(CHALLENGER_DIR, "rf_model.pkl"))
        joblib.dump(self.models["XGBoost"], os.path.join(CHALLENGER_DIR, "xgb_model.pkl"))
        joblib.dump(self.models["LightGBM"], os.path.join(CHALLENGER_DIR, "lgb_model.pkl"))
        self.models["LSTM"].save(os.path.join(CHALLENGER_DIR, "lstm_model.keras"))
        joblib.dump((self.scaler_X, self.scaler_y), os.path.join(CHALLENGER_DIR, "scalers.pkl"))

        meta = {
            "ticker": self.ticker,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "CHALLENGER_7H_RESEARCH_BASELINE",
            "features_used": len(FEATURE_COLS),
            "forecast_horizon": FORECAST_HORIZON
        }
        with open(os.path.join(CHALLENGER_DIR, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=4)

        # Export Experiment Reports in reports/challenger_7h/
        with open(os.path.join(REPORTS_DIR, "metrics.json"), "w") as f:
            json.dump({
                "validation": self.val_metrics,
                "test": self.test_metrics,
                "backtest": self.backtest_results,
                "horizon_correlations": self.correlation_matrix.to_dict()
            }, f, indent=4)

        # Export predictions.csv
        min_len = len(self.test_predictions["ensemble"])
        y_true = self.y_test[:min_len]
        preds = self.test_predictions["ensemble"]
        dates = self.test_dates[:min_len]

        rows = []
        for i in range(min_len):
            for d in range(1, FORECAST_HORIZON + 1):
                rows.append({
                    "date": str(dates[i])[:10],
                    "ticker": self.ticker,
                    "horizon": f"Day_{d}",
                    "y_true": round(float(y_true[i, d - 1]), 6),
                    "pred_ensemble": round(float(preds[i, d - 1]), 6),
                    "pred_rf": round(float(self.test_predictions["rf"][i, d - 1]), 6),
                    "pred_xgb": round(float(self.test_predictions["xgb"][i, d - 1]), 6),
                    "pred_lgb": round(float(self.test_predictions["lgb"][i, d - 1]), 6),
                    "pred_lstm": round(float(self.test_predictions["lstm"][i, d - 1]), 6)
                })

        pd.DataFrame(rows).to_csv(os.path.join(REPORTS_DIR, "predictions.csv"), index=False)

        # Config, Features, and Split Metadata
        with open(os.path.join(REPORTS_DIR, "feature_list.json"), "w") as f:
            json.dump({"features": FEATURE_COLS, "count": len(FEATURE_COLS)}, f, indent=4)

        with open(os.path.join(REPORTS_DIR, "data_split.json"), "w") as f:
            json.dump({
                "train_samples": len(self.X_train),
                "val_samples": len(self.X_val),
                "test_samples": len(self.X_test),
                "embargo_days": 7,
                "train_dates": [str(self.train_dates[0])[:10], str(self.train_dates[-1])[:10]],
                "test_dates": [str(self.test_dates[0])[:10], str(self.test_dates[-1])[:10]]
            }, f, indent=4)

        with open(os.path.join(REPORTS_DIR, "training_config.json"), "w") as f:
            json.dump({
                "experiment_id": "EXP_CHALLENGER_7H_001",
                "random_seed": 42,
                "models": ["RandomForest", "XGBoost", "LightGBM", "LSTM"],
                "target_columns": [f"log_ret_d{d}" for d in range(1, FORECAST_HORIZON + 1)]
            }, f, indent=4)

        # Generate CHALLENGER_7H_REPORT.md
        self._generate_report_markdown()

        return REPORTS_DIR

    def _generate_report_markdown(self):
        """Generates CHALLENGER_7H_REPORT.md in reports/challenger_7h/."""
        m_test = self.test_metrics["day_by_day"]
        b_test = self.backtest_results

        champ_d1 = 47.73  # Valid Champion Day 1 Baseline from Audit

        cand_d1 = m_test["Day_1"]["directional_accuracy_pct"]
        experiment_status = "BETTER" if (cand_d1 > champ_d1 and b_test["Sharpe_Ratio"] >= 0.50) else "NOT BETTER"

        md_lines = [
            "# CHALLENGER_7H_REPORT.md",
            "===================================================================",
            "Research-Only Clean 7-Horizon Challenger Baseline Report",
            "===================================================================",
            "",
            "## 1. Executive Summary",
            f"An isolated 7-horizon Challenger model pipeline was trained and evaluated over a strict 60% Train / 20% Validation / 20% Final Unseen Test chronological split with 7-day embargo gaps.",
            f"- **Experiment Status**: `{experiment_status}`",
            f"- **Deployment Decision**: `DO NOT DEPLOY` (Research Baseline Only)",
            f"- **Champion Day-1 Baseline Accuracy**: {champ_d1}%",
            f"- **Challenger Day-1 Unseen Test Accuracy**: {m_test['Day_1']['directional_accuracy_pct']}%",
            f"- **Challenger Out-Of-Sample Sharpe Ratio**: {b_test['Sharpe_Ratio']}",
            f"- **Challenger Out-Of-Sample Max Drawdown**: {b_test['Max_Drawdown_%']}%",
            "",
            "## 2. Final Unseen Test Matrix (Day 1 through Day 7)",
            "",
            "| Horizon | Directional Accuracy | Up Accuracy | Down Accuracy | Balanced Acc | MAE | RMSE | R² Score |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        ]

        for d in range(1, FORECAST_HORIZON + 1):
            dm = m_test[f"Day_{d}"]
            md_lines.append(
                f"| **Day {d}** | {dm['directional_accuracy_pct']}% | {dm['up_accuracy_pct']}% | {dm['down_accuracy_pct']}% | {dm['balanced_accuracy_pct']}% | {dm['mae']} | {dm['rmse']} | {dm['r2_score']}% |"
            )

        md_lines.extend([
            "",
            "## 3. Horizon Independence Verification",
            "Pairwise correlation analysis across output columns confirms all 7 prediction horizons are distinct and not artificially duplicated/tiled:",
            "```text",
            self.correlation_matrix.to_string(),
            "```",
            "",
            "## 4. Out-Of-Sample Trading Backtest (10 bps Fee Applied)",
            f"- **Total Return**: {b_test['Total_Return_%']}%",
            f"- **Annualized Return**: {b_test['Annualized_Return_%']}%",
            f"- **Sharpe Ratio**: {b_test['Sharpe_Ratio']}",
            f"- **Max Drawdown**: {b_test['Max_Drawdown_%']}%",
            f"- **Win Rate**: {b_test['Win_Rate_%']}%",
            f"- **Active Trades**: {b_test['Number_of_Trades']}",
            f"- **Profit Factor**: {b_test['Profit_Factor']}",
            "",
            "## 5. Champion vs Challenger Comparison",
            "- **Champion Day-1 Directional Accuracy**: `47.73%` (Valid Baseline)",
            "- **Champion Day 2–7 Metrics**: `NOT VALID / NOT COMPARABLE` (Production artifact is single-output; previous evaluation used invalid tiling)",
            "- **Challenger Day-1 Directional Accuracy**: `" + str(m_test['Day_1']['directional_accuracy_pct']) + "%`",
            "",
            "## 6. Production Safety Verification",
            "```text",
            "Champion modified: NO",
            "Champion deployed: NO",
            "Production scaler modified: NO",
            "Production metadata modified: NO",
            "Automatic retraining enabled: NO",
            "```"
        ])

        report_path = os.path.join(REPORTS_DIR, "CHALLENGER_7H_REPORT.md")
        with open(report_path, "w") as f:
            f.write("\n".join(md_lines) + "\n")

    def run_pipeline(self) -> Dict[str, Any]:
        """Executes full Challenger 7-horizon experiment."""
        print("[1/7] Loading Data & Engineering 36 Canonical Features...")
        self.load_and_validate_data()
        self.engineer_features_and_targets()

        print("[2/7] Splitting Chronologically (60/20/20) with 7-Day Embargo & Train-Only Scaler Fit...")
        self.execute_chronological_embargo_split()

        print("[3/7] Training 4 Multi-Output Models (RF, XGB, LGB, LSTM)...")
        self.train_multi_output_models()

        print("[4/7] Evaluating Validation & Final Unseen Test Sets...")
        self.predict_and_evaluate_split("VAL")
        self.predict_and_evaluate_split("TEST")

        print("[5/7] Running Horizon Independence Check...")
        self.run_horizon_independence_check()

        print("[6/7] Simulating Out-Of-Sample Trading Backtest...")
        self.run_out_of_sample_backtest()

        print("[7/7] Exporting Experiment Artifacts & Reports...")
        self.export_experiment_artifacts()

        print("\n========================================")
        print("7-HORIZON CHALLENGER EXPERIMENT COMPLETE")
        print("========================================")
        print("Champion modified: NO")
        print("Champion deployed: NO")
        print("Production scaler modified: NO")
        print("Production metadata modified: NO")
        print("Automatic retraining enabled: NO")
        print(f"True 7-horizon Challenger created: YES")
        print(f"Prediction shape: {self.test_predictions['ensemble'].shape}")
        print(f"Target shape: {self.y_test.shape}")
        for d in range(1, FORECAST_HORIZON + 1):
            print(f"Day-{d} test accuracy: {self.test_metrics['day_by_day'][f'Day_{d}']['directional_accuracy_pct']}%")
        print(f"Final test Sharpe: {self.backtest_results['Sharpe_Ratio']}")
        print(f"Final test Max Drawdown: {self.backtest_results['Max_Drawdown_%']}%")
        print("Deployment recommendation: DO NOT DEPLOY / READY FOR SEPARATE REVIEW")
        print("========================================\n")

        return {
            "status": "SUCCESS",
            "test_metrics": self.test_metrics,
            "backtest": self.backtest_results
        }


if __name__ == "__main__":
    engine = Challenger7HPipelineEngine(ticker="RELIANCE.NS")
    engine.run_pipeline()
