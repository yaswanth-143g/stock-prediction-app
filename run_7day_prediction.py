#!/usr/bin/env python3
"""
run_7day_prediction.py — Enterprise 7-Day Stock Price Prediction CLI & Engine
==============================================================================
Runs 7-day multi-horizon stock price trajectory forecasts for any valid stock ticker.
Uses 36 canonical technical features, pre-loaded multi-model ensemble (RF, XGBoost, LSTM, LightGBM),
and fast RAM caching for sub-50ms inference latency.

Usage:
  python run_7day_prediction.py --ticker TATAMOTORS.NS
  python run_7day_prediction.py --ticker RELIANCE.NS --json
  python run_7day_prediction.py --batch AAPL,RELIANCE.NS,INFY.NS
"""

import sys
import os
import argparse
import json
from datetime import datetime

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure sys.stdout for UTF-8 encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from predict import predict_7days, get_model_manager, get_nse_market_status


def print_prediction_report(res: dict):
    """Prints a beautiful formatted ASCII report for 7-day stock price prediction."""
    ticker = res.get("final_ticker", res.get("ticker", "UNKNOWN"))
    curr_price = res.get("current_price", 0.0)
    pred_prices = res.get("predicted_prices", [])
    pred_dates = res.get("prediction_dates", [])
    upper_bounds = res.get("upper_bounds", [])
    lower_bounds = res.get("lower_bounds", [])
    exp_return = res.get("expected_return_pct", 0.0)
    signal = res.get("signal", "HOLD")
    stop_loss = res.get("stop_loss", 0.0)
    take_profit = res.get("take_profit", 0.0)
    rr_ratio = res.get("risk_reward_ratio", 0.0)
    prob_rise = res.get("prob_rise_pct", 50.0)
    prob_fall = res.get("prob_fall_pct", 50.0)
    latency = res.get("inference_latency_ms", 0.0)
    source = res.get("source", "model_inference")
    model_name = res.get("model_name", "Multi-Model Ensemble")
    
    print("\n" + "=" * 76)
    print(f"       🚀 7-DAY FUTURE STOCK PRICE PREDICTION REPORT: {ticker}")
    print("=" * 76)
    print(f" Current Price       : ₹{curr_price:,.2f}")
    if pred_prices:
        target_price = pred_prices[-1]
        price_diff = target_price - curr_price
        print(f" 7-Day Target Price  : ₹{target_price:,.2f} ({exp_return:+.2f}% / ₹{price_diff:+.2f})")
    print(f" Recommendation      : {signal}")
    print(f" Probability Rise/Fall: {prob_rise}% Rise | {prob_fall}% Fall")
    print(f" Risk / Reward Ratio : {rr_ratio}:1  | Stop-Loss: ₹{stop_loss:,.2f} | Take-Profit: ₹{take_profit:,.2f}")
    print(f" Model Engine        : {model_name}")
    print(f" Inference Latency   : {latency} ms (Source: {source})")
    
    if res.get("is_replaced"):
        print(f" Note                : {res.get('replacement_message')}")
        
    print("-" * 76)
    print(" DAY-BY-DAY 7-DAY PRICE TRAJECTORY FORECAST")
    print("-" * 76)
    print(f" {'Day':<6} | {'Date':<12} | {'Forecast Price':<15} | {'Change %':<10} | {'Lower Bound':<13} | {'Upper Bound':<13}")
    print("-" * 76)

    prev_p = curr_price
    for i in range(len(pred_prices)):
        d_num = f"Day {i+1}"
        d_date = pred_dates[i] if i < len(pred_dates) else f"T+{i+1}"
        p_val = pred_prices[i]
        chg_pct = ((p_val - prev_p) / prev_p) * 100.0
        l_bound = lower_bounds[i] if i < len(upper_bounds) else p_val * 0.98
        u_bound = upper_bounds[i] if i < len(upper_bounds) else p_val * 1.02
        
        print(f" {d_num:<6} | {d_date:<12} | ₹{p_val:<14,.2f} | {chg_pct:+8.2f}% | ₹{l_bound:<12,.2f} | ₹{u_bound:<12,.2f}")
        prev_p = p_val

    print("-" * 76)
    
    # Model Breakdown Table
    breakdown = res.get("model_breakdown", {})
    if breakdown and len(breakdown.get("random_forest", [])) == len(pred_prices):
        print(" ENSEMBLE MODEL BREAKDOWN (7th DAY PRICE ESTIMATES)")
        print("-" * 76)
        rf_7d = breakdown["random_forest"][-1]
        xgb_7d = breakdown["xgboost"][-1]
        lstm_7d = breakdown["lstm"][-1]
        lgb_7d = breakdown["lightgbm"][-1]
        weights = res.get("ensemble_weights") or {}
        
        print(f"  Random Forest (Weight: {weights.get('random_forest', 0.35):.2f}) : ₹{rf_7d:,.2f}")
        print(f"  XGBoost       (Weight: {weights.get('xgboost', 0.25):.2f}) : ₹{xgb_7d:,.2f}")
        print(f"  LSTM Net      (Weight: {weights.get('lstm', 0.25):.2f}) : ₹{lstm_7d:,.2f}")
        print(f"  LightGBM      (Weight: {weights.get('lightgbm', 0.15):.2f}) : ₹{lgb_7d:,.2f}")
        print("-" * 76)

    # Fundamental & News Sentiment
    fundamentals = res.get("fundamentals") or {}
    sentiment = res.get("news_sentiment") or {}
    if fundamentals:
        pe = fundamentals.get("pe_ratio", "N/A")
        pb = fundamentals.get("pb_ratio", "N/A")
        roe = fundamentals.get("roe", "N/A")
        print(f" FUNDAMENTALS  : P/E: {pe} | P/B: {pb} | ROE: {roe}%")
    if sentiment:
        score = sentiment.get("sentiment_score", 0.0)
        label = sentiment.get("sentiment_label", "NEUTRAL")
        print(f" NEWS SENTIMENT: {label} (Score: {score:+.2f})")
    
    print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="7-Day Future Stock Price Prediction CLI Engine")
    parser.add_argument("--ticker", "-t", type=str, default="RELIANCE.NS", help="Stock Ticker Symbol (e.g. RELIANCE.NS, TATAMOTORS.NS, AAPL)")
    parser.add_argument("--batch", "-b", type=str, default=None, help="Comma-separated tickers for batch prediction (e.g. RELIANCE.NS,TATAMOTORS.NS,INFY.NS)")
    parser.add_argument("--json", "-j", action="store_true", help="Output raw JSON response payload")
    parser.add_argument("--output", "-o", type=str, default=None, help="Save prediction report to JSON file path")
    args = parser.parse_args()

    # Preload models
    get_model_manager()

    if args.batch:
        tickers = [t.strip() for t in args.batch.split(",") if t.strip()]
        results = []
        for tk in tickers:
            try:
                res = predict_7days(tk)
                results.append(res)
                if not args.json:
                    print_prediction_report(res)
            except Exception as e:
                print(f"❌ Error predicting for ticker {tk}: {e}", file=sys.stderr)
        
        if args.json:
            print(json.dumps(results, indent=2))
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"✅ Batch prediction results saved to {args.output}")

    else:
        try:
            res = predict_7days(args.ticker)
            if args.json:
                print(json.dumps(res, indent=2))
            else:
                print_prediction_report(res)

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(res, f, indent=2)
                print(f"✅ Prediction report saved to {args.output}")

        except Exception as e:
            print(f"❌ Error predicting for ticker {args.ticker}: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
