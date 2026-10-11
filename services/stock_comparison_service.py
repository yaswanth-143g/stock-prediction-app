"""
services/stock_comparison_service.py — Stock Comparison & Financial Analytics Engine
=======================================================================================
Provides modular historical data retrieval, trading-date alignment, financial risk
calculations (returns, volatility, drawdown, base-100 normalization), and optional
AI model forecast comparison for multi-stock evaluation.
"""

import math
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger("StockComparisonService")
logger.setLevel(logging.INFO)

# Supported historical comparison periods mapped to max days
PERIOD_MAP = {
    "1m": 30,
    "3m": 90,
    "6m": 180,
    "1y": 365,
    "2y": 730
}
MAX_PERIOD_DAYS = 730  # Hard 2-year cap as required by specifications
MIN_STOCKS = 2
MAX_STOCKS = 5


def validate_comparison_inputs(symbols: List[str], period: str = "1y") -> Tuple[bool, str, List[str], str]:
    """
    Validates input parameters for stock comparison.
    Rules:
    - Minimum 2, maximum 5 stocks.
    - No empty/whitespace symbols.
    - Rejects duplicate symbols.
    - Validates historical date range (max 2 years).
    
    Returns (is_valid, error_message, cleaned_symbols, validated_period).
    """
    if not symbols or not isinstance(symbols, list):
        return False, "Please select at least 2 stocks for comparison.", [], period

    cleaned_symbols = []
    seen = set()

    for s in symbols:
        if not s or not isinstance(s, str):
            continue
        clean_s = s.strip().upper()
        if clean_s.endswith("."):
            clean_s = clean_s.rstrip(".")
        if clean_s and clean_s not in seen:
            seen.add(clean_s)
            cleaned_symbols.append(clean_s)

    if len(cleaned_symbols) < MIN_STOCKS:
        return False, f"Comparison requires at least {MIN_STOCKS} distinct stocks. You selected {len(cleaned_symbols)}.", cleaned_symbols, period

    if len(cleaned_symbols) > MAX_STOCKS:
        return False, f"Comparison allows a maximum of {MAX_STOCKS} stocks. You selected {len(cleaned_symbols)}.", cleaned_symbols[:MAX_STOCKS], period

    clean_period = str(period).strip().lower()
    if clean_period not in PERIOD_MAP:
        clean_period = "1y"

    return True, "", cleaned_symbols, clean_period


def get_stock_historical_series(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Retrieves historical OHLCV data for a ticker using project's MarketDataService or local DB.
    Guarantees 'Date' (formatted YYYY-MM-DD) and 'Close' columns ordered chronologically.
    """
    try:
        from services.market_data_service import get_market_data_service
        market_svc = get_market_data_service()
        df = market_svc.get_historical_data(ticker, period=period)
    except Exception as e:
        logger.warning(f"MarketDataService failed for {ticker}: {e}")
        df = pd.DataFrame()

    if df is None or df.empty:
        # Fallback to local CSV files if available
        import os
        csv_path = os.path.join("Stock_Data", f"{ticker}.csv")
        if not os.path.exists(csv_path):
            clean_name = ticker.replace(".NS", "").replace(".BO", "")
            csv_path = os.path.join("Stock_Data", f"{clean_name}.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                df = pd.DataFrame()

    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame(columns=["Date", "Close", "Volume"])

    # Standardize DataFrame columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df_clean = df.dropna(subset=["Close"]).copy()
    if "Date" not in df_clean.columns:
        if "index" in df_clean.columns:
            df_clean.rename(columns={"index": "Date"}, inplace=True)
        elif df_clean.index.name in ["Date", "Datetime"]:
            df_clean = df_clean.reset_index()

    if "Date" not in df_clean.columns:
        return pd.DataFrame(columns=["Date", "Close", "Volume"])

    df_clean["Date"] = pd.to_datetime(df_clean["Date"]).dt.strftime("%Y-%m-%d")
    df_clean["Close"] = pd.to_numeric(df_clean["Close"], errors="coerce")
    df_clean = df_clean.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

    # Filter by period length cap
    max_days = PERIOD_MAP.get(period, 365)
    cutoff_date = (datetime.now() - timedelta(days=max_days)).strftime("%Y-%m-%d")
    df_clean = df_clean[df_clean["Date"] >= cutoff_date].reset_index(drop=True)

    return df_clean[["Date", "Close"] + (["Volume"] if "Volume" in df_clean.columns else [])]


def align_stock_histories(stock_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Aligns multiple stock histories by trading date for cross-stock comparisons.
    Uses inner date intersection for multi-stock returns/volatility calculations.
    
    Returns:
    - aligned_df: DataFrame with 'Date' column and a column per stock symbol containing Close price.
    - metadata: Metadata dict including date range, common date count, and warning messages.
    """
    if not stock_data:
        return pd.DataFrame(), {"warning": "No stock data provided for alignment.", "common_dates_count": 0}

    valid_series = {}
    for ticker, df in stock_data.items():
        if df is not None and not df.empty and "Date" in df.columns and "Close" in df.columns:
            # Drop duplicates by Date keeping the latest
            s = df.drop_duplicates(subset=["Date"]).set_index("Date")["Close"]
            valid_series[ticker] = s

    if not valid_series:
        return pd.DataFrame(), {"warning": "Historical data is unavailable for all selected stocks.", "common_dates_count": 0}

    # Inner join on Date index to find exact overlapping trading dates
    aligned_df = pd.DataFrame(valid_series).dropna(how="any").reset_index()
    aligned_df = aligned_df.sort_values("Date").reset_index(drop=True)

    common_count = len(aligned_df)
    warning_msg = None

    if common_count < 5:
        warning_msg = f"Insufficient overlapping trading history ({common_count} common trading days found). Try selecting a broader or different historical date range."

    meta = {
        "common_dates_count": common_count,
        "start_date": aligned_df["Date"].iloc[0] if common_count > 0 else None,
        "end_date": aligned_df["Date"].iloc[-1] if common_count > 0 else None,
        "warning": warning_msg
    }

    return aligned_df, meta


def compute_stock_metrics(prices: pd.Series, dates: pd.Series) -> Dict[str, Any]:
    """
    Computes precise financial metrics for a single stock price series:
    1. Absolute Price Change ($P_{end} - P_{start}$)
    2. Percentage Return ($\frac{P_{end}}{P_{start}} - 1 \times 100$)
    3. Daily Returns & Annualized Volatility ($\sigma_{daily} \times \sqrt{252}$)
    4. Maximum Drawdown ($\min(\frac{P_t}{\text{Running Peak}} - 1) \times 100$)
    5. Normalized Performance (Base 100 trajectory)
    6. Latest Closing Price and Date.
    """
    clean_p = pd.to_numeric(prices, errors="coerce").dropna()
    if len(clean_p) < 2:
        start_price = float(clean_p.iloc[0]) if len(clean_p) == 1 else 0.0
        return {
            "latest_price": start_price,
            "latest_date": str(dates.iloc[-1]) if len(dates) > 0 else "N/A",
            "starting_price": start_price,
            "starting_date": str(dates.iloc[0]) if len(dates) > 0 else "N/A",
            "absolute_change": 0.0,
            "percentage_return": 0.0,
            "annualized_volatility": 0.0,
            "max_drawdown": 0.0,
            "observation_count": len(clean_p),
            "normalized_series": [100.0] if len(clean_p) == 1 else []
        }

    start_p = float(clean_p.iloc[0])
    end_p = float(clean_p.iloc[-1])
    start_d = str(dates.iloc[0])
    end_d = str(dates.iloc[-1])

    # 1. Absolute Change
    abs_change = round(end_p - start_p, 2)

    # 2. Percentage Return
    pct_return = round(((end_p / start_p) - 1.0) * 100.0, 2) if start_p > 0 else 0.0

    # 3. Daily Returns & Annualized Volatility
    daily_returns = clean_p.pct_change().dropna()
    if len(daily_returns) > 1:
        daily_std = daily_returns.std()
        ann_vol = round(float(daily_std * np.sqrt(252) * 100.0), 2)
    else:
        ann_vol = 0.0

    # 4. Maximum Drawdown
    running_peak = clean_p.cummax()
    drawdowns = (clean_p / running_peak) - 1.0
    max_dd = round(float(drawdowns.min() * 100.0), 2)

    # 5. Base 100 Normalized Trajectory
    normalized_values = [round(float((p / start_p) * 100.0), 2) for p in clean_p]

    return {
        "latest_price": round(end_p, 2),
        "latest_date": end_d,
        "starting_price": round(start_p, 2),
        "starting_date": start_d,
        "absolute_change": abs_change,
        "percentage_return": pct_return,
        "annualized_volatility": ann_vol,
        "max_drawdown": max_dd,
        "observation_count": len(clean_p),
        "normalized_series": normalized_values
    }


def compare_stocks(symbols: List[str], period: str = "1y") -> Dict[str, Any]:
    """
    Main engine for stock comparison.
    1. Validates input symbols and period.
    2. Retrieves historical data for each stock.
    3. Aligns dates across stocks.
    4. Computes individual and cross-stock metrics.
    5. Formats standardized response payload for charts and tables.
    """
    is_valid, err_msg, clean_syms, clean_period = validate_comparison_inputs(symbols, period)
    if not is_valid:
        return {
            "status": "error",
            "message": err_msg,
            "period": period,
            "symbols": symbols
        }

    # Fetch individual stock histories
    raw_histories = {}
    for sym in clean_syms:
        raw_histories[sym] = get_stock_historical_series(sym, period=clean_period)

    # Align dates for comparison
    aligned_df, meta = align_stock_histories(raw_histories)
    if aligned_df.empty or meta.get("common_dates_count", 0) == 0:
        return {
            "status": "error",
            "message": meta.get("warning") or "Historical data is unavailable for the selected stocks and period. Please try another date range.",
            "period": clean_period,
            "symbols": clean_syms
        }

    dates = aligned_df["Date"].tolist()
    metrics_table = []
    charts_data = {
        "dates": dates,
        "normalized": {},
        "closing_prices": {}
    }

    # Distinct curated chart color palette for up to 5 stocks
    palette = ["#38BDF8", "#34D399", "#F59E0B", "#EF4444", "#8B5CF6"]

    for idx, sym in enumerate(clean_syms):
        if sym in aligned_df.columns:
            prices = aligned_df[sym]
            metrics = compute_stock_metrics(prices, aligned_df["Date"])
            color = palette[idx % len(palette)]

            metrics_table.append({
                "symbol": sym,
                "color": color,
                "latest_price": metrics["latest_price"],
                "latest_date": metrics["latest_date"],
                "starting_price": metrics["starting_price"],
                "starting_date": metrics["starting_date"],
                "absolute_change": metrics["absolute_change"],
                "percentage_return": metrics["percentage_return"],
                "annualized_volatility": metrics["annualized_volatility"],
                "max_drawdown": metrics["max_drawdown"],
                "observation_count": metrics["observation_count"]
            })

            charts_data["normalized"][sym] = {
                "color": color,
                "values": metrics["normalized_series"]
            }
            charts_data["closing_prices"][sym] = {
                "color": color,
                "values": [round(float(p), 2) for p in prices]
            }

    return {
        "status": "success",
        "period": clean_period,
        "comparison_count": len(metrics_table),
        "symbols": clean_syms,
        "metadata": meta,
        "metrics_table": metrics_table,
        "charts_data": charts_data,
        "disclaimer": "Historical stock comparison metrics are provided for educational and analytical purposes only and do not constitute financial advice or investment guarantees."
    }


def compare_predictions(symbols: List[str]) -> Dict[str, Any]:
    """
    Optional ML prediction comparison endpoint.
    Retrieves 7-day model predictions using the project's existing predict_stock engine.
    """
    is_valid, err_msg, clean_syms, _ = validate_comparison_inputs(symbols, "1y")
    if not is_valid and len(clean_syms) < 2:
        return {
            "status": "error",
            "message": err_msg,
            "symbols": symbols
        }

    predictions_data = []
    common_forecast_dates = []

    try:
        from predict import predict_stock
        for sym in clean_syms:
            try:
                res = predict_stock(sym)
                if res and res.get("status") == "success":
                    if not common_forecast_dates and res.get("prediction_dates"):
                        common_forecast_dates = res.get("prediction_dates")

                    curr_p = float(res.get("current_price", 0.0))
                    pred_prices = [float(p) for p in res.get("predicted_prices", [])]
                    exp_ret = float(res.get("expected_return_pct", 0.0) or res.get("expected_return", 0.0))

                    predictions_data.append({
                        "symbol": sym,
                        "current_price": round(curr_p, 2),
                        "predicted_next_day_price": pred_prices[0] if pred_prices else round(curr_p, 2),
                        "predicted_7d_price": pred_prices[-1] if pred_prices else round(curr_p, 2),
                        "expected_return_pct": round(exp_ret, 2),
                        "signal": res.get("signal", "HOLD"),
                        "confidence_score": res.get("confidence_score") or res.get("confidence_pct") or 75.0,
                        "forecast_dates": res.get("prediction_dates", []),
                        "predicted_prices": pred_prices
                    })
            except Exception as pe:
                logger.warning(f"Prediction comparison failed for {sym}: {pe}")
    except Exception as e:
        logger.error(f"Error executing compare_predictions: {e}")
        return {
            "status": "error",
            "message": f"ML model prediction comparison unavailable: {str(e)}",
            "symbols": clean_syms
        }

    return {
        "status": "success",
        "symbols": clean_syms,
        "forecast_dates": common_forecast_dates,
        "predictions": predictions_data,
        "note": "AI predictions represent algorithmic 7-day model trajectory forecasts based on technical pattern recognition."
    }
