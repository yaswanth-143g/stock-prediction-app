"""
quant/options_analytics.py
===========================
Live NIFTY / BANKNIFTY Options Analytics Engine.
Calculates Put-Call Ratio (PCR OI & Volume), Max Pain Strike,
Implied Volatility (IV) Skew, and Open Interest Support/Resistance Levels.
"""

import logging
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

INDEX_SYMBOL_MAP = {
    "NIFTY": "^NSEI",
    "NIFTY.NS": "^NSEI",
    "NIFTY50": "^NSEI",
    "^NSEI": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "BANKNIFTY.NS": "^NSEBANK",
    "^NSEBANK": "^NSEBANK"
}


def get_canonical_options_ticker(symbol: str) -> str:
    sym_upper = symbol.strip().upper()
    return INDEX_SYMBOL_MAP.get(sym_upper, sym_upper if sym_upper.endswith(".NS") else f"{sym_upper}.NS")


def analyze_options_chain(symbol: str = "NIFTY") -> Dict[str, Any]:
    """
    Fetches options chain and computes PCR (Volume & OI), Max Pain Strike,
    IV Skew, and Call/Put Open Interest distribution.
    """
    clean_sym = get_canonical_options_ticker(symbol)
    display_name = symbol.strip().upper()

    spot_price = 24500.0 if "NIFTY" in display_name else 52500.0
    options_chain_data = None
    exp_dates = []

    try:
        tk = yf.Ticker(clean_sym)
        exp_dates = tk.options
        if exp_dates:
            target_exp = exp_dates[0]
            opt = tk.option_chain(target_exp)
            calls_raw = opt.calls
            puts_raw = opt.puts

            # Fetch spot price
            hist = tk.history(period="1d")
            if not hist.empty and "Close" in hist.columns:
                spot_price = float(hist["Close"].iloc[-1])

            options_chain_data = (calls_raw, puts_raw, target_exp)
    except Exception as e:
        logger.warning(f"Options API fetch warning for {symbol}: {e}")

    if options_chain_data is None:
        return {
            "status": "unavailable",
            "symbol": display_name,
            "message": "Live options chain data is unavailable from market provider.",
            "options_chain_matrix": []
        }

    calls_df, puts_df, expiry_date = options_chain_data

    # 1. Put-Call Ratio (PCR)
    total_call_oi = float(calls_df["openInterest"].sum()) if "openInterest" in calls_df.columns else 1.0
    total_put_oi = float(puts_df["openInterest"].sum()) if "openInterest" in puts_df.columns else 1.0
    total_call_vol = float(calls_df["volume"].sum()) if "volume" in calls_df.columns else 1.0
    total_put_vol = float(puts_df["volume"].sum()) if "volume" in puts_df.columns else 1.0

    pcr_oi = round(total_put_oi / max(1.0, total_call_oi), 2)
    pcr_vol = round(total_put_vol / max(1.0, total_call_vol), 2)

    pcr_signal = "BULLISH 🚀" if pcr_oi >= 1.25 else ("BEARISH 📉" if pcr_oi <= 0.75 else "NEUTRAL ⏸️")

    # 2. Max Pain Calculation
    all_strikes = sorted(list(set(calls_df["strike"].tolist() + puts_df["strike"].tolist())))
    max_pain_strike = _calculate_max_pain(all_strikes, calls_df, puts_df)

    # 3. Key Support & Resistance Levels from Open Interest
    calls_sorted = calls_df.sort_values(by="openInterest", ascending=False) if "openInterest" in calls_df.columns else calls_df
    puts_sorted = puts_df.sort_values(by="openInterest", ascending=False) if "openInterest" in puts_df.columns else puts_df

    resistance_levels = [round(float(s), 2) for s in calls_sorted["strike"].head(3).tolist()] if not calls_sorted.empty else []
    support_levels = [round(float(s), 2) for s in puts_sorted["strike"].head(3).tolist()] if not puts_sorted.empty else []

    # 4. Implied Volatility (IV) Skew
    mean_iv_calls = float(calls_df["impliedVolatility"].mean() * 100) if "impliedVolatility" in calls_df.columns else 14.5
    mean_iv_puts = float(puts_df["impliedVolatility"].mean() * 100) if "impliedVolatility" in puts_df.columns else 16.2
    iv_skew = round(mean_iv_puts - mean_iv_calls, 2)

    # Build options matrix summary around ATM strike
    step = 50.0 if "NIFTY" in display_name else 100.0
    atm_strike = round(spot_price / step) * step

    matrix = []
    strikes_sample = [s for s in all_strikes if abs(s - atm_strike) <= step * 8]
    for st in strikes_sample[:15]:
        c_row = calls_df[calls_df["strike"] == st]
        p_row = puts_df[puts_df["strike"] == st]
        c_oi = int(c_row["openInterest"].iloc[0]) if not c_row.empty and "openInterest" in c_row.columns else 0
        p_oi = int(p_row["openInterest"].iloc[0]) if not p_row.empty and "openInterest" in p_row.columns else 0
        c_ltp = float(c_row["lastPrice"].iloc[0]) if not c_row.empty and "lastPrice" in c_row.columns else 0.0
        p_ltp = float(p_row["lastPrice"].iloc[0]) if not p_row.empty and "lastPrice" in p_row.columns else 0.0

        matrix.append({
            "strike": round(st, 2),
            "is_atm": abs(st - atm_strike) < (step / 2),
            "call_oi": c_oi,
            "call_ltp": round(c_ltp, 2),
            "put_oi": p_oi,
            "put_ltp": round(p_ltp, 2)
        })

    return {
        "status": "success",
        "symbol": display_name,
        "spot_price": round(spot_price, 2),
        "expiry_date": expiry_date if 'expiry_date' in locals() else "Near Month",
        "pcr_oi": pcr_oi,
        "pcr_volume": pcr_vol,
        "pcr_signal": pcr_signal,
        "max_pain_strike": max_pain_strike,
        "iv_skew_pct": iv_skew,
        "avg_call_iv": round(mean_iv_calls, 2),
        "avg_put_iv": round(mean_iv_puts, 2),
        "resistance_levels": resistance_levels,
        "support_levels": support_levels,
        "options_chain_matrix": matrix
    }


def _calculate_max_pain(strikes: List[float], calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> float:
    """Computes strike price where option buyers suffer maximum financial loss."""
    if not strikes:
        return 0.0

    call_oi_map = dict(zip(calls_df["strike"], calls_df["openInterest"])) if "openInterest" in calls_df.columns else {}
    put_oi_map = dict(zip(puts_df["strike"], puts_df["openInterest"])) if "openInterest" in puts_df.columns else {}

    min_pain = float("inf")
    max_pain_strike = strikes[len(strikes) // 2]

    for s_eval in strikes:
        total_loss = 0.0
        # Call losses: max(0, s_eval - K) * call_oi
        for k, oi in call_oi_map.items():
            if s_eval > k:
                total_loss += (s_eval - k) * oi
        # Put losses: max(0, K - s_eval) * put_oi
        for k, oi in put_oi_map.items():
            if s_eval < k:
                total_loss += (k - s_eval) * oi

        if total_loss < min_pain:
            min_pain = total_loss
            max_pain_strike = s_eval

    return round(float(max_pain_strike), 2)


# Simulated options chain generator removed per Zero Synthetic Data policy.
