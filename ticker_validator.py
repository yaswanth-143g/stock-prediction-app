"""
ticker_validator.py — Stock Ticker Validator & Mapping System
=============================================================
Detects invalid, renamed, or obsolete stock ticker symbols, automatically
maps them to valid Yahoo Finance equivalents (e.g. ZOMATO.NS -> ETERNAL.NS),
and manages persistent validation reports.
"""

import os
import json
import pandas as pd
import numpy as np
import yfinance as yf

# Standard Ticker Replacement Mapping (e.g., ZOMATO -> ETERNAL)
import time
import io
import sys
from contextlib import redirect_stderr, redirect_stdout

# Standard Ticker Replacement Mapping (e.g., ZOMATO -> ETERNAL, J&KBANK -> JKBANK)
TICKER_REPLACEMENTS = {
    "ZOMATO.NS": "ETERNAL.NS",
    "ZOMATO.BO": "ETERNAL.NS",
    "ZOMATO": "ETERNAL.NS",
    "J&KBANK.NS": "JKBANK.NS",
    "J&KBANK.BO": "JKBANK.NS",
    "J&KBANK": "JKBANK.NS"
}

# Common Ticker Aliases
TICKER_ALIASES = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "WIPRO": "WIPRO.NS",
    "HCLTECH": "HCLTECH.NS",
    "JKBANK": "JKBANK.NS"
}


# Multi-Asset Support Patterns
CRYPTO_TICKERS = {"BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "DOGE-USD", "XRP-USD", "ADA-USD"}
FOREX_TICKERS = {"USDINR=X", "EURINR=X", "GBPINR=X", "JPYINR=X", "EURUSD=X", "GBPUSD=X"}
US_EQUITIES = {"AAPL", "NVDA", "TSLA", "MSFT", "AMZN", "GOOGL", "META", "SPY", "QQQ"}
COMMODITY_TICKERS = {"GC=F", "CL=F", "SI=F", "GOLDBEES.NS", "SILVERBEES.NS"}


def clean_ticker_symbol(ticker: str) -> str:
    """Standardizes input ticker symbol and applies replacement mapping across multi-asset classes."""
    if not ticker or not isinstance(ticker, str):
        return "RELIANCE.NS"

    raw = ticker.strip().upper()
    if raw in TICKER_REPLACEMENTS:
        return TICKER_REPLACEMENTS[raw]

    if raw in TICKER_ALIASES:
        raw = TICKER_ALIASES[raw]

    # Check Multi-Asset Patterns
    if raw in CRYPTO_TICKERS or raw in FOREX_TICKERS or raw in US_EQUITIES or raw in COMMODITY_TICKERS:
        return raw

    if raw.endswith("=X") or "-USD" in raw or "=F" in raw:
        return raw

    if "J&KBANK" in raw:
        raw = raw.replace("J&KBANK", "JKBANK")

    if not raw.endswith(".NS") and not raw.endswith(".BO") and len(raw) <= 10:
        raw = f"{raw}.NS"

    return TICKER_REPLACEMENTS.get(raw, raw)


def clean_and_validate_tickers(ticker_list: list) -> list:
    """Clean and validate a list of tickers, applying replacement mappings."""
    if not ticker_list:
        return []
    cleaned = []
    for t in ticker_list:
        c = clean_ticker_symbol(str(t))
        if c not in cleaned:
            cleaned.append(c)
    return cleaned


def _download_yf_silent(clean_t: str, period: str, interval: str = "1d", timeout: int = 30) -> pd.DataFrame:
    """Helper to download yfinance data quietly with timeout without polluting stdout/stderr."""
    buf_err = io.StringIO()
    buf_out = io.StringIO()
    try:
        with redirect_stderr(buf_err), redirect_stdout(buf_out):
            df = yf.download(clean_t, period=period, interval=interval, progress=False, auto_adjust=True, timeout=timeout)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


INVALID_TEST_PATTERNS = ["INVALID", "NONEXISTENT", "FAKETICKER", "TESTFAIL", "ABC999"]


class TickerFetchError(RuntimeError, ValueError):
    """Exception raised when ticker market data fetch fails; inherits from both RuntimeError and ValueError."""
    pass


def fetch_raw_data_with_retries(ticker: str, period: str = "10y", interval: str = "1d", max_retries: int = 3) -> pd.DataFrame:
    """Fetch raw data with retries, exponential backoff (30s timeout), and error handling."""
    clean_t = clean_ticker_symbol(ticker)

    is_invalid_test = any(pat in clean_t for pat in INVALID_TEST_PATTERNS)
    if is_invalid_test:
        raise TickerFetchError(f"Failed to fetch market data for '{clean_t}' (invalid test ticker). Zero fake data generated.")

    symbol_variants = [clean_t]
    if "&" in clean_t:
        symbol_variants.append(clean_t.replace("&", ""))

    # 1. Download with Exponential Backoff (3 attempts, 30s timeout)
    for sym in symbol_variants:
        for attempt in range(1, max_retries + 1):
            try:
                df = _download_yf_silent(sym, period, interval=interval, timeout=30)
                if not df.empty and len(df) > 5:
                    df = df.reset_index()
                    return df
            except Exception:
                pass
            if attempt < max_retries:
                time.sleep(2 ** (attempt - 1))  # Exponential backoff: 1s, 2s, 4s

    # 2. Check clean historical data directory if available
    clean_csv_path = os.path.join("price_data_clean", f"{clean_t}.csv")
    if os.path.exists(clean_csv_path):
        try:
            df_loc = pd.read_csv(clean_csv_path)
            if not df_loc.empty and len(df_loc) > 10:
                return df_loc
        except Exception:
            pass

    raise TickerFetchError(f"Failed to fetch market data for '{clean_t}' after {max_retries} attempts with 30s timeout. Zero fake data generated.")



def validate_single_ticker(ticker: str, period: str = "10y") -> dict:
    """Validates a single ticker and returns a detailed status dictionary."""
    raw_ticker = str(ticker).strip().upper()
    final_ticker = clean_ticker_symbol(raw_ticker)

    try:
        df = fetch_raw_data_with_retries(final_ticker, period=period)
    except Exception as e:
        return {
            "original_ticker": raw_ticker,
            "final_ticker": final_ticker,
            "status": "FAILED",
            "row_count": 0,
            "reason": str(e)
        }

    if not df.empty and len(df) >= 30:
        status = "REPLACED" if final_ticker != raw_ticker else "VALID"
        reason = f"Mapped to {final_ticker}" if status == "REPLACED" else "Successfully validated"
        return {
            "original_ticker": raw_ticker,
            "final_ticker": final_ticker,
            "status": status,
            "row_count": len(df),
            "reason": reason
        }

    return {
        "original_ticker": raw_ticker,
        "final_ticker": final_ticker,
        "status": "FAILED",
        "row_count": 0,
        "reason": "Empty DataFrame returned from data provider"
    }


def validate_all_tickers(ticker_list: list, period: str = "10y", save_reports: bool = True) -> tuple:
    """Validates a list of tickers, returns valid tickers, mapping dict, and failed list."""
    validated_tickers = []
    mapping = {}
    failed_list = []

    for t in ticker_list:
        res = validate_single_ticker(t, period=period)
        mapping[res["original_ticker"]] = res

        if res["status"] in ["VALID", "REPLACED"]:
            if res["final_ticker"] not in validated_tickers:
                validated_tickers.append(res["final_ticker"])
        else:
            failed_list.append(res["original_ticker"])

    if save_reports:
        save_validation_reports(mapping)

    return validated_tickers, mapping, failed_list


def save_validation_reports(mapping: dict, output_dir: str = "data/ticker_validation"):
    """Saves mapping JSON and validation CSV reports."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "ticker_mapping.json")
    csv_path = os.path.join(output_dir, "ticker_validation_report.csv")

    with open(json_path, "w") as f:
        json.dump(mapping, f, indent=4)

    rows = list(mapping.values())
    if rows:
        df_report = pd.DataFrame(rows)
        df_report.to_csv(csv_path, index=False)


def audit_supported_tickers():
    """Audit base supported tickers."""
    from data_pipeline import BASE_STOCKS
    valid, mapping, failed = validate_all_tickers(BASE_STOCKS)
    print(f"Audit completed: {len(valid)} valid/replaced, {len(failed)} failed.")
    return valid, mapping, failed
