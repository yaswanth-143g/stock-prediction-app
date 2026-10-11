"""
market_data/free_api_provider.py
================================
Secondary Free Provider wrapping optional free market APIs (Alpha Vantage / Finnhub)
or local Stock_Data/*.csv dataset fallback when primary provider is unavailable.
Never fabricates random data.
"""

import os
import logging
import datetime
from datetime import timezone, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd
import requests

from market_data.base_provider import BaseMarketDataProvider
from market_data.validators import MarketDataValidator

logger = logging.getLogger("FreeApiProvider")
IST = timezone(timedelta(hours=5, minutes=30))


class FreeApiProvider(BaseMarketDataProvider):
    """
    Secondary Free Provider reading optional FREE API keys (Finnhub / Alpha Vantage)
    or local CSV dataset fallbacks.
    """

    def __init__(self):
        self.finnhub_key = os.environ.get("FINNHUB_API_KEY", "").strip()
        self.alphavantage_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()

    @property
    def provider_name(self) -> str:
        return "free_api_provider"

    def is_available(self) -> bool:
        return True  # Local CSV fallback is always available

    def _fetch_finnhub_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches quote via free Finnhub API if key present."""
        if not self.finnhub_key:
            return None
        clean_sym = symbol.strip().upper().replace(".NS", "").replace(".BO", "")
        url = f"https://finnhub.io/api/v1/quote?symbol={clean_sym}.NS&token={self.finnhub_key}"
        try:
            res = requests.get(url, timeout=4)
            if res.status_code == 200:
                data = res.json()
                ltp = float(data.get("c", 0.0))
                if ltp > 0:
                    prev_c = float(data.get("pc", ltp))
                    return {
                        "ticker": clean_sym,
                        "exchange": "NSE",
                        "ltp": ltp,
                        "open": float(data.get("o", ltp)),
                        "high": float(data.get("h", ltp)),
                        "low": float(data.get("l", ltp)),
                        "previous_close": prev_c,
                        "volume": 0,
                        "change": round(ltp - prev_c, 2),
                        "change_percent": round(((ltp - prev_c) / prev_c) * 100.0, 2) if prev_c > 0 else 0.0,
                        "timestamp": datetime.datetime.now(IST).isoformat(),
                        "source": "finnhub_free_api"
                    }
        except Exception as e:
            logger.debug(f"Finnhub free quote error for {symbol}: {e}")
        return None

    def _fetch_csv_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches quote from local Stock_Data/*.csv dataset fallback."""
        clean_sym = symbol.strip().upper().replace(".NS", "").replace(".BO", "")
        candidates = [
            os.path.join("price_data_clean", f"{clean_sym}.NS.csv"),
            os.path.join("price_data_clean", f"{clean_sym}.csv"),
        ]

        csv_path = None
        for c in candidates:
            if os.path.exists(c):
                csv_path = c
                break

        if not csv_path:
            return None

        try:
            df = pd.read_csv(csv_path)
            is_valid, cleaned_df, msg = MarketDataValidator.validate_df(df, ticker=clean_sym)
            if is_valid and cleaned_df is not None and not cleaned_df.empty:
                close_s = cleaned_df["Close"]
                if len(close_s) >= 1:
                    ltp = float(close_s.iloc[-1])
                    prev_c = float(close_s.iloc[-2]) if len(close_s) >= 2 else ltp
                    open_p = float(cleaned_df["Open"].iloc[-1]) if "Open" in cleaned_df.columns else ltp
                    high_p = float(cleaned_df["High"].iloc[-1]) if "High" in cleaned_df.columns else ltp
                    low_p = float(cleaned_df["Low"].iloc[-1]) if "Low" in cleaned_df.columns else ltp
                    vol = int(cleaned_df["Volume"].iloc[-1]) if "Volume" in cleaned_df.columns else 0

                    change = round(ltp - prev_c, 2)
                    change_pct = round((change / prev_c) * 100.0, 2) if prev_c > 0 else 0.0

                    return {
                        "ticker": clean_sym,
                        "exchange": "NSE",
                        "ltp": round(ltp, 2),
                        "open": round(open_p, 2),
                        "high": round(high_p, 2),
                        "low": round(low_p, 2),
                        "previous_close": round(prev_c, 2),
                        "volume": vol,
                        "change": change,
                        "change_percent": change_pct,
                        "timestamp": datetime.datetime.now(IST).isoformat(),
                        "source": "local_csv_fallback"
                    }
        except Exception as e:
            logger.debug(f"Local CSV quote read error for {symbol}: {e}")

        return None

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        # 1. Try Finnhub free API if configured
        q = self._fetch_finnhub_quote(symbol)
        if q:
            valid, clean_q, _ = MarketDataValidator.validate_quote(q)
            if valid:
                return clean_q

        # 2. Try Local CSV dataset fallback
        q_csv = self._fetch_csv_quote(symbol)
        if q_csv:
            valid, clean_q, _ = MarketDataValidator.validate_quote(q_csv)
            if valid:
                return clean_q

        return None

    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        results = {}
        for s in symbols:
            q = self.get_quote(s)
            if q:
                results[s] = q
        return results

    def get_history(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        clean_sym = symbol.strip().upper().replace(".NS", "").replace(".BO", "")
        candidates = [
            os.path.join("price_data_clean", f"{clean_sym}.NS.csv"),
            os.path.join("price_data_clean", f"{clean_sym}.csv"),
        ]

        for c in candidates:
            if os.path.exists(c):
                try:
                    df = pd.read_csv(c)
                    valid, clean_df, _ = MarketDataValidator.validate_df(df, ticker=clean_sym)
                    if valid and clean_df is not None:
                        return clean_df
                except Exception:
                    pass

        return pd.DataFrame()

    def get_intraday(self, symbol: str, interval: str = "5m") -> pd.DataFrame:
        return self.get_history(symbol, period="1d", interval=interval)

    def get_market_status(self) -> Dict[str, Any]:
        return {
            "status": "success",
            "is_market_open": False,
            "market_status": "SECONDARY FALLBACK 🟡",
            "exchange": "NSE / BSE"
        }
