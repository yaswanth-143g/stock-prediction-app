"""
market_data/yfinance_provider.py
================================
Primary Free Provider wrapping yfinance with retries and validation.
Supports live quotes, batch quotes, historical OHLCV data, and market status.
"""

import logging
import datetime
from datetime import timezone, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd

from market_data.base_provider import BaseMarketDataProvider
from market_data.validators import MarketDataValidator

logger = logging.getLogger("YFinanceProvider")
IST = timezone(timedelta(hours=5, minutes=30))

try:
    from ticker_validator import fetch_raw_data_with_retries
except ImportError:
    def fetch_raw_data_with_retries(ticker, **kw):
        import yfinance as yf
        return yf.download(ticker, progress=False, **kw)


class YFinanceProvider(BaseMarketDataProvider):
    """
    Primary Free Provider implementation wrapping yfinance with retry resilience.
    """

    @property
    def provider_name(self) -> str:
        return "yfinance_provider"

    def is_available(self) -> bool:
        try:
            import yfinance
            return True
        except ImportError:
            return False

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        clean_t = symbol.strip().upper()
        if not clean_t.endswith(".NS") and not clean_t.endswith(".BO") and len(clean_t) <= 12:
            yf_ticker = f"{clean_t}.NS"
        else:
            yf_ticker = clean_t

        clean_sym = clean_t.replace(".NS", "").replace(".BO", "")

        try:
            df = fetch_raw_data_with_retries(yf_ticker, period="5d")
            is_valid, cleaned_df, msg = MarketDataValidator.validate_df(df, ticker=clean_sym)

            if is_valid and cleaned_df is not None and not cleaned_df.empty:
                close_s = cleaned_df["Close"]
                open_s = cleaned_df["Open"] if "Open" in cleaned_df.columns else close_s
                high_s = cleaned_df["High"] if "High" in cleaned_df.columns else close_s
                low_s = cleaned_df["Low"] if "Low" in cleaned_df.columns else close_s
                vol_s = cleaned_df["Volume"] if "Volume" in cleaned_df.columns else pd.Series([0])

                if len(close_s) >= 1:
                    ltp = float(close_s.iloc[-1])
                    prev_close = float(close_s.iloc[-2]) if len(close_s) >= 2 else ltp
                    open_p = float(open_s.iloc[-1])
                    high_p = float(high_s.iloc[-1])
                    low_p = float(low_s.iloc[-1])
                    volume = int(vol_s.iloc[-1]) if not pd.isna(vol_s.iloc[-1]) else 0

                    change = round(ltp - prev_close, 2)
                    change_pct = round((change / prev_close) * 100.0, 2) if prev_close > 0 else 0.0

                    raw_quote = {
                        "ticker": clean_sym,
                        "exchange": "NSE",
                        "ltp": round(ltp, 2),
                        "open": round(open_p, 2),
                        "high": round(high_p, 2),
                        "low": round(low_p, 2),
                        "previous_close": round(prev_close, 2),
                        "volume": volume,
                        "change": change,
                        "change_percent": change_pct,
                        "timestamp": datetime.datetime.now(IST).isoformat(),
                        "source": self.provider_name
                    }

                    # Validate output quote
                    valid_quote, clean_quote, val_msg = MarketDataValidator.validate_quote(raw_quote)
                    if valid_quote:
                        return clean_quote
                    else:
                        logger.warning(f"Quote validation failed for {symbol}: {val_msg}")
        except Exception as e:
            logger.warning(f"YFinanceProvider quote fetch error for {symbol}: {e}")

        return None

    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        results = {}
        for s in symbols:
            q = self.get_quote(s)
            if q:
                results[s] = q
        return results

    def get_history(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        clean_t = symbol.strip().upper()
        if not clean_t.endswith(".NS") and not clean_t.endswith(".BO") and len(clean_t) <= 12:
            yf_ticker = f"{clean_t}.NS"
        else:
            yf_ticker = clean_t

        try:
            df = fetch_raw_data_with_retries(yf_ticker, period=period)
            is_valid, cleaned_df, msg = MarketDataValidator.validate_df(df, ticker=clean_t)
            if is_valid and cleaned_df is not None:
                return cleaned_df
        except Exception as e:
            logger.warning(f"YFinanceProvider get_history error for {symbol}: {e}")

        return pd.DataFrame()

    def get_intraday(self, symbol: str, interval: str = "5m") -> pd.DataFrame:
        return self.get_history(symbol, period="1d", interval=interval)

    def get_market_status(self) -> Dict[str, Any]:
        now_ist = datetime.datetime.now(IST)
        is_weekday = now_ist.weekday() < 5
        current_time = now_ist.time()

        pre_open_start = datetime.datetime.strptime("09:00:00", "%H:%M:%S").time()
        market_start = datetime.datetime.strptime("09:15:00", "%H:%M:%S").time()
        market_end = datetime.datetime.strptime("15:30:00", "%H:%M:%S").time()
        post_market_end = datetime.datetime.strptime("16:00:00", "%H:%M:%S").time()

        if not is_weekday:
            status = "CLOSED"
            status_str = "CLOSED 🔴"
            is_open = False
        elif market_start <= current_time <= market_end:
            status = "OPEN"
            status_str = "OPEN 🟢"
            is_open = True
        elif pre_open_start <= current_time < market_start:
            status = "PRE_OPEN"
            status_str = "PRE-OPEN 🟡"
            is_open = False
        elif market_end < current_time <= post_market_end:
            status = "POST_MARKET"
            status_str = "POST-MARKET 🟡"
            is_open = False
        else:
            status = "CLOSED"
            status_str = "CLOSED 🔴"
            is_open = False

        return {
            "status": "success",
            "is_market_open": is_open,
            "market_status": status,
            "market_status_display": status_str,
            "exchange": "NSE / BSE",
            "trading_hours": "09:15 - 15:30 IST",
            "current_time_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST")
        }
