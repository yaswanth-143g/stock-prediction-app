"""
market_data/cache.py
====================
Smart TTL Cache and Rate Throttling Manager for Market Data.
Caches quotes with short TTL and historical data with longer TTL.
Prevents request storms and protects free API rate limits.
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd

logger = logging.getLogger("SmartMarketDataCache")

DEFAULT_QUOTE_TTL_SECONDS = 15       # 15 seconds short TTL for quotes
DEFAULT_HIST_TTL_SECONDS = 300      # 5 minutes longer TTL for historical data
DEFAULT_MIN_REQUEST_INTERVAL = 1.0  # Minimum 1 second between API calls for same symbol


class SmartMarketDataCache:
    """
    In-memory smart TTL cache for quotes, historical DataFrames, and last valid fallback data.
    Implements rate-limit throttling to prevent request storms.
    """

    def __init__(self, quote_ttl: int = DEFAULT_QUOTE_TTL_SECONDS, hist_ttl: int = DEFAULT_HIST_TTL_SECONDS):
        self.quote_ttl = quote_ttl
        self.hist_ttl = hist_ttl
        self._quote_cache: Dict[str, Dict[str, Any]] = {}
        self._hist_cache: Dict[str, Tuple[pd.DataFrame, float]] = {}
        self._last_valid_quote_cache: Dict[str, Dict[str, Any]] = {}
        self._last_request_time: Dict[str, float] = {}

    def is_rate_limited(self, symbol: str, min_interval: float = DEFAULT_MIN_REQUEST_INTERVAL) -> bool:
        """Checks whether requests for a symbol are being throttled."""
        clean_sym = symbol.strip().upper()
        now = time.time()
        last_time = self._last_request_time.get(clean_sym, 0.0)
        if now - last_time < min_interval:
            return True
        return False

    def record_request(self, symbol: str):
        """Records timestamp of API request for rate-limit tracking."""
        clean_sym = symbol.strip().upper()
        self._last_request_time[clean_sym] = time.time()

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieves non-expired cached quote if available."""
        clean_sym = symbol.strip().upper()
        entry = self._quote_cache.get(clean_sym)
        if entry:
            expires_at = entry.get("expires_at", 0)
            if time.time() < expires_at:
                data = dict(entry["data"])
                data["source"] = "smart_cache"
                return data
            else:
                del self._quote_cache[clean_sym]
        return None

    def set_quote(self, symbol: str, quote: Dict[str, Any], custom_ttl: Optional[int] = None):
        """Caches valid quote and updates last valid quote fallback cache."""
        if not quote:
            return
        clean_sym = symbol.strip().upper()
        ttl = custom_ttl if custom_ttl is not None else self.quote_ttl
        expires_at = time.time() + ttl

        self._quote_cache[clean_sym] = {
            "data": quote,
            "expires_at": expires_at
        }
        # Always update last valid quote fallback
        self._last_valid_quote_cache[clean_sym] = dict(quote)

    def get_last_valid_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves last known valid quote regardless of TTL.
        Used as emergency fallback when all market providers are offline.
        """
        clean_sym = symbol.strip().upper()
        quote = self._last_valid_quote_cache.get(clean_sym)
        if quote:
            cached_quote = dict(quote)
            cached_quote["source"] = "cached_last_valid"
            cached_quote["is_stale"] = True
            cached_quote["freshness_label"] = "⚠ Cached Last Valid Data"
            return cached_quote
        return None

    def get_history(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
        """Retrieves cached historical DataFrame if non-expired."""
        key = f"{symbol.strip().upper()}:{period}:{interval}"
        entry = self._hist_cache.get(key)
        if entry:
            df, expires_at = entry
            if time.time() < expires_at:
                return df.copy()
            else:
                del self._hist_cache[key]
        return None

    def set_history(self, symbol: str, period: str, interval: str, df: pd.DataFrame, custom_ttl: Optional[int] = None):
        """Caches historical DataFrame."""
        if df is None or df.empty:
            return
        key = f"{symbol.strip().upper()}:{period}:{interval}"
        ttl = custom_ttl if custom_ttl is not None else self.hist_ttl
        expires_at = time.time() + ttl
        self._hist_cache[key] = (df.copy(), expires_at)

    def clear(self):
        """Clears all caches."""
        self._quote_cache.clear()
        self._hist_cache.clear()
        self._last_request_time.clear()
