"""
market_data/market_data_manager.py
==================================
Central Market Data Provider Manager with Multi-Stage Fallback Ladder.
Orchestrates quote and historical data requests through:
Primary Provider (yfinance) -> Secondary Provider (Free API/CSV) -> Smart Cache Last Valid Data.
Ensures the dashboard never crashes even during external API downtime.
"""

import logging
from typing import Dict, Any, List, Optional
import pandas as pd

from market_data.base_provider import BaseMarketDataProvider
from market_data.yfinance_provider import YFinanceProvider
from market_data.free_api_provider import FreeApiProvider
from market_data.cache import SmartMarketDataCache
from market_data.validators import MarketDataValidator

logger = logging.getLogger("MarketDataManager")


class MarketDataManager:
    """
    Singleton Manager for Free Market Data Providers with Fallback & Cache.
    """

    def __init__(self):
        self.primary_provider: BaseMarketDataProvider = YFinanceProvider()
        self.secondary_provider: BaseMarketDataProvider = FreeApiProvider()
        self.cache = SmartMarketDataCache()

    def get_quote(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Public entry point for single stock quote.
        Fallback Ladder: Smart Cache -> Primary Provider -> Secondary Provider -> Emergency Last Valid Cache.
        """
        clean_sym = symbol.strip().upper().replace(".NS", "").replace(".BO", "")
        formatted_sym = f"{clean_sym}.NS" if len(clean_sym) <= 12 else clean_sym

        # 1. Check Smart TTL Cache
        if use_cache:
            cached_quote = self.cache.get_quote(formatted_sym) or self.cache.get_quote(clean_sym)
            if cached_quote:
                return cached_quote

        # 2. Try Primary Provider (yfinance)
        if self.primary_provider.is_available():
            try:
                quote = self.primary_provider.get_quote(formatted_sym)
                if quote and quote.get("ltp", 0.0) > 0:
                    self.cache.set_quote(formatted_sym, quote)
                    self.cache.set_quote(clean_sym, quote)
                    return quote
            except Exception as e:
                logger.warning(f"Primary provider failed for {symbol}: {e}")

        # 3. Try Secondary Provider (Free API / Local CSV)
        if self.secondary_provider.is_available():
            try:
                quote = self.secondary_provider.get_quote(formatted_sym) or self.secondary_provider.get_quote(clean_sym)
                if quote and quote.get("ltp", 0.0) > 0:
                    self.cache.set_quote(formatted_sym, quote)
                    self.cache.set_quote(clean_sym, quote)
                    return quote
            except Exception as e:
                logger.warning(f"Secondary provider failed for {symbol}: {e}")

        # 4. Emergency Fallback: Cached Last Valid Data
        last_valid = self.cache.get_last_valid_quote(formatted_sym) or self.cache.get_last_valid_quote(clean_sym)
        if last_valid:
            return last_valid

        # 5. Final fallback return (Never crash)
        return {
            "ticker": clean_sym,
            "exchange": "NSE",
            "ltp": 0.0,
            "open": 0.0,
            "high": 0.0,
            "low": 0.0,
            "previous_close": 0.0,
            "volume": 0,
            "change": 0.0,
            "change_percent": 0.0,
            "timestamp": "",
            "source": "unavailable",
            "is_live": False,
            "is_stale": True,
            "freshness_label": "⚠ Market Data Unavailable"
        }

    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Retrieves batch quotes across all symbols using fallback ladder."""
        results = {}
        for s in symbols:
            results[s] = self.get_quote(s)
        return results

    def get_history(self, symbol: str, period: str = "1y", interval: str = "1d", use_cache: bool = True) -> pd.DataFrame:
        """
        Retrieves historical OHLCV DataFrame using fallback ladder and RAM caching.
        """
        clean_sym = symbol.strip().upper().replace(".NS", "").replace(".BO", "")
        formatted_sym = f"{clean_sym}.NS" if len(clean_sym) <= 12 else clean_sym

        # 1. Check Cache
        if use_cache:
            cached_df = self.cache.get_history(formatted_sym, period, interval)
            if cached_df is not None and not cached_df.empty:
                return cached_df

        # 2. Try Primary Provider
        df = self.primary_provider.get_history(formatted_sym, period, interval)
        if df is not None and not df.empty:
            self.cache.set_history(formatted_sym, period, interval, df)
            return df

        # 3. Try Secondary Provider
        df = self.secondary_provider.get_history(formatted_sym, period, interval)
        if df is not None and not df.empty:
            self.cache.set_history(formatted_sym, period, interval, df)
            return df

        return pd.DataFrame()

    def get_intraday(self, symbol: str, interval: str = "5m") -> pd.DataFrame:
        """Retrieves intraday OHLCV DataFrame."""
        return self.get_history(symbol, period="1d", interval=interval)

    def get_market_status(self) -> Dict[str, Any]:
        """Retrieves current market trading status."""
        return self.primary_provider.get_market_status()


# Singleton Instance & Getter
_market_data_manager_instance = None


def get_market_data_manager() -> MarketDataManager:
    """Returns global singleton instance of MarketDataManager."""
    global _market_data_manager_instance
    if _market_data_manager_instance is None:
        _market_data_manager_instance = MarketDataManager()
    return _market_data_manager_instance
