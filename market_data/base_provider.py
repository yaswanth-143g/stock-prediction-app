"""
market_data/base_provider.py
============================
Abstract Base Class for Free Market Data Providers.
Defines canonical interface methods for live quotes, batch quotes,
historical OHLCV data, intraday series, and market status checks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd


class BaseMarketDataProvider(ABC):
    """
    Abstract Base Class for Market Data Providers.
    All free data providers (yfinance, secondary APIs, local CSV fallbacks)
    must implement this standardized interface.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns unique identifier name of the provider."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Checks whether provider dependencies and endpoints are accessible."""
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetches live/near-real-time quote for a single symbol.
        Returns standardized dict with keys:
        ticker, exchange, ltp, open, high, low, previous_close, volume, change, change_percent, timestamp, source
        """
        pass

    @abstractmethod
    def get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetches batch quotes for multiple symbols.
        Returns dictionary mapping symbol -> quote dict.
        """
        pass

    @abstractmethod
    def get_history(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Retrieves historical OHLCV DataFrame for a symbol.
        DataFrame columns: Date, Open, High, Low, Close, Volume
        """
        pass

    @abstractmethod
    def get_intraday(self, symbol: str, interval: str = "5m") -> pd.DataFrame:
        """
        Retrieves intraday OHLCV DataFrame for a symbol.
        DataFrame columns: Date, Open, High, Low, Close, Volume
        """
        pass

    @abstractmethod
    def get_market_status(self) -> Dict[str, Any]:
        """
        Retrieves current exchange market status (OPEN / CLOSED / PRE-MARKET).
        """
        pass
