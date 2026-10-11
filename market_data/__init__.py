"""
market_data package
===================
Free Market Data Layer with Abstraction Provider Interface, Multi-Stage Fallback Ladder,
Strict Data Validation, and Smart TTL Caching.
"""

from market_data.base_provider import BaseMarketDataProvider
from market_data.validators import MarketDataValidator
from market_data.cache import SmartMarketDataCache
from market_data.yfinance_provider import YFinanceProvider
from market_data.free_api_provider import FreeApiProvider
from market_data.market_data_manager import MarketDataManager, get_market_data_manager

__all__ = [
    "BaseMarketDataProvider",
    "MarketDataValidator",
    "SmartMarketDataCache",
    "YFinanceProvider",
    "FreeApiProvider",
    "MarketDataManager",
    "get_market_data_manager"
]
