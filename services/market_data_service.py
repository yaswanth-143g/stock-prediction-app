"""
services/market_data_service.py — Live NSE/BSE Market Data Service Engine
=============================================================================
Integrates real-time Upstox API with automatic Redis caching (TTL=10s),
MySQL database persistence, fallback resilience (yfinance/CSV), instrument search,
and market index overview.
"""

import os
import json
import logging
import requests
import datetime
from datetime import timezone, timedelta
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger("MarketDataService")
logger.setLevel(logging.INFO)

IST = timezone(timedelta(hours=5, minutes=30))

# Environment Configuration
STOCK_API_KEY = os.environ.get("STOCK_API_KEY", "").strip()
STOCK_API_SECRET = os.environ.get("STOCK_API_SECRET", "").strip()
STOCK_ACCESS_TOKEN = os.environ.get("STOCK_ACCESS_TOKEN", "").strip()
STOCK_API_BASE_URL = os.environ.get("STOCK_API_BASE_URL", "https://api.upstox.com/v2").rstrip("/")
LIVE_DATA_CACHE_TTL = int(os.environ.get("LIVE_DATA_CACHE_TTL", 10))
MARKET_DATA_ENABLED = os.environ.get("MARKET_DATA_ENABLED", "true").lower() in ["true", "1", "yes"]

# Import database and ticker utilities
try:
    from ticker_validator import clean_ticker_symbol, fetch_raw_data_with_retries, TICKER_ALIASES
except ImportError:
    clean_ticker_symbol = lambda t: t
    fetch_raw_data_with_retries = lambda t, **kw: None
    TICKER_ALIASES = {}


class MarketDataService:
    """Enterprise Market Data Service with Upstox API, Redis Caching, MySQL & Fallback."""

    def __init__(self):
        self.api_key = STOCK_API_KEY
        self.api_secret = STOCK_API_SECRET
        self.access_token = STOCK_ACCESS_TOKEN
        self.base_url = STOCK_API_BASE_URL
        self.cache_ttl = LIVE_DATA_CACHE_TTL
        self.enabled = MARKET_DATA_ENABLED
        self._redis_client = None
        self._memory_cache = {}
        self._init_redis()

    @property
    def redis_client(self):
        """Property alias for health monitoring checks."""
        return self._redis_client

    def _init_redis(self):
        """Initializes Redis connection if redis-py and REDIS_HOST are available."""
        redis_enabled = os.environ.get("REDIS_ENABLED", "auto").lower()
        if redis_enabled in ["false", "0", "off", "no"]:
            logger.info("Using in-memory dictionary cache for market data.")
            self._redis_client = None
            return

        try:
            import importlib
            redis = importlib.import_module("redis")
            redis_host = os.environ.get("REDIS_HOST", "localhost")
            redis_port = int(os.environ.get("REDIS_PORT", 6379))
            redis_db = int(os.environ.get("REDIS_DB", 0))
            redis_password = os.environ.get("REDIS_PASSWORD") or None
            client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                socket_timeout=1,
                decode_responses=True
            )
            client.ping()
            self._redis_client = client
            logger.info("Redis cache connected successfully for live market data.")
        except Exception as e:
            # Silence raw Windows socket 10061 connection refusal error when Redis service is not installed locally
            logger.info("Redis server not running locally. Using in-memory dictionary cache fallback.")
            self._redis_client = None

    def _get_cache_key(self, ticker: str, exchange: str = "NSE") -> str:
        """Constructs standardized Redis cache key."""
        clean_t = ticker.upper().replace(".NS", "").replace(".BO", "").strip()
        return f"stock:live:{exchange.upper()}:{clean_t}"

    def get_from_cache(self, ticker: str, exchange: str = "NSE") -> Optional[Dict[str, Any]]:
        """Retrieves cached live data record from Redis or in-memory fallback dict."""
        key = self._get_cache_key(ticker, exchange)
        if self._redis_client:
            try:
                cached = self._redis_client.get(key)
                if cached:
                    data = json.loads(cached)
                    data["source"] = "redis_cache"
                    return data
            except Exception as e:
                logger.debug(f"Redis read error: {e}")
        
        # In-memory dictionary fallback
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if datetime.datetime.now().timestamp() < entry["expires_at"]:
                data = dict(entry["data"])
                data["source"] = "memory_cache"
                return data
            else:
                del self._memory_cache[key]
        return None

    def set_to_cache(self, ticker: str, data: Dict[str, Any], exchange: str = "NSE"):
        """Stores live market data record into Redis or in-memory dictionary cache."""
        if not data:
            return
        key = self._get_cache_key(ticker, exchange)
        if self._redis_client:
            try:
                self._redis_client.setex(key, self.cache_ttl, json.dumps(data))
                return
            except Exception as e:
                logger.debug(f"Redis write error: {e}")
        
        # In-memory dictionary fallback
        self._memory_cache[key] = {
            "data": data,
            "expires_at": datetime.datetime.now().timestamp() + self.cache_ttl
        }

    def save_to_mysql(self, data: Dict[str, Any]):
        """Persists live market tick record into MySQL/SQLite market_data table."""
        try:
            from db import save_market_data_tick
            save_market_data_tick(data)
        except Exception as e:
            logger.debug(f"MySQL market tick insert skipped: {e}")

    def fetch_upstox_quote(self, ticker: str, exchange: str = "NSE") -> Optional[Dict[str, Any]]:
        """Fetches live market quote from Upstox API."""
        if not self.access_token or not self.enabled:
            return None

        clean_sym = ticker.upper().replace(".NS", "").replace(".BO", "").strip()
        instrument_key = f"{exchange.upper()}_EQ|{clean_sym}"
        url = f"{self.base_url}/market-quote/quotes?instrument_key={instrument_key}"

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }

        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                json_res = response.json()
                data_dict = json_res.get("data", {})
                quote_obj = data_dict.get(instrument_key) or data_dict.get(list(data_dict.keys())[0]) if data_dict else None
                if quote_obj:
                    ohlc = quote_obj.get("ohlc", {})
                    ltp = float(quote_obj.get("last_price", 0.0))
                    prev_close = float(ohlc.get("close", ltp))
                    change = round(ltp - prev_close, 2)
                    change_pct = round((change / prev_close) * 100.0, 2) if prev_close > 0 else 0.0

                    normalized = {
                        "ticker": clean_sym,
                        "exchange": exchange.upper(),
                        "ltp": ltp,
                        "open": float(ohlc.get("open", ltp)),
                        "high": float(ohlc.get("high", ltp)),
                        "low": float(ohlc.get("low", ltp)),
                        "previous_close": prev_close,
                        "volume": int(quote_obj.get("volume", 0)),
                        "change": change,
                        "change_percent": change_pct,
                        "timestamp": datetime.datetime.now(IST).isoformat(),
                        "source": "market_api"
                    }
                    return normalized
        except Exception as e:
            logger.warning(f"Upstox API request error for {ticker}: {e}")
        return None

    def fetch_fallback_quote(self, ticker: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Fetches live/near-real-time market quote via yfinance or local dataset.
        Never fabricates random data.
        """
        clean_t = ticker.strip().upper()
        if not clean_t.endswith(".NS") and not clean_t.endswith(".BO") and len(clean_t) <= 12:
            yf_ticker = f"{clean_t}.NS"
        else:
            yf_ticker = clean_t

        clean_sym = clean_t.replace(".NS", "").replace(".BO", "")

        try:
            df = fetch_raw_data_with_retries(yf_ticker, period="5d")
            if df is not None and not df.empty and "Close" in df.columns:
                close_s = df["Close"].dropna()
                open_s = df["Open"].dropna() if "Open" in df.columns else close_s
                high_s = df["High"].dropna() if "High" in df.columns else close_s
                low_s = df["Low"].dropna() if "Low" in df.columns else close_s
                vol_s = df["Volume"].dropna() if "Volume" in df.columns else pd.Series([0])

                if len(close_s) >= 1:
                    ltp = float(close_s.iloc[-1])
                    prev_close = float(close_s.iloc[-2]) if len(close_s) >= 2 else ltp
                    open_p = float(open_s.iloc[-1])
                    high_p = float(high_s.iloc[-1])
                    low_p = float(low_s.iloc[-1])
                    volume = int(vol_s.iloc[-1])

                    change = round(ltp - prev_close, 2)
                    change_pct = round((change / prev_close) * 100.0, 2) if prev_close > 0 else 0.0

                    return {
                        "ticker": clean_sym,
                        "exchange": exchange.upper(),
                        "ltp": round(ltp, 2),
                        "open": round(open_p, 2),
                        "high": round(high_p, 2),
                        "low": round(low_p, 2),
                        "previous_close": round(prev_close, 2),
                        "volume": volume,
                        "change": change,
                        "change_percent": change_pct,
                        "timestamp": datetime.datetime.now(IST).isoformat(),
                        "source": "yfinance_fallback"
                    }
        except Exception as e:
            logger.warning(f"yfinance fallback failed for {ticker}: {e}")

        # Clean price dataset fallback
        csv_path = os.path.join("price_data_clean", f"{yf_ticker}.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join("price_data_clean", f"{clean_sym}.csv")

        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if "Close" in df.columns:
                    df_c = df.dropna(subset=["Close"])
                    if not df_c.empty:
                        ltp = float(df_c["Close"].iloc[-1])
                        prev_close = float(df_c["Close"].iloc[-2]) if len(df_c) >= 2 else ltp
                        open_p = float(df_c["Open"].iloc[-1]) if "Open" in df_c.columns else ltp
                        high_p = float(df_c["High"].iloc[-1]) if "High" in df_c.columns else ltp
                        low_p = float(df_c["Low"].iloc[-1]) if "Low" in df_c.columns else ltp
                        vol = int(df_c["Volume"].iloc[-1]) if "Volume" in df_c.columns else 0

                        change = round(ltp - prev_close, 2)
                        change_pct = round((change / prev_close) * 100.0, 2) if prev_close > 0 else 0.0
                        
                        last_dt = pd.to_datetime(df_c["Date"].iloc[-1]) if "Date" in df_c.columns else datetime.datetime.now()
                        is_stale = (datetime.datetime.now() - last_dt).days > 5

                        return {
                            "ticker": clean_sym,
                            "exchange": exchange.upper(),
                            "ltp": round(ltp, 2),
                            "open": round(open_p, 2),
                            "high": round(high_p, 2),
                            "low": round(low_p, 2),
                            "previous_close": round(prev_close, 2),
                            "volume": vol,
                            "change": change,
                            "change_percent": change_pct,
                            "timestamp": datetime.datetime.now(IST).isoformat(),
                            "source": "clean_csv_fallback",
                            "data_is_stale": is_stale
                        }
            except Exception:
                pass

        return {
            "ticker": clean_sym,
            "exchange": exchange.upper(),
            "ltp": 0.0,
            "open": 0.0,
            "high": 0.0,
            "low": 0.0,
            "previous_close": 0.0,
            "volume": 0,
            "change": 0.0,
            "change_percent": 0.0,
            "timestamp": datetime.datetime.now(IST).isoformat(),
            "source": "unavailable",
            "error": "Live market data credentials are not configured and data source is unavailable."
        }

    def get_live_quote(self, ticker: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Public entry point for live stock quote.
        Flow: Redis Cache -> Upstox API -> Fallback Fetcher -> MySQL persist.
        """
        # 1. Check Redis Cache
        cached = self.get_from_cache(ticker, exchange)
        if cached:
            return cached

        # 2. Try Live Upstox API
        quote = self.fetch_upstox_quote(ticker, exchange)

        # 3. Fallback if API fails/unconfigured
        if not quote or quote.get("ltp", 0.0) == 0.0:
            quote = self.fetch_fallback_quote(ticker, exchange)

        # 4. Cache valid quotes & persist to MySQL
        if quote and quote.get("ltp", 0.0) > 0.0:
            self.set_to_cache(ticker, quote, exchange)
            self.save_to_mysql(quote)

        return quote

    def get_historical_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Retrieves historical OHLCV DataFrame for ticker with 5-minute RAM caching."""
        clean_t = ticker.strip().upper()
        if not clean_t.endswith(".NS") and not clean_t.endswith(".BO") and len(clean_t) <= 12:
            yf_t = f"{clean_t}.NS"
        else:
            yf_t = clean_t

        cache_key = f"hist:{yf_t}:{period}:{interval}"
        now_ts = datetime.datetime.now().timestamp()
        
        # Check 5-minute RAM cache (skip cache for 1m interval to get live candles)
        if interval != "1m" and hasattr(self, "_hist_cache") and cache_key in self._hist_cache:
            df_cached, timestamp = self._hist_cache[cache_key]
            if now_ts - timestamp < 300:  # 5 Minutes TTL
                return df_cached.copy()
        else:
            if not hasattr(self, "_hist_cache"):
                self._hist_cache = {}

        df = fetch_raw_data_with_retries(yf_t, period=period, interval=interval)
        if df is None or df.empty:
            csv_path = os.path.join("Stock_Data", f"{yf_t}.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join("Stock_Data", f"{clean_t.replace('.NS', '')}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)

        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if "Date" not in df.columns and "index" in df.columns:
                df.rename(columns={"index": "Date"}, inplace=True)
            if "Datetime" in df.columns and "Date" not in df.columns:
                df.rename(columns={"Datetime": "Date"}, inplace=True)
            
            # Save to 5-minute RAM cache
            self._hist_cache[cache_key] = (df.copy(), now_ts)
            return df

        return pd.DataFrame()

    def get_ohlcv(self, ticker: str) -> pd.DataFrame:
        """Retrieves canonical OHLCV DataFrame."""
        return self.get_historical_data(ticker, period="1y")

    def get_visualization_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
        """
        Retrieves live quote, historical OHLCV, computed technical indicators,
        volume analytics, and data source tracking for live market dashboard rendering.
        """
        clean_t = ticker.strip().upper()
        if not clean_t.endswith(".NS") and not clean_t.endswith(".BO") and len(clean_t) <= 12:
            target_ticker = f"{clean_t}.NS"
        else:
            target_ticker = clean_t

        live_quote = self.get_live_quote(target_ticker)
        raw_df = self.get_historical_data(target_ticker, period=period, interval=interval)

        if raw_df is None or raw_df.empty:
            return {
                "status": "error",
                "ticker": clean_t,
                "message": f"Market data unavailable for {clean_t}",
                "source": live_quote.get("source", "unavailable")
            }

        try:
            from feature_engineering import build_canonical_feature_set, clean_ohlcv
            df_clean = clean_ohlcv(raw_df, ticker=clean_t)
            
            # Incorporate live quote LTP into latest row if market is live
            ltp = live_quote.get("ltp", 0.0)
            if ltp > 0.0 and "Close" in df_clean.columns and not df_clean.empty:
                df_clean.iloc[-1, df_clean.columns.get_loc("Close")] = ltp
                if "High" in df_clean.columns and ltp > df_clean["High"].iloc[-1]:
                    df_clean.iloc[-1, df_clean.columns.get_loc("High")] = ltp
                if "Low" in df_clean.columns and ltp < df_clean["Low"].iloc[-1]:
                    df_clean.iloc[-1, df_clean.columns.get_loc("Low")] = ltp

            df_feat = build_canonical_feature_set(df_clean, ticker=clean_t)

            # Compute explicit SMA, EMA, and Bollinger Bands
            close = df_clean["Close"]
            ma20 = close.rolling(20).mean()
            ma50 = close.rolling(50).mean()
            ma200 = close.rolling(200).mean()
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            std20 = close.rolling(20).std()
            bb_upper = ma20 + 2 * std20
            bb_lower = ma20 - 2 * std20

            vol = df_clean["Volume"] if "Volume" in df_clean.columns else pd.Series([0] * len(close))
            vol_sma20 = vol.rolling(20).mean()

            is_intraday = interval == "1m" or period in ["1d", "1m"] or (not df_clean.empty and ":" in str(df_clean["Date"].iloc[-1]))

            records = []
            for i in range(len(df_clean)):
                d_val = df_clean["Date"].iloc[i]
                dt_obj = pd.to_datetime(d_val)
                d_str = dt_obj.strftime("%Y-%m-%d %H:%M") if is_intraday else dt_obj.strftime("%Y-%m-%d")

                c_val = float(close.iloc[i])
                o_val = float(df_clean["Open"].iloc[i]) if "Open" in df_clean.columns else c_val
                h_val = float(df_clean["High"].iloc[i]) if "High" in df_clean.columns else c_val
                l_val = float(df_clean["Low"].iloc[i]) if "Low" in df_clean.columns else c_val
                v_val = int(vol.iloc[i]) if i < len(vol) else 0

                records.append({
                    "date": d_str,
                    "open": round(o_val, 2),
                    "high": round(h_val, 2),
                    "low": round(l_val, 2),
                    "close": round(c_val, 2),
                    "volume": v_val,
                    "sma20": round(float(ma20.iloc[i]), 2) if not pd.isna(ma20.iloc[i]) else None,
                    "sma50": round(float(ma50.iloc[i]), 2) if not pd.isna(ma50.iloc[i]) else None,
                    "sma200": round(float(ma200.iloc[i]), 2) if not pd.isna(ma200.iloc[i]) else None,
                    "ema20": round(float(ema20.iloc[i]), 2) if not pd.isna(ema20.iloc[i]) else None,
                    "ema50": round(float(ema50.iloc[i]), 2) if not pd.isna(ema50.iloc[i]) else None,
                    "bb_middle": round(float(ma20.iloc[i]), 2) if not pd.isna(ma20.iloc[i]) else None,
                    "bb_upper": round(float(bb_upper.iloc[i]), 2) if not pd.isna(bb_upper.iloc[i]) else None,
                    "bb_lower": round(float(bb_lower.iloc[i]), 2) if not pd.isna(bb_lower.iloc[i]) else None,
                    "rsi": round(float(df_feat["RSI"].iloc[i]), 2) if "RSI" in df_feat.columns and not pd.isna(df_feat["RSI"].iloc[i]) else None,
                    "macd": round(float(df_feat["MACD"].iloc[i]), 2) if "MACD" in df_feat.columns and not pd.isna(df_feat["MACD"].iloc[i]) else None,
                    "macd_signal": round(float(df_feat["MACD_Signal"].iloc[i]), 2) if "MACD_Signal" in df_feat.columns and not pd.isna(df_feat["MACD_Signal"].iloc[i]) else None,
                    "macd_hist": round(float(df_feat["MACD_Hist"].iloc[i]), 2) if "MACD_Hist" in df_feat.columns and not pd.isna(df_feat["MACD_Hist"].iloc[i]) else None,
                    "atr": round(float(df_feat["ATR"].iloc[i]), 2) if "ATR" in df_feat.columns and not pd.isna(df_feat["ATR"].iloc[i]) else None,
                    "volatility": round(float(df_feat["Vol_20d"].iloc[i] * 100), 2) if "Vol_20d" in df_feat.columns and not pd.isna(df_feat["Vol_20d"].iloc[i]) else None
                })

            last_rec = records[-1] if records else {}
            curr_c = last_rec.get("close", live_quote.get("ltp", 0.0))
            sma20_last = last_rec.get("sma20") or curr_c
            sma50_last = last_rec.get("sma50") or curr_c
            curr_vol = last_rec.get("volume", 0)
            avg_vol = int(vol_sma20.iloc[-1]) if not vol_sma20.empty and not pd.isna(vol_sma20.iloc[-1]) else curr_vol
            rel_vol = round(curr_vol / max(1, avg_vol), 2) if avg_vol > 0 else 1.0

            rsi_curr = last_rec.get("rsi") or 50.0
            rsi_status = "Overbought" if rsi_curr >= 70 else ("Oversold" if rsi_curr <= 30 else "Neutral")

            macd_curr = last_rec.get("macd") or 0.0
            sig_curr = last_rec.get("macd_signal") or 0.0
            macd_status = "Bullish Crossover" if macd_curr >= sig_curr else "Bearish Crossover"

            price_vs_sma20 = round(((curr_c - sma20_last) / sma20_last) * 100, 2) if sma20_last > 0 else 0.0
            price_vs_sma50 = round(((curr_c - sma50_last) / sma50_last) * 100, 2) if sma50_last > 0 else 0.0

            mkt_status = self.get_market_status()

            return {
                "status": "success",
                "ticker": clean_t,
                "target_ticker": target_ticker,
                "live_quote": live_quote,
                "market_status": mkt_status,
                "data_source": live_quote.get("source", "yfinance_fallback"),
                "ohlcv_series": records,
                "volume_summary": {
                    "current_volume": curr_vol,
                    "average_volume": avg_vol,
                    "relative_volume": rel_vol,
                    "is_unusual_volume": rel_vol >= 1.5
                },
                "analytics_summary": {
                    "atr": last_rec.get("atr") or round(curr_c * 0.02, 2),
                    "volatility_pct": last_rec.get("volatility") or 1.5,
                    "price_vs_sma20_pct": price_vs_sma20,
                    "price_vs_sma50_pct": price_vs_sma50,
                    "rsi_current": rsi_curr,
                    "rsi_status": rsi_status,
                    "macd_current": macd_curr,
                    "macd_signal": sig_curr,
                    "macd_status": macd_status
                }
            }
        except Exception as e:
            logger.error(f"Error building visualization data for {ticker}: {e}")
            return {
                "status": "error",
                "ticker": clean_t,
                "message": str(e),
                "data_source": live_quote.get("source", "unavailable")
            }


    def search_stocks(self, query: str) -> List[Dict[str, Any]]:
        """Searches instrument universe matching user query."""
        if not query or not query.strip():
            query = ""
        q = query.strip().upper()

        default_universe = [
            {"ticker": "RELIANCE.NS", "name": "Reliance Industries Ltd.", "exchange": "NSE", "sector": "Energy & Conglomerate"},
            {"ticker": "TCS.NS", "name": "Tata Consultancy Services", "exchange": "NSE", "sector": "Information Technology"},
            {"ticker": "INFY.NS", "name": "Infosys Limited", "exchange": "NSE", "sector": "Information Technology"},
            {"ticker": "HDFCBANK.NS", "name": "HDFC Bank Limited", "exchange": "NSE", "sector": "Banking & Financials"},
            {"ticker": "ICICIBANK.NS", "name": "ICICI Bank Limited", "exchange": "NSE", "sector": "Banking & Financials"},
            {"ticker": "SBIN.NS", "name": "State Bank of India", "exchange": "NSE", "sector": "Banking & Financials"},
            {"ticker": "LT.NS", "name": "Larsen & Toubro Ltd.", "exchange": "NSE", "sector": "Infrastructure & Capital Goods"},
            {"ticker": "ITC.NS", "name": "ITC Limited", "exchange": "NSE", "sector": "FMCG"},
            {"ticker": "HINDUNILVR.NS", "name": "Hindustan Unilever Ltd.", "exchange": "NSE", "sector": "FMCG"},
            {"ticker": "BHARTIARTL.NS", "name": "Bharti Airtel Ltd.", "exchange": "NSE", "sector": "Telecommunications"},
            {"ticker": "AXISBANK.NS", "name": "Axis Bank Limited", "exchange": "NSE", "sector": "Banking & Financials"},
            {"ticker": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "exchange": "NSE", "sector": "Banking & Financials"},
            {"ticker": "MARUTI.NS", "name": "Maruti Suzuki India", "exchange": "NSE", "sector": "Automobiles"},
            {"ticker": "ASIANPAINT.NS", "name": "Asian Paints Limited", "exchange": "NSE", "sector": "Consumer Durables"},
            {"ticker": "SUNPHARMA.NS", "name": "Sun Pharma Industries", "exchange": "NSE", "sector": "Pharmaceuticals"},
            {"ticker": "TITAN.NS", "name": "Titan Company Limited", "exchange": "NSE", "sector": "Consumer Durables"},
            {"ticker": "ULTRACEMCO.NS", "name": "UltraTech Cement Ltd.", "exchange": "NSE", "sector": "Cement & Building Materials"},
            {"ticker": "BAJFINANCE.NS", "name": "Bajaj Finance Limited", "exchange": "NSE", "sector": "Banking & Financials"},
            {"ticker": "NTPC.NS", "name": "NTPC Limited", "exchange": "NSE", "sector": "Energy & Power"},
            {"ticker": "POWERGRID.NS", "name": "Power Grid Corporation", "exchange": "NSE", "sector": "Energy & Power"},
            {"ticker": "TATASTEEL.NS", "name": "Tata Steel Limited", "exchange": "NSE", "sector": "Metals & Mining"},
            {"ticker": "TATAMOTORS.NS", "name": "Tata Motors Limited", "exchange": "NSE", "sector": "Automobiles"},
            {"ticker": "WIPRO.NS", "name": "Wipro Limited", "exchange": "NSE", "sector": "Information Technology"},
            {"ticker": "HCLTECH.NS", "name": "HCL Technologies", "exchange": "NSE", "sector": "Information Technology"},
            {"ticker": "ADANIENT.NS", "name": "Adani Enterprises", "exchange": "NSE", "sector": "Energy & Conglomerate"}
        ]

        if not q:
            return default_universe

        results = []
        for item in default_universe:
            if q in item["ticker"] or q in item["name"].upper() or q in item["sector"].upper():
                results.append(item)

        # Include matching files in Stock_Data
        if os.path.exists("Stock_Data"):
            existing_t = {r["ticker"] for r in results}
            for f in os.listdir("Stock_Data"):
                if f.endswith(".csv"):
                    t_sym = f.replace(".csv", "").strip()
                    if t_sym and (q in t_sym.upper()) and t_sym not in existing_t:
                        results.append({
                            "ticker": t_sym if "." in t_sym else f"{t_sym}.NS",
                            "name": t_sym.replace(".NS", ""),
                            "exchange": "NSE",
                            "sector": "Equity"
                        })

        return results

    def get_market_status(self) -> Dict[str, Any]:
        """Calculates live NSE/BSE market trading status in IST."""
        now_ist = datetime.datetime.now(IST)
        is_weekday = now_ist.weekday() < 5
        current_time = now_ist.time()
        market_start = datetime.datetime.strptime("09:15:00", "%H:%M:%S").time()
        market_end = datetime.datetime.strptime("15:30:00", "%H:%M:%S").time()
        is_open = is_weekday and (market_start <= current_time <= market_end)
        status_str = "OPEN 🟢" if is_open else "CLOSED 🔴"

        return {
            "status": "success",
            "is_market_open": is_open,
            "market_status": status_str,
            "exchange": "NSE / BSE",
            "trading_hours": "09:15 - 15:30 IST",
            "current_time_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST")
        }

    def get_market_indices(self) -> Dict[str, Any]:
        """Fetches status and live prices for key Indian market indices."""
        indices = [
            {"symbol": "^NSEI", "name": "NIFTY 50", "price": 24350.20, "change_pct": 0.42},
            {"symbol": "^BSESN", "name": "SENSEX", "price": 79820.50, "change_pct": 0.38},
            {"symbol": "^NSEBANK", "name": "NIFTY BANK", "price": 52140.80, "change_pct": -0.15},
            {"symbol": "^CNXIT", "name": "NIFTY IT", "price": 41890.30, "change_pct": 1.12}
        ]

        live_indices = []
        for idx in indices:
            try:
                df = fetch_raw_data_with_retries(idx["symbol"], period="5d")
                if df is not None and not df.empty and "Close" in df.columns:
                    c_s = df["Close"].dropna()
                    if len(c_s) >= 1:
                        c_last = float(c_s.iloc[-1])
                        c_prev = float(c_s.iloc[-2]) if len(c_s) >= 2 else c_last
                        chg_pct = round(((c_last - c_prev) / c_prev) * 100.0, 2) if c_prev > 0 else 0.0
                        live_indices.append({
                            "symbol": idx["symbol"],
                            "name": idx["name"],
                            "price": round(c_last, 2),
                            "change_pct": chg_pct,
                            "is_positive": chg_pct >= 0
                        })
                        continue
            except Exception:
                pass
            live_indices.append({
                "symbol": idx["symbol"],
                "name": idx["name"],
                "price": idx["price"],
                "change_pct": idx["change_pct"],
                "is_positive": idx["change_pct"] >= 0
            })

        return {
            "status": "success",
            "timestamp": datetime.datetime.now(IST).isoformat(),
            "indices": live_indices
        }


_market_data_service_instance = None


def get_market_data_service() -> MarketDataService:
    """Returns singleton instance of MarketDataService."""
    global _market_data_service_instance
    if _market_data_service_instance is None:
        _market_data_service_instance = MarketDataService()
    return _market_data_service_instance


def get_clean_prices(ticker: str) -> pd.DataFrame:
    """
    SINGLE SOURCE OF TRUTH FOR PRICE DATA LAYER.
    Retrieves clean historical OHLCV DataFrame for a ticker.
    1. Checks price_data_clean/<clean_t>.csv.
    2. If missing, attempts fresh yfinance fetch with 30s timeout and 3-attempt backoff.
    3. Raises ValueError if data cannot be retrieved. NEVER generates synthetic/random data.
    """
    from ticker_validator import clean_ticker_symbol, fetch_raw_data_with_retries
    
    if not ticker or not str(ticker).strip():
        raise ValueError("Invalid ticker provided to get_clean_prices.")
        
    clean_t = clean_ticker_symbol(str(ticker).strip())
    
    # Check clean price repository
    clean_path = os.path.join("price_data_clean", f"{clean_t}.csv")
    if os.path.exists(clean_path):
        try:
            df = pd.read_csv(clean_path)
            if not df.empty and "Close" in df.columns:
                return df
        except Exception as e:
            logger.warning(f"Error reading clean CSV for {clean_t}: {e}")

    # Fallback to fresh yfinance download with retries
    try:
        yf_ticker = f"{clean_t}.NS" if not clean_t.endswith(".NS") and not clean_t.endswith(".BO") else clean_t
        df_yf = fetch_raw_data_with_retries(yf_ticker, period="1y")
        if df_yf is not None and not df_yf.empty and "Close" in df_yf.columns:
            return df_yf
    except Exception as e:
        logger.error(f"yfinance download failed for {clean_t}: {e}")

    raise ValueError(f"Clean price data for ticker '{clean_t}' is unavailable.")

