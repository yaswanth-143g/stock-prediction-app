"""
market_data/validators.py
=========================
Strict Data Validation and Data Freshness Engine.
Validates quotes, OHLCV DataFrames, timestamps, price anomalies,
and attaches data freshness metadata (data_age_seconds, is_live, is_stale).
"""

import logging
import datetime
from datetime import timezone, timedelta
from typing import Dict, Any, Tuple, Optional
import pandas as pd

logger = logging.getLogger("MarketDataValidator")
IST = timezone(timedelta(hours=5, minutes=30))

MAX_STALE_SECONDS = 300  # 5 minutes threshold for quote freshness flagging
MAX_PRICE_JUMP_RATIO = 0.50  # 50% single-tick anomaly threshold


class MarketDataValidator:
    """
    Validates market data quotes and historical DataFrames before downstream consumption.
    Rejects negative/zero prices, high < low anomalies, corrupted timestamps, and huge spikes.
    """

    @staticmethod
    def validate_quote(quote: Dict[str, Any], max_age_seconds: int = MAX_STALE_SECONDS) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Validates a single market quote record.
        Returns: (is_valid: bool, validated_quote: dict or None, reason: str)
        """
        if not quote or not isinstance(quote, dict):
            return False, None, "Quote payload is empty or invalid type"

        symbol = str(quote.get("ticker") or quote.get("symbol") or "").strip().upper()
        if not symbol:
            return False, None, "Missing stock ticker symbol"

        try:
            ltp = float(quote.get("ltp", 0.0))
            open_p = float(quote.get("open", ltp))
            high_p = float(quote.get("high", ltp))
            low_p = float(quote.get("low", ltp))
            prev_close = float(quote.get("previous_close", ltp))
        except (ValueError, TypeError) as e:
            return False, None, f"Non-numeric price field: {e}"

        # 1. Price non-positivity check
        if ltp <= 0:
            return False, None, f"Invalid zero or negative LTP ({ltp}) for {symbol}"

        # 2. High vs Low sanity check
        if high_p > 0 and low_p > 0 and high_p < low_p:
            return False, None, f"Invalid price range: High ({high_p}) < Low ({low_p}) for {symbol}"

        # 3. Abnormal price jump check vs previous close
        if prev_close > 0:
            pct_change = abs((ltp - prev_close) / prev_close)
            if pct_change > MAX_PRICE_JUMP_RATIO:
                logger.warning(f"Abnormal price jump detected for {symbol}: LTP={ltp}, PrevClose={prev_close} ({pct_change*100:.1f}%)")

        # 4. Calculate Data Freshness Metadata
        ts_str = quote.get("timestamp")
        data_age_seconds = 0
        is_live = False
        is_stale = False

        if ts_str:
            try:
                if isinstance(ts_str, datetime.datetime):
                    quote_dt = ts_str
                else:
                    quote_dt = datetime.datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))

                if quote_dt.tzinfo is None:
                    quote_dt = quote_dt.replace(tzinfo=IST)

                now_dt = datetime.datetime.now(IST)
                data_age_seconds = max(0, int((now_dt - quote_dt).total_seconds()))
                is_stale = data_age_seconds > max_age_seconds
                is_live = data_age_seconds <= 60
            except Exception as e:
                logger.debug(f"Timestamp parsing error for {symbol}: {e}")
                data_age_seconds = 0
                is_stale = False
                is_live = True

        validated = dict(quote)
        validated["ticker"] = symbol
        validated["ltp"] = round(ltp, 2)
        validated["open"] = round(open_p if open_p > 0 else ltp, 2)
        validated["high"] = round(high_p if high_p > 0 else ltp, 2)
        validated["low"] = round(low_p if low_p > 0 else ltp, 2)
        validated["previous_close"] = round(prev_close if prev_close > 0 else ltp, 2)
        validated["data_age_seconds"] = data_age_seconds
        validated["is_live"] = is_live
        validated["is_stale"] = is_stale
        validated["freshness_label"] = "Fresh" if not is_stale else "⚠ Data may be delayed"

        return True, validated, "OK"

    @staticmethod
    def validate_df(df: pd.DataFrame, ticker: str = "") -> Tuple[bool, Optional[pd.DataFrame], str]:
        """
        Validates and cleans historical/intraday OHLCV DataFrame.
        Ensures required columns exist, drops NaN, removes duplicate dates, and ensures positive values.
        """
        if df is None or df.empty:
            return False, None, "DataFrame is empty or None"

        cleaned = df.copy()

        # Handle MultiIndex columns if present
        if isinstance(cleaned.columns, pd.MultiIndex):
            cleaned.columns = cleaned.columns.get_level_values(0)

        # Standardize column names
        cols_map = {col: str(col).strip().capitalize() for col in cleaned.columns}
        cleaned.rename(columns=cols_map, inplace=True)

        required_cols = ["Close"]
        missing = [c for c in required_cols if c not in cleaned.columns]
        if missing:
            return False, None, f"Missing required columns {missing} in DataFrame"

        # Ensure Date column or Index
        if "Date" not in cleaned.columns:
            if isinstance(cleaned.index, pd.DatetimeIndex):
                cleaned.reset_index(inplace=True)
                cleaned.rename(columns={"index": "Date"}, inplace=True)
            elif "index" in cleaned.columns:
                cleaned.rename(columns={"index": "Date"}, inplace=True)

        # Drop rows with NaN in Close
        cleaned.dropna(subset=["Close"], inplace=True)

        # Filter out non-positive prices
        cleaned = cleaned[cleaned["Close"] > 0]

        # Fix High < Low anomalies if present
        if "High" in cleaned.columns and "Low" in cleaned.columns:
            cleaned = cleaned[cleaned["High"] >= cleaned["Low"]]

        # Drop duplicate dates
        if "Date" in cleaned.columns:
            cleaned.drop_duplicates(subset=["Date"], keep="last", inplace=True)
            cleaned.sort_values(by="Date", inplace=True)

        if cleaned.empty:
            return False, None, f"No valid rows remaining after validation for {ticker}"

        return True, cleaned, "OK"
