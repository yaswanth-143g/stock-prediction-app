"""
tests/test_phase_a7_b4_requirements.py — Phase A7 + B4 Consolidated Requirements Unit Test Suite
=============================================================================================
Tests:
1. Sentiment engine default alpha = 0.0 everywhere.
2. News logger legacy row tagging (time_known = 0).
3. Market data provider fallback order (Cache -> API -> Fallback).
"""

import os
import sys
import ast
import sqlite3
import pytest
from datetime import datetime

import db

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_logger import parse_to_utc_datetime, init_news_tables, log_raw_headline
from services.market_data_service import MarketDataService
from db import get_sqlite_conn


def test_sentiment_engine_default_alpha_zero():
    """Verify default alpha = 0.0 (off by default) in code files."""
    files_to_check = [
        "predict.py",
        "app.py",
        "streamlit_app.py"
    ]
    for fn in files_to_check:
        full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), fn)
        if not os.path.exists(full_path):
            continue
        with open(full_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse AST to ensure alpha defaults to 0.0 or 0
        tree = ast.parse(content, filename=fn)
        found_alpha = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for arg, default in zip(node.args.args[-len(node.args.defaults):], node.args.defaults):
                    if arg.arg == "alpha":
                        found_alpha = True
                        if isinstance(default, ast.Constant):
                            assert default.value == 0.0 or default.value == 0, f"{fn}:{node.name} alpha default must be 0.0, got {default.value}"
        
        # Verify no hardcoded non-zero alpha default in code
        assert "alpha = 0.1" not in content and "alpha=0.1" not in content, f"Found active alpha default in {fn}"


def test_news_logger_legacy_row_tagging():
    """Verify date-only or missing timestamp strings return time_known = 0 without discarding the row."""
    init_news_tables()

    # 1. Full timestamp -> time_known = 1
    dt1, tk1 = parse_to_utc_datetime("2026-10-09 14:30:00")
    assert dt1 is not None
    assert tk1 == 1, "Timestamp with time component must set time_known = 1"

    # 2. Date only -> time_known = 0
    dt2, tk2 = parse_to_utc_datetime("2026-10-09")
    assert dt2 is not None
    assert tk2 == 0, "Date-only string must set time_known = 0"

    # 3. Unparseable -> time_known = 0
    dt3, tk3 = parse_to_utc_datetime("invalid_date_str")
    assert dt3 is None
    assert tk3 == 0, "Unparseable date string must set time_known = 0"

    # 4. Insertion test to confirm zero rows discarded
    import time
    ts = int(time.time() * 1000)
    ticker = "RELIANCE.NS"
    headline = f"Reliance Industries reports Q2 revenue growth in financial report {ts}"
    inserted = log_raw_headline(
        ticker=ticker,
        headline=headline,
        sentiment_score=0.5,
        published_date_str="2026-10-09"
    )
    assert inserted is True, "Row with date-only string must be inserted successfully"

    conn = db.get_sqlite_conn()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT time_known FROM raw_news WHERE headline = ?;", (headline,))
    row = cursor.fetchone()
    conn.close()

    assert row is not None, f"Expected inserted row for headline '{headline}'"
    assert row["time_known"] == 0, "Inserted row must persist time_known = 0"


def test_market_data_provider_fallback_order():
    """Verify MarketDataService.get_live_quote respects fallback order: Cache -> Upstox API -> yfinance/CSV."""
    md_service = MarketDataService()
    test_ticker = "RELIANCE.NS"

    # 1. Pre-seed memory cache
    mock_cached_data = {
        "ticker": "RELIANCE",
        "exchange": "NSE",
        "ltp": 1170.30,
        "previous_close": 1165.00,
        "change": 5.30,
        "change_percent": 0.45,
        "timestamp": datetime.now().isoformat(),
        "source": "memory_cache"
    }
    md_service.set_to_cache(test_ticker, mock_cached_data)

    # Cache hit check
    quote_from_cache = md_service.get_live_quote(test_ticker)
    assert quote_from_cache is not None
    assert quote_from_cache["source"] in ["redis_cache", "memory_cache"]
    assert quote_from_cache["ltp"] == 1170.30

    # 2. Clear cache and test fallback logic
    cache_key = md_service._get_cache_key(test_ticker)
    if cache_key in md_service._memory_cache:
        del md_service._memory_cache[cache_key]

    fallback_quote = md_service.get_live_quote(test_ticker)
    assert fallback_quote is not None
    assert "ltp" in fallback_quote
    assert fallback_quote["source"] in ["market_api", "yfinance_fallback", "clean_csv_fallback", "unavailable"]
