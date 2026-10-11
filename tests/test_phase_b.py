"""
tests/test_phase_b.py — Unit Tests for Phase B Raw Headline Logger
===================================================================
"""

import os
import sys
import sqlite3
import pytest
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from news_logger import (
    init_news_tables,
    generate_headline_hash,
    parse_to_utc_datetime,
    log_raw_headline,
    update_daily_sentiment_summary,
    get_daily_sentiment_history
)
from db import get_sqlite_conn


def test_table_creation():
    """Verify raw_news and daily_sentiment_summary tables exist."""
    init_news_tables()
    conn = get_sqlite_conn()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='raw_news';")
    assert cursor.fetchone() is not None, "raw_news table must exist"

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_sentiment_summary';")
    assert cursor.fetchone() is not None, "daily_sentiment_summary table must exist"

    conn.close()


def test_sha256_deduplication_hash():
    """Verify SHA-256 hash generation is case-insensitive and deterministic."""
    h1 = generate_headline_hash("RELIANCE.NS", "Reliance Industries quarterly profits rise 15%")
    h2 = generate_headline_hash("reliance.ns", "  Reliance Industries Quarterly Profits Rise 15%  ")
    h3 = generate_headline_hash("TCS.NS", "Reliance Industries quarterly profits rise 15%")

    assert h1 == h2, "Hashes must match for identical ticker and normalized headline text"
    assert h1 != h3, "Hashes must differ for different tickers"
    assert len(h1) == 64, "SHA-256 hash must be 64 characters long"


def test_utc_date_parsing():
    """Verify parsing arbitrary date strings into UTC datetimes."""
    dt1, time_known_1 = parse_to_utc_datetime("Fri, 09 Oct 2026 12:00:00 GMT")
    assert dt1.tzinfo == timezone.utc
    assert dt1.year == 2026
    assert time_known_1 == 1

    dt2, time_known_2 = parse_to_utc_datetime("2026-10-09 17:30:00")
    assert dt2.tzinfo == timezone.utc
    assert time_known_2 == 1


def test_log_headline_and_confidence_weighted_aggregation():
    """Verify raw headline logging and confidence-weighted daily sentiment summary math."""
    test_ticker = "TEST_PHASEB.NS"
    pub_date = "2026-10-09 10:00:00"

    # Headline 1: Score = +0.80, Confidence = 0.90
    log_raw_headline(
        ticker=test_ticker,
        headline="Test Company reports record Q3 earnings growth",
        sentiment_score=0.80,
        confidence=0.90,
        source="TestRSS",
        published_date_str=pub_date
    )

    # Headline 2: Score = -0.40, Confidence = 0.50
    log_raw_headline(
        ticker=test_ticker,
        headline="Test Company faces minor supply chain delay",
        sentiment_score=-0.40,
        confidence=0.50,
        source="TestRSS",
        published_date_str=pub_date
    )

    # Duplicate insertion attempt (must be skipped)
    inserted_dup = log_raw_headline(
        ticker=test_ticker,
        headline="Test Company reports record Q3 earnings growth",
        sentiment_score=0.80,
        confidence=0.90,
        source="TestRSS",
        published_date_str=pub_date
    )
    assert not inserted_dup, "Duplicate headline must be skipped by log_raw_headline"

    # Expected weighted sentiment calculation:
    # S_daily = (0.80 * 0.90 + (-0.40) * 0.50) / (0.90 + 0.50)
    # S_daily = (0.72 - 0.20) / 1.40 = 0.52 / 1.40 = 0.371428... ~ 0.3714
    expected_weighted_sentiment = round((0.80 * 0.90 - 0.40 * 0.50) / (0.90 + 0.50), 4)

    history = get_daily_sentiment_history(test_ticker, days=1)
    assert len(history) > 0, "Daily summary history must contain entry"

    summary = history[0]
    assert summary["ticker"] == test_ticker
    assert summary["headline_count"] == 2
    assert abs(summary["weighted_sentiment"] - expected_weighted_sentiment) < 1e-3, (
        f"Expected weighted sentiment {expected_weighted_sentiment}, got {summary['weighted_sentiment']}"
    )
