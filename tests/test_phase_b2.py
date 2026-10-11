"""
tests/test_phase_b2.py — Unit Test Suite for Phase B2 Specification
====================================================================
"""

import os
import sys
import threading
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
    is_templated_fallback_headline,
    get_daily_sentiment_history
)
import db
from db import get_sqlite_conn


def test_duplicate_insert_ignored():
    """Verify duplicate headline insertion with identical headline_hash is ignored gracefully."""
    ticker = "TEST_DUP.NS"
    headline = "Company announces major expansion in renewable energy"
    pub_date = "Fri, 09 Oct 2026 10:00:00 GMT"

    init_news_tables()
    
    res1 = log_raw_headline(ticker=ticker, headline=headline, sentiment_score=0.75, published_date_str=pub_date)
    assert res1 is True, "First insertion must succeed"

    res2 = log_raw_headline(ticker=ticker, headline=headline, sentiment_score=0.75, published_date_str=pub_date)
    assert res2 is False, "Duplicate insertion with identical SHA-256 hash must be skipped"


def test_fallback_news_never_stored():
    """Verify fallback/templated news items are completely blocked from raw_news DB storage."""
    init_news_tables()
    ticker = "TEST_FALLBACK.NS"
    fallback_headline = "Market sentiment remains neutral for TEST_FALLBACK.NS"

    res = log_raw_headline(
        ticker=ticker,
        headline=fallback_headline,
        sentiment_score=0.0,
        published_date_str="2026-10-09 12:00:00",
        is_fallback=True
    )
    assert res is False, "Fallback headline must NEVER be inserted into raw_news"

    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM raw_news WHERE ticker = ?;", (ticker,))
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 0, "No fallback rows should exist in raw_news table"


def test_timestamp_parsing_variants_and_malformed_null():
    """Verify timestamp parsing for GMT, +0530, and malformed strings (expects NULL, not scrape time)."""
    # 1. GMT format
    dt_gmt, time_known_1 = parse_to_utc_datetime("Fri, 09 Oct 2026 12:00:00 GMT")
    assert dt_gmt is not None
    assert dt_gmt.year == 2026 and dt_gmt.hour == 12
    assert time_known_1 == 1

    # 2. +0530 IST offset format
    dt_ist, time_known_2 = parse_to_utc_datetime("Fri, 09 Oct 2026 17:30:00 +0530")
    assert dt_ist is not None
    assert dt_ist.year == 2026 and dt_ist.hour == 12  # 17:30 IST - 5:30 = 12:00 UTC
    assert time_known_2 == 1

    # 3. Malformed string -> must return (None, 0) (NULL in DB), NOT scrape time!
    dt_invalid, time_known_3 = parse_to_utc_datetime("invalid_date_string_12345")
    assert dt_invalid is None, "Malformed date string must return None (NULL date)"
    assert time_known_3 == 0


def test_score_provenance_lexicon_mode():
    """Verify score provenance when FinBERT is unavailable (lexicon score stored, finbert_score is NULL)."""
    init_news_tables()
    ticker = "TEST_PROVENANCE.NS"
    headline = "Quarterly profits jump 20 percent on strong export growth"

    inserted = log_raw_headline(
        ticker=ticker,
        headline=headline,
        sentiment_score=0.85,
        confidence=0.90,
        scorer_used="lexicon",
        finbert_score=None,
        lexicon_score=0.85,
        published_date_str="2026-10-09 14:00:00"
    )
    assert inserted is True, "Headline insertion must succeed"

    conn = db.get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT scorer_used, lexicon_score, finbert_score, final_score FROM raw_news WHERE ticker = ?;", (ticker,))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row["scorer_used"] == "lexicon"
    assert row["lexicon_score"] == 0.85
    assert row["finbert_score"] is None, "finbert_score must be NULL when FinBERT is not used"
    assert row["final_score"] == 0.85


def test_concurrent_writes_thread_safety():
    """Verify concurrent writes from multiple threads run without database locks under SQLite WAL mode."""
    init_news_tables()
    ticker_prefix = "TEST_THREAD_"
    errors = []

    def worker_thread(thread_id):
        try:
            for i in range(10):
                log_raw_headline(
                    ticker=f"{ticker_prefix}{thread_id}.NS",
                    headline=f"Thread {thread_id} headline iteration {i} on market trends",
                    sentiment_score=0.1 * i,
                    confidence=0.8,
                    published_date_str="2026-10-09 15:00:00"
                )
        except Exception as e:
            errors.append(e)

    t1 = threading.Thread(target=worker_thread, args=(1,))
    t2 = threading.Thread(target=worker_thread, args=(2,))

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(errors) == 0, f"Concurrent threads encountered errors: {errors}"
