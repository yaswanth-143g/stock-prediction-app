"""
news_logger.py — Full Spec UTC-Aware Historical Raw News Logger & Aggregator (Phase B2)
========================================================================================
Manages persistent storage of raw financial news headlines in `raw_news`
and calculates confidence-weighted daily sentiment summaries in `daily_sentiment_summary`.
"""

import os
import re
import json
import sqlite3
import hashlib
import logging
import threading
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

try:
    from dateutil import parser as date_parser
except ImportError:
    date_parser = None

from db import get_sqlite_conn

logger = logging.getLogger("NewsLogger")
logger.setLevel(logging.INFO)

# Templated fallback headline phrases to detect and reject synthetic headlines
FALLBACK_PATTERNS = [
    r"market sentiment remains neutral for",
    r"no major news reported today",
    r"trading volume steady near average for",
    r"analysts maintain neutral stance on",
    r"quarterly financial reporting pending for",
    r"sector macro trends remain stable for"
]


def init_news_tables(conn=None):
    """Initializes raw_news, daily_sentiment_summary, and scraper_runs database tables and handles column migrations."""
    close_after = False
    if conn is None:
        conn = get_sqlite_conn()
        close_after = True

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            headline TEXT NOT NULL,
            source TEXT DEFAULT 'RSS',
            published_utc TEXT,
            published_raw TEXT,
            scraped_at_utc TEXT NOT NULL,
            headline_hash TEXT UNIQUE NOT NULL,
            finbert_sentiment_score REAL DEFAULT 0.0,
            finbert_confidence REAL DEFAULT 0.70,
            finbert_score REAL,
            lexicon_score REAL,
            final_score REAL NOT NULL DEFAULT 0.0,
            scorer_used TEXT NOT NULL DEFAULT 'lexicon',
            is_fallback INTEGER NOT NULL DEFAULT 0,
            time_known INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_news_hash ON raw_news (headline_hash);
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_raw_news_ticker_date ON raw_news (ticker, published_utc);
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_sentiment_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            date_utc TEXT NOT NULL,
            weighted_sentiment REAL NOT NULL,
            headline_count INTEGER NOT NULL DEFAULT 0,
            score_std REAL DEFAULT 0.0,
            sent_conf REAL NOT NULL DEFAULT 0.0,
            updated_at_utc TEXT NOT NULL,
            UNIQUE(ticker, date_utc)
        );
    """)

    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_sentiment_ticker_date ON daily_sentiment_summary (ticker, date_utc);
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scraper_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            tickers_ok INTEGER NOT NULL,
            tickers_failed INTEGER NOT NULL,
            rows_inserted INTEGER NOT NULL
        );
    """)

    # Auto-migrate table for existing databases missing Phase B2/B3 columns
    existing_cols = [r[1] for r in cursor.execute("PRAGMA table_info(raw_news);").fetchall()]
    migrations = [
        ("published_raw", "TEXT"),
        ("finbert_score", "REAL"),
        ("lexicon_score", "REAL"),
        ("final_score", "REAL DEFAULT 0.0"),
        ("scorer_used", "TEXT DEFAULT 'lexicon'"),
        ("is_fallback", "INTEGER DEFAULT 0"),
        ("time_known", "INTEGER DEFAULT 1"),
        ("dedupe_key", "TEXT"),
        ("lag_hours", "REAL")
    ]
    for col_name, col_type in migrations:
        if col_name not in existing_cols:
            try:
                cursor.execute(f"ALTER TABLE raw_news ADD COLUMN {col_name} {col_type};")
            except Exception as ex:
                logger.debug(f"Migration note ({col_name}): {ex}")

    summary_cols = [r[1] for r in cursor.execute("PRAGMA table_info(daily_sentiment_summary);").fetchall()]
    summary_migrations = [
        ("score_std", "REAL DEFAULT 0.0"),
        ("sent_conf", "REAL DEFAULT 0.0")
    ]
    for col_name, col_type in summary_migrations:
        if col_name not in summary_cols:
            try:
                cursor.execute(f"ALTER TABLE daily_sentiment_summary ADD COLUMN {col_name} {col_type};")
            except Exception as ex:
                logger.debug(f"Migration note summary ({col_name}): {ex}")

    conn.commit()
    if close_after:
        conn.close()


def generate_headline_hash(ticker: str, headline: str) -> str:
    """Generates a deterministic SHA-256 hash for ticker + headline deduplication."""
    clean_ticker = ticker.strip().upper()
    clean_headline = headline.strip().lower()
    raw_str = f"{clean_ticker}:{clean_headline}"
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()


def is_templated_fallback_headline(headline: str) -> bool:
    """Detects whether a headline matches synthetic fallback news templates."""
    if not headline:
        return True
    text_lower = headline.strip().lower()
    for pattern in FALLBACK_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def parse_to_utc_datetime(date_str: Optional[str]) -> tuple[Optional[datetime], int]:
    """
    Parses arbitrary date strings into a tuple of (UTC-aware datetime, time_known_flag [1 or 0]).
    
    Time Known Logic:
    - If explicit time components (HH:MM or HH:MM:SS) exist in date_str, time_known = 1.
    - If only date (YYYY-MM-DD or DD Mon YYYY) exists without time, time_known = 0.
    - If unparseable, returns (None, 0).
    """
    if not date_str or not isinstance(date_str, str) or not date_str.strip():
        return (None, 0)

    clean_str = date_str.strip()

    # Check if string contains time indicators (colon separating numbers)
    has_time = bool(re.search(r"\b\d{1,2}:\d{2}\b", clean_str))

    try:
        if date_parser:
            dt = date_parser.parse(clean_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return (dt, 1 if has_time else 0)
    except Exception:
        pass

    # Standard format fallbacks
    for fmt, specifies_time in [
        ("%a, %d %b %Y %H:%M:%S %z", True),
        ("%a, %d %b %Y %H:%M:%S GMT", True),
        ("%Y-%m-%dT%H:%M:%SZ", True),
        ("%Y-%m-%d %H:%M:%S", True),
        ("%d %b %Y %H:%M:%S", True),
        ("%Y-%m-%d", False),
        ("%d %b %Y", False)
    ]:
        try:
            dt = datetime.strptime(clean_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return (dt, 1 if specifies_time else 0)
        except Exception:
            continue

    return (None, 0)


def log_raw_headline(
    ticker: str,
    headline: str,
    sentiment_score: float,
    confidence: Optional[float] = None,
    source: str = "RSS",
    published_date_str: Optional[str] = None,
    scorer_used: str = "lexicon",
    finbert_score: Optional[float] = None,
    lexicon_score: Optional[float] = None,
    is_fallback: bool = False,
    conn=None
) -> bool:
    """
    Logs a single raw headline into raw_news table and updates daily_sentiment_summary.
    Never mutates or truncates headline text.
    BLOCKS FALLBACK NEWS: Rejected if is_fallback=True OR matches templated phrases.
    """
    if not headline or not ticker:
        return False

    # Primary check: explicit is_fallback flag; Secondary check: templated phrase matching
    if is_fallback or is_templated_fallback_headline(headline):
        logger.debug(f"Blocked fallback headline insertion for {ticker}")
        return False

    if confidence is None or confidence <= 0.0:
        confidence = float(max(0.50, min(1.00, abs(sentiment_score))))

    pub_dt, time_known = parse_to_utc_datetime(published_date_str)
    pub_utc_str = pub_dt.strftime("%Y-%m-%d %H:%M:%S") if pub_dt else None
    date_utc_str = pub_dt.strftime("%Y-%m-%d") if pub_dt else None
    scraped_dt = datetime.now(timezone.utc)
    scraped_utc_str = scraped_dt.strftime("%Y-%m-%d %H:%M:%S")

    # Lag hours & dedupe key calculation
    lag_hours = max(0.0, (scraped_dt - pub_dt).total_seconds() / 3600.0) if pub_dt else None
    norm_h = re.sub(r"\s*[-–—|]\s*(Yahoo Finance|Economic Times|Moneycontrol|Reuters|Bloomberg|CNBC|Mint|Business Today|Financial Express|NDTV|Upstox|Investing\.com|Times of India).*$", "", headline, flags=re.IGNORECASE)
    norm_h = re.sub(r"[^\w\s]", "", norm_h.lower()).strip()
    norm_h = re.sub(r"\s+", " ", norm_h)
    text_hash = hashlib.sha256(norm_h.encode("utf-8")).hexdigest()[:16]
    dedupe_key = f"{ticker.upper()}:{date_utc_str or 'NODATE'}:{text_hash}"

    h_hash = generate_headline_hash(ticker, headline)

    # Score provenance logic
    final_score = float(sentiment_score)
    if scorer_used == "finbert":
        f_score = float(finbert_score) if finbert_score is not None else final_score
        l_score = float(lexicon_score) if lexicon_score is not None else None
    else:
        f_score = None
        l_score = float(lexicon_score) if lexicon_score is not None else final_score

    logger.info(f"Scoring headline for {ticker} using {scorer_used}: '{headline[:40]}...' (Score: {final_score:+.2f})")

    close_after = False
    if conn is None:
        conn = get_sqlite_conn()
        close_after = True

    cursor = conn.cursor()
    inserted = False

    try:
        cursor.execute("""
            INSERT OR IGNORE INTO raw_news (
                ticker, headline, source, published_utc, published_raw, scraped_at_utc,
                headline_hash, finbert_sentiment_score, finbert_confidence,
                finbert_score, lexicon_score, final_score, scorer_used, is_fallback, time_known,
                dedupe_key, lag_hours
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ticker.upper(), headline, source, pub_utc_str, published_date_str, scraped_utc_str,
            h_hash, final_score, float(confidence),
            f_score, l_score, final_score, scorer_used, 0, time_known,
            dedupe_key, lag_hours
        ))
        
        if cursor.rowcount > 0:
            inserted = True
            if date_utc_str:
                update_daily_sentiment_summary(ticker.upper(), date_utc_str, conn=conn)

        conn.commit()
    except Exception as e:
        logger.warning(f"Error logging raw headline for {ticker}: {e}")
    finally:
        if close_after:
            conn.close()

    return inserted


def log_raw_headline_async(
    ticker: str,
    headline: str,
    sentiment_score: float,
    confidence: Optional[float] = None,
    source: str = "RSS",
    published_date_str: Optional[str] = None,
    scorer_used: str = "lexicon",
    finbert_score: Optional[float] = None,
    lexicon_score: Optional[float] = None,
    is_fallback: bool = False
):
    """Asynchronously logs raw headline in a background thread for 0ms request latency."""
    t = threading.Thread(
        target=log_raw_headline,
        kwargs={
            "ticker": ticker,
            "headline": headline,
            "sentiment_score": sentiment_score,
            "confidence": confidence,
            "source": source,
            "published_date_str": published_date_str,
            "scorer_used": scorer_used,
            "finbert_score": finbert_score,
            "lexicon_score": lexicon_score,
            "is_fallback": is_fallback
        },
        daemon=True
    )
    t.start()


def update_daily_sentiment_summary(ticker: str, date_utc: str, conn=None):
    """
    Recalculates confidence-weighted daily sentiment summary for (ticker, date_utc).
    
    Formulae:
        S_daily = sum(score_i * confidence_i) / sum(confidence_i)
        sent_conf = mean * n / (n + 3)   [Shrinkage score]
        score_std = std(score_i)
    
    NOTE ON ML FEATURE PIPELINE:
    This daily_sentiment_summary table is maintained strictly for UI display and analytics.
    All ML training and feature engineering MUST extract raw headlines directly from raw_news
    using the IST 15:30 cutoff trading-day alignment rule (not from this display table).
    """
    close_after = False
    if conn is None:
        conn = get_sqlite_conn()
        close_after = True

    cursor = conn.cursor()
    ticker_clean = ticker.upper()

    cursor.execute("""
        SELECT final_score, finbert_confidence
        FROM raw_news
        WHERE ticker = ? AND date(published_utc) = date(?) AND is_fallback = 0 AND published_utc IS NOT NULL
    """, (ticker_clean, date_utc))

    rows = cursor.fetchall()
    if not rows:
        if close_after:
            conn.close()
        return

    scores = [float(r[0]) for r in rows]
    confidences = [float(r[1]) for r in rows]
    n = len(scores)

    total_weight = sum(confidences)
    if total_weight > 0:
        weighted_sentiment = float(sum(s * c for s, c in zip(scores, confidences)) / total_weight)
    else:
        weighted_sentiment = float(sum(scores) / n)

    score_std = float(np.std(scores)) if n > 1 else 0.0
    sent_conf = float(weighted_sentiment * (n / (n + 3.0)))

    now_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute("""
        INSERT INTO daily_sentiment_summary (
            ticker, date_utc, weighted_sentiment, headline_count, score_std, sent_conf, updated_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, date_utc) DO UPDATE SET
            weighted_sentiment = excluded.weighted_sentiment,
            headline_count = excluded.headline_count,
            score_std = excluded.score_std,
            sent_conf = excluded.sent_conf,
            updated_at_utc = excluded.updated_at_utc
    """, (ticker_clean, date_utc, round(weighted_sentiment, 4), n, round(score_std, 4), round(sent_conf, 4), now_utc_str))

    conn.commit()

    # Update Stock_Data/sentiment_history.json
    save_to_sentiment_history_json(ticker_clean, date_utc, weighted_sentiment, n, score_std, sent_conf)

    if close_after:
        conn.close()


def save_to_sentiment_history_json(ticker: str, date_utc: str, weighted_sentiment: float, n: int, std: float, sent_conf: float):
    """Updates Stock_Data/sentiment_history.json with n, std, sent_conf in addition to existing keys."""
    os.makedirs("Stock_Data", exist_ok=True)
    hist_file = os.path.join("Stock_Data", "sentiment_history.json")

    data = {}
    if os.path.exists(hist_file):
        try:
            with open(hist_file, "r") as f:
                data = json.load(f)
        except Exception:
            data = {}

    if ticker not in data:
        data[ticker] = {}

    data[ticker][date_utc] = {
        "score": round(weighted_sentiment, 4),
        "headline_count": n,
        "n": n,
        "std": round(std, 4),
        "sent_conf": round(sent_conf, 4),
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        with open(hist_file, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as ex:
        logger.debug(f"JSON persistence error: {ex}")


def get_daily_sentiment_history(ticker: str, days: int = 30) -> List[Dict[str, Any]]:
    """Retrieves historical daily sentiment summaries for a ticker."""
    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT ticker, date_utc, weighted_sentiment, headline_count, score_std, sent_conf, updated_at_utc
        FROM daily_sentiment_summary
        WHERE ticker = ?
        ORDER BY date_utc DESC
        LIMIT ?
    """, (ticker.upper(), days))

    rows = cursor.fetchall()
    conn.close()

    return [{
        "ticker": r[0],
        "date_utc": r[1],
        "weighted_sentiment": r[2],
        "headline_count": r[3],
        "score_std": r[4] if len(r) > 4 else 0.0,
        "sent_conf": r[5] if len(r) > 5 else 0.0,
        "updated_at_utc": r[6] if len(r) > 6 else r[4]
    } for r in rows]


def log_scraper_run(started_at: str, finished_at: str, tickers_ok: int, tickers_failed: int, rows_inserted: int) -> int:
    """Logs a scraper execution record into scraper_runs table and returns run_id."""
    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO scraper_runs (started_at, finished_at, tickers_ok, tickers_failed, rows_inserted)
        VALUES (?, ?, ?, ?, ?)
    """, (started_at, finished_at, tickers_ok, tickers_failed, rows_inserted))
    conn.commit()
    run_id = cursor.lastrowid
    conn.close()
    return run_id


# Auto-initialize database tables on import
init_news_tables()
