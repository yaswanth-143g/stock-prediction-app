"""
news_health.py — CLI Health Diagnostic Tool for Stock News Sentiment Pipeline
============================================================================
Usage:
    python news_health.py

Prints raw news statistics:
- Total rows
- Minimum & Maximum scrape / publication dates
- Daily row counts since first scrape date (flags 0-row days after first scrape date)
- Rows per ticker per day metrics
- Relevance metric (% headlines mentioning ticker or company name)
- % rows with time_known = 1 vs time_known = 0
- % rows scored by FinBERT vs Lexicon
- Source-wise time_known breakdown
"""

import os
import sys
import json
import re
import sqlite3
from datetime import datetime, timezone, timedelta

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from db import get_sqlite_conn
from news_sentiment import is_headline_relevant


def print_news_health():
    conn = get_sqlite_conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    print("==========================================================")
    print("        STOCK NEWS SENTIMENT PIPELINE HEALTH REPORT       ")
    print("==========================================================")

    # 1. Total Rows & Date Bounds
    c.execute("SELECT COUNT(*), MIN(DATE(scraped_at_utc)), MAX(DATE(scraped_at_utc)), MIN(DATE(published_utc)), MAX(DATE(published_utc)) FROM raw_news;")
    row = c.fetchone()
    total_rows = row[0]
    min_scrape = row[1]
    max_scrape = row[2]
    min_pub = row[3]
    max_pub = row[4]

    print(f"\n[SUMMARY] TOTAL RAW NEWS ROWS: {total_rows}")
    print(f"   Scrape Date Range:      {min_scrape} to {max_scrape}")
    print(f"   Publication Date Range: {min_pub} to {max_pub}")

    if total_rows == 0:
        print("[WARNING] No headlines found in raw_news database.")
        conn.close()
        return

    # Determine first active scrape date for zero-row filtering
    first_scrape_date_str = min_scrape or min_pub or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        first_scrape_date = datetime.strptime(first_scrape_date_str, "%Y-%m-%d").date()
    except Exception:
        first_scrape_date = datetime.now(timezone.utc).date()

    today = datetime.now(timezone.utc).date()
    days_span = max(1, (today - first_scrape_date).days + 1)

    # 2. Daily Row Counts (Only reporting days AFTER first scrape date)
    c.execute("""
        SELECT DATE(scraped_at_utc) as scrape_date, COUNT(*) as cnt
        FROM raw_news
        WHERE scraped_at_utc IS NOT NULL
        GROUP BY DATE(scraped_at_utc)
        ORDER BY scrape_date DESC;
    """)
    daily_counts = {r["scrape_date"]: r["cnt"] for r in c.fetchall()}

    print(f"\n[DAILY] ROW COUNTS SINCE FIRST SCRAPE DATE ({first_scrape_date_str} to {today}):")
    zero_days = []
    for i in range(days_span):
        d_date = today - timedelta(days=i)
        d_str = d_date.strftime("%Y-%m-%d")
        cnt = daily_counts.get(d_str, 0)
        status = "[ZERO ROWS!]" if cnt == 0 else f"{cnt:3d} rows"
        print(f"   {d_str}: {status}")
        if cnt == 0:
            zero_days.append(d_str)

    if zero_days:
        print(f"\n[ALERT] DAYS WITH ZERO ROWS SINCE PIPELINE LAUNCH ({len(zero_days)} days): {', '.join(zero_days)}")
    else:
        print("\n[OK] All days since first scrape date have active headlines logged.")

    # 3. Rows Per Ticker & Rows Per Ticker Per Day
    c.execute("""
        SELECT ticker, COUNT(*) as cnt
        FROM raw_news
        GROUP BY ticker
        ORDER BY cnt DESC;
    """)
    ticker_counts = c.fetchall()
    num_tickers = max(1, len(ticker_counts))
    avg_per_ticker_per_day = total_rows / (num_tickers * days_span)

    print(f"\n[TICKERS] ROWS PER TICKER ({num_tickers} Active Tickers, Span: {days_span} days):")
    print(f"   Average Rows / Ticker / Day: {avg_per_ticker_per_day:.2f} rows/ticker/day")
    for r in ticker_counts[:15]:
        r_per_day = r['cnt'] / days_span
        print(f"   {r['ticker']:<18}: {r['cnt']:4d} rows ({r_per_day:.2f} rows/day)")
    if len(ticker_counts) > 15:
        print(f"   ... and {len(ticker_counts) - 15} more tickers.")

    # 4. Relevance Metric (% headlines matching ticker or company name)
    c.execute("SELECT ticker, headline FROM raw_news;")
    all_headlines = c.fetchall()
    relevant_cnt = sum(1 for h in all_headlines if is_headline_relevant(h["headline"], h["ticker"]))
    relevance_pct = (relevant_cnt / total_rows * 100.0) if total_rows > 0 else 0.0

    print(f"\n[RELEVANCE] COMPANY KEYWORD METRIC:")
    print(f"   Relevant Headlines:   {relevant_cnt:4d} / {total_rows} rows ({relevance_pct:5.1f}%)")
    print(f"   Non-Relevant/General: {total_rows - relevant_cnt:4d} / {total_rows} rows ({100.0 - relevance_pct:5.1f}%)")

    # 5. % Rows with time_known = 1
    c.execute("SELECT COUNT(*) FROM raw_news WHERE time_known = 1;")
    time_known_cnt = c.fetchone()[0]
    time_known_pct = (time_known_cnt / total_rows * 100.0) if total_rows > 0 else 0.0

    c.execute("SELECT COUNT(*) FROM raw_news WHERE time_known = 0;")
    time_unknown_cnt = c.fetchone()[0]
    time_unknown_pct = (time_unknown_cnt / total_rows * 100.0) if total_rows > 0 else 0.0

    print(f"\n[TIMESTAMP] PRECISION (time_known):")
    print(f"   Full Time Known (time_known = 1): {time_known_cnt:4d} rows ({time_known_pct:5.1f}%)")
    print(f"   Date Only / Unknown (time_known = 0): {time_unknown_cnt:4d} rows ({time_unknown_pct:5.1f}%)")

    # Source-wise breakdown of time_known = 0
    c.execute("""
        SELECT source, COUNT(*) as total_src,
               SUM(CASE WHEN time_known = 0 THEN 1 ELSE 0 END) as zero_time_cnt
        FROM raw_news
        GROUP BY source;
    """)
    src_stats = c.fetchall()
    print("   Source breakdown for date-only (time_known = 0):")
    for s in src_stats:
        print(f"     - {s['source']:<18}: {s['zero_time_cnt']} / {s['total_src']} rows date-only")

    # 6. % Scored by FinBERT vs Lexicon
    c.execute("SELECT COUNT(*) FROM raw_news WHERE scorer_used = 'finbert';")
    finbert_cnt = c.fetchone()[0]
    finbert_pct = (finbert_cnt / total_rows * 100.0) if total_rows > 0 else 0.0

    c.execute("SELECT COUNT(*) FROM raw_news WHERE scorer_used = 'lexicon';")
    lexicon_cnt = c.fetchone()[0]
    lexicon_pct = (lexicon_cnt / total_rows * 100.0) if total_rows > 0 else 0.0

    print(f"\n[PROVENANCE] SENTIMENT ENGINE:")
    print(f"   FinBERT Scored: {finbert_cnt:4d} rows ({finbert_pct:5.1f}%)")
    print(f"   Lexicon Scored: {lexicon_cnt:4d} rows ({lexicon_pct:5.1f}%)")

    # 7. Scraper Runs Summary
    c.execute("SELECT * FROM scraper_runs ORDER BY run_id DESC LIMIT 3;")
    runs = c.fetchall()
    print(f"\n[SCRAPER RUNS] RECENT RUNS ({len(runs)} recorded):")
    for r in runs:
        print(f"   Run #{r['run_id']} [{r['started_at']} -> {r['finished_at']}]: Tickers OK={r['tickers_ok']}, Failed={r['tickers_failed']}, Inserted={r['rows_inserted']}")

    print("==========================================================")
    conn.close()


if __name__ == "__main__":
    print_news_health()
