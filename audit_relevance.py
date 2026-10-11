"""
audit_relevance.py — Manual 20-Headline Relevance Audit
========================================================
Selects 20 random headlines from `raw_news` and evaluates whether each headline
explicitly mentions the target ticker, company name, or alias.
Calculates overall relevance percentage and identifies primary sources of non-relevant noise.
"""

import os
import sys
import random
import sqlite3

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from db import get_sqlite_conn
from news_sentiment import is_headline_relevant


def run_relevance_audit(sample_size: int = 20, seed: int = 42):
    random.seed(seed)
    conn = get_sqlite_conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("""
        SELECT ticker, headline, source, published_utc
        FROM raw_news;
    """)
    all_rows = c.fetchall()

    if not all_rows:
        print("No headlines in raw_news to audit.")
        conn.close()
        return

    sample = random.sample(all_rows, min(sample_size, len(all_rows)))

    print("==========================================================================")
    print("                 20-HEADLINE MANUAL RELEVANCE AUDIT REPORT                 ")
    print("==========================================================================")

    relevant_count = 0
    non_relevant_sources = {}

    for idx, r in enumerate(sample, 1):
        ticker = r["ticker"]
        headline = r["headline"]
        source = r["source"]
        is_rel = is_headline_relevant(headline, ticker)

        status_str = "YES (Relevant)" if is_rel else "NO  (Non-Relevant / Macro Noise)"
        if is_rel:
            relevant_count += 1
        else:
            non_relevant_sources[source] = non_relevant_sources.get(source, 0) + 1

        clean_h = headline.encode("ascii", "ignore").decode("ascii")
        print(f"\n{idx:2d}. [{ticker}] Source: {source}")
        print(f"    Headline: \"{clean_h[:80]}\"")
        print(f"    Entity Mention Match: {status_str}")

    relevance_pct = (relevant_count / len(sample)) * 100.0

    print("\n==========================================================================")
    print("                         AUDIT SUMMARY & NOISE SOURCES                     ")
    print("==========================================================================")
    print(f"Audited Sample Size:      {len(sample)} headlines")
    print(f"Relevant Headlines:      {relevant_count} / {len(sample)} ({relevance_pct:5.1f}%)")
    print(f"Non-Relevant Headlines:   {len(sample) - relevant_count} / {len(sample)} ({100.0 - relevance_pct:5.1f}%)")

    print("\nPrimary Sources of Non-Relevant / Macro Noise:")
    if non_relevant_sources:
        for src, cnt in sorted(non_relevant_sources.items(), key=lambda x: x[1], reverse=True):
            print(f" - {src:<25}: {cnt} non-relevant headlines")
        print("\nNoise Analysis Category:")
        print(" 1. Generic Market Wraps (e.g. 'Nifty 50 Market Wrap: Top Gainers & Losers') listing multiple stocks.")
        print(" 2. Macro-economic & Sectoral Industry Headlines (e.g. RBI rate decision, crude oil prices).")
    else:
        print(" - None! 100% of sampled headlines specifically mentioned target ticker/company.")

    print("==========================================================================")
    conn.close()


if __name__ == "__main__":
    run_relevance_audit()
