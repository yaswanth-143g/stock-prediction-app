"""
phase_d_ready.py — Phase D Experiment Readiness Diagnostic
==========================================================
Audits the raw_news database under strict quantitative criteria:
1. Eligible tickers only (from price_data_clean/eligible_tickers.json)
2. Fresh headlines only (lag_hours <= 24)
3. Deduped headlines (unique dedupe_key per ticker-date)
4. Relevant headlines only (matching company name/alias keywords)

Computes:
- First & last date covered
- Total trading days covered
- Headlines per ticker (mean, min, max, top/bottom)
- Total ticker-days with at least one relevant headline
- Statistical power estimates for detecting IC = 0.03 and IC = 0.05
- Gating decision: READY if >= 120 trading days; else reports days remaining.
"""

import os
import sys
import json
import sqlite3
import math
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from db import get_sqlite_conn
from news_sentiment import is_headline_relevant


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_statistical_power(effect_size: float, n_samples: float, alpha: float = 0.05) -> float:
    """
    Computes two-tailed statistical power for detecting correlation / IC.
    Under H0: t ~ N(0, 1) with critical value z_crit = 1.96 for alpha=0.05.
    Under H1: non-centrality parameter delta = effect_size * sqrt(n_samples).
    """
    if n_samples <= 3 or effect_size <= 0:
        return 0.0
    z_crit = 1.95996  # for alpha = 0.05
    delta = effect_size * math.sqrt(n_samples)
    power = normal_cdf(delta - z_crit) + normal_cdf(-delta - z_crit)
    return max(0.0, min(1.0, power))


def check_phase_d_readiness():
    print("==========================================================================")
    print("             PHASE D SENTIMENT ALPHA READINESS DIAGNOSTIC                 ")
    print("==========================================================================")

    # 1. Load eligible tickers
    eligible_path = os.path.join(PROJECT_ROOT, "price_data_clean", "eligible_tickers.json")
    if not os.path.exists(eligible_path):
        print(f"[ERROR] eligible_tickers.json not found at {eligible_path}")
        return
    with open(eligible_path, "r") as f:
        eligible_set = set(json.load(f)["eligible_tickers"])

    conn = get_sqlite_conn()
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("""
        SELECT id, ticker, headline, published_utc, scraped_at_utc, lag_hours, dedupe_key
        FROM raw_news
        ORDER BY published_utc ASC, id ASC;
    """)
    rows = c.fetchall()
    conn.close()

    total_raw_rows = len(rows)

    # 2. Filter rows
    seen_dedupe_keys = set()
    valid_rows = []

    for r in rows:
        ticker = r["ticker"]
        # Must be eligible
        if ticker not in eligible_set:
            continue
        # lag_hours <= 24
        lag = r["lag_hours"]
        if lag is None or lag > 24.0:
            continue
        # Deduped
        d_key = r["dedupe_key"]
        if d_key:
            if d_key in seen_dedupe_keys:
                continue
            seen_dedupe_keys.add(d_key)
        # Relevance filter
        if not is_headline_relevant(r["headline"], ticker):
            continue

        valid_rows.append(r)

    print(f"Total raw_news rows in DB         : {total_raw_rows}")
    print(f"Valid Phase D headlines (clean)   : {len(valid_rows)}")
    print("Filters applied: eligible_tickers, lag_hours <= 24, deduped, is_headline_relevant")

    if not valid_rows:
        print("\n[STATUS] NOT READY (0 valid headlines meeting strict Phase D criteria).")
        print("Required: 120 trading days. Remaining: 120 days.")
        print("==========================================================================")
        return

    # 3. Date bounds & trading days
    dates = []
    ticker_counts = defaultdict(int)
    ticker_days = set()

    for r in valid_rows:
        pub = r["published_utc"] or r["scraped_at_utc"]
        d_str = pub.split(" ")[0].split("T")[0]
        dates.append(d_str)
        ticker_counts[r["ticker"]] += 1
        ticker_days.add((r["ticker"], d_str))

    dates_sorted = sorted(dates)
    unique_dates = sorted(list(set(dates)))
    min_date = dates_sorted[0]
    max_date = dates_sorted[-1]
    trading_days_covered = len(unique_dates)

    # 4. Ticker statistics
    counts_list = list(ticker_counts.values())
    avg_per_ticker = sum(counts_list) / len(counts_list) if counts_list else 0
    min_per_ticker = min(counts_list) if counts_list else 0
    max_per_ticker = max(counts_list) if counts_list else 0

    print(f"\n--- Coverage Metrics ---")
    print(f"First Date Covered                : {min_date}")
    print(f"Last Date Covered                 : {max_date}")
    print(f"Unique News Trading Days Covered  : {trading_days_covered} days")
    print(f"Total Unique Ticker-Days Covered  : {len(ticker_days)} ticker-days")
    print(f"Active Eligible Tickers with News : {len(ticker_counts)} tickers")
    print(f"Headlines Per Ticker (Mean)       : {avg_per_ticker:.2f}")
    print(f"Headlines Per Ticker (Min / Max)  : {min_per_ticker} / {max_per_ticker}")

    # Top tickers
    sorted_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 5 Tickers by Valid Headlines:")
    for t, cnt in sorted_tickers[:5]:
        print(f"  - {t:<15}: {cnt} headlines")

    # 5. Statistical Power Estimates
    # Form 1: Power across pooled ticker-day cross-section (N = total ticker-days)
    m_ticker_days = len(ticker_days)
    power_pooled_03 = compute_statistical_power(0.03, m_ticker_days)
    power_pooled_05 = compute_statistical_power(0.05, m_ticker_days)

    # Form 2: Power for daily cross-sectional Mean IC (T = trading days, SE = sigma_IC / sqrt(T))
    # Assuming standard empirical daily IC volatility sigma_IC ~ 0.15
    sigma_ic = 0.15
    effective_t_03 = (0.03 / sigma_ic) ** 2 * trading_days_covered
    effective_t_05 = (0.05 / sigma_ic) ** 2 * trading_days_covered
    power_daily_03 = compute_statistical_power(0.03 / sigma_ic, trading_days_covered)
    power_daily_05 = compute_statistical_power(0.05 / sigma_ic, trading_days_covered)

    print(f"\n--- Statistical Power Estimates (alpha = 0.05, two-tailed) ---")
    print(f"1. Pooled Ticker-Day Level (N = {m_ticker_days} observations):")
    print(f"   - Power to detect IC = 0.03    : {power_pooled_03 * 100:.1f}%")
    print(f"   - Power to detect IC = 0.05    : {power_pooled_05 * 100:.1f}%")
    print(f"2. Daily Cross-Sectional IC Level (T = {trading_days_covered} days, sigma_IC = {sigma_ic}):")
    print(f"   - Power to detect IC = 0.03    : {power_daily_03 * 100:.1f}%")
    print(f"   - Power to detect IC = 0.05    : {power_daily_05 * 100:.1f}%")

    # 6. Gating Verdict
    REQUIRED_TRADING_DAYS = 120
    print("\n==========================================================================")
    if trading_days_covered >= REQUIRED_TRADING_DAYS:
        print(f"[STATUS] READY (Criteria met: {trading_days_covered} >= {REQUIRED_TRADING_DAYS} trading days)")
    else:
        days_remaining = REQUIRED_TRADING_DAYS - trading_days_covered
        print(f"[STATUS] NOT READY (Trading days covered: {trading_days_covered} / {REQUIRED_TRADING_DAYS}. Days remaining: {days_remaining})")
        print(f"[EXPLANATION] Phase D statistical validity requires >= 120 distinct trading")
        print(f"             days of live news to reliably measure Information Coefficient.")
        print(f"             Currently have {trading_days_covered} days. News collector is actively running.")
    print("==========================================================================")


if __name__ == "__main__":
    check_phase_d_readiness()
