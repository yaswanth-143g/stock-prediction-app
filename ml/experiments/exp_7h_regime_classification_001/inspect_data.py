"""
inspect_data.py — Data Quality Audit & Top 20 Liquid Stock Selection
=====================================================================
Analyzes merge-csv.csv, checks column structure, duplicate features,
date ranges, missing values, and selects top 20 liquid NSE stocks
strictly using pre-test historical data.
"""

import pandas as pd
import numpy as np

def run_data_audit(csv_path="merge-csv.csv"):
    print(f"Reading dataset: {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Raw shape: {df.shape}")
    print(f"Raw columns ({len(df.columns)}): {df.columns.tolist()}")

    # 1. Check duplicate column names
    col_counts = pd.Series(df.columns).value_counts()
    duplicate_cols = col_counts[col_counts > 1].index.tolist()
    print(f"Duplicate column names in raw CSV: {duplicate_cols}")

    # Explicit assertion check for raw Close
    close_count = df.columns.tolist().count("Close")
    print(f"Raw Close count: {close_count}")

    # Standardize column names if duplicates exist
    if duplicate_cols:
        df = df.loc[:, ~df.columns.duplicated()].copy()
        print(f"Shape after removing duplicate columns: {df.shape}")

    # Ensure required columns
    req_cols = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Date parsing & sorting
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    min_date = df["Date"].min()
    max_date = df["Date"].max()
    print(f"Total Date Range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    print(f"Total Tickers: {df['Ticker'].nunique()}")

    # Determine 80% chronological cutoff for Train+Val to prevent test set leakage in stock selection
    all_dates = sorted(df["Date"].unique())
    cutoff_idx = int(len(all_dates) * 0.80)
    cutoff_date = all_dates[cutoff_idx]
    print(f"80% Chronological Cutoff Date (pre-test): {pd.Timestamp(cutoff_date).strftime('%Y-%m-%d')}")

    # Filter pre-test historical data for liquidity selection
    hist_df = df[df["Date"] <= cutoff_date].copy()
    hist_df["Turnover"] = hist_df["Close"] * hist_df["Volume"]

    # Calculate average daily turnover and trading day count per ticker
    stock_stats = hist_df.groupby("Ticker").agg(
        trading_days=("Close", "count"),
        mean_turnover=("Turnover", "mean"),
        mean_close=("Close", "mean"),
        mean_volume=("Volume", "mean")
    ).reset_index()

    # Require minimum 500 trading days in historical period to avoid newly listed / sparse stocks
    min_trading_days = int(hist_df["Date"].nunique() * 0.50)
    filtered_stats = stock_stats[stock_stats["trading_days"] >= min_trading_days].copy()

    # Sort by mean daily turnover (INR volume * price)
    top20_df = filtered_stats.sort_values(by="mean_turnover", ascending=False).head(20)
    top20_tickers = top20_df["Ticker"].tolist()

    print("\n========================================================")
    print("SELECTED TOP 20 LIQUID NSE STOCKS (Pre-Test Historical Data):")
    print("========================================================")
    for idx, row in top20_df.reset_index(drop=True).iterrows():
        print(f"{idx+1:2d}. {row['Ticker']:<15} | Days: {row['trading_days']:4d} | Mean Close: INR {row['mean_close']:8.2f} | Mean Turnover: INR {row['mean_turnover']:14,.2f}")

    return df, top20_tickers

if __name__ == "__main__":
    df, top20 = run_data_audit()
