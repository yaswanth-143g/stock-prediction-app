import os
import json
import pandas as pd
import yfinance as yf

print("=== PHASE A4 ITEM 5: DATA FLAGS & ELIGIBILITY AUDIT ===")

CLEAN_DIR = "price_data_clean"
NOW = pd.Timestamp("2026-10-09")

jumps = ["ABFRL.NS", "ALLCARGO.NS", "CDSL.NS", "INDIAMART.NS", "INDUSINDBK.NS"]
stale = ["CENTURYTEX.NS", "GSPL.NS", "GUJGASLTD.NS", "ITDCEM.NS", "JBCHEPHARM.NS", "JKBANK.NS", "LTIM.NS"]

# 1. Jump Tickers Audit
print("\n--- 1. JUMP TICKERS AUDIT ---")
for t in jumps:
    fpath = os.path.join(CLEAN_DIR, f"{t}.csv")
    if not os.path.exists(fpath):
        print(f"{t}: File missing")
        continue
    df = pd.read_csv(fpath)
    df["Date"] = pd.to_datetime(df["Date"])
    df["pct_change"] = df["Close"].pct_change()
    max_jump_idx = df["pct_change"].abs().idxmax()
    jump_row = df.iloc[max_jump_idx]
    prev_row = df.iloc[max_jump_idx - 1]
    j_date = jump_row["Date"].strftime("%Y-%m-%d")
    j_pct = jump_row["pct_change"] * 100.0
    
    # Check yfinance auto_adjust=True
    yf_verdict = "Unknown"
    try:
        yf_df = yf.Ticker(t).history(period="max", auto_adjust=True)
        if not yf_df.empty:
            yf_df.index = pd.to_datetime(yf_df.index).tz_localize(None)
            if j_date in yf_df.index.strftime("%Y-%m-%d"):
                yf_row_idx = yf_df.index.get_loc(j_date)
                if isinstance(yf_row_idx, slice):
                    yf_row_idx = yf_row_idx.start
                if yf_row_idx > 0:
                    yf_jump = (yf_df["Close"].iloc[yf_row_idx] - yf_df["Close"].iloc[yf_row_idx-1]) / yf_df["Close"].iloc[yf_row_idx-1] * 100.0
                    yf_verdict = f"yfinance auto_adjust jump on same date = {yf_jump:+.2f}%"
    except Exception as e:
        yf_verdict = f"yfinance fetch error: {e}"

    print(f"Ticker: {t:<15} | Date: {j_date} | Local Jump: {prev_row['Close']:.2f} -> {jump_row['Close']:.2f} ({j_pct:+.2f}%)")
    print(f"  -> Explanation & Verdict: {yf_verdict}")

# 2. Stale Tickers Audit
print("\n--- 2. STALE TICKERS AUDIT ---")
for t in stale:
    fpath = os.path.join(CLEAN_DIR, f"{t}.csv")
    if not os.path.exists(fpath):
        print(f"{t}: File missing")
        continue
    df = pd.read_csv(fpath)
    df["Date"] = pd.to_datetime(df["Date"])
    last_d = df["Date"].max().strftime("%Y-%m-%d")
    
    # Check if active or delisted
    status = "Active"
    try:
        yf_hist = yf.Ticker(t).history(period="5d", auto_adjust=True)
        if yf_hist.empty:
            status = "Delisted / Merged / Suspended"
        else:
            yf_last_d = yf_hist.index[-1].strftime("%Y-%m-%d")
            status = f"yfinance active, latest quote date = {yf_last_d}"
    except Exception as e:
        status = f"yfinance check error: {e}"

    print(f"Ticker: {t:<15} | Last Local Date: {last_d} | Status: {status}")

# 3. Absent Tickers Audit from merge-csv.csv
print("\n--- 3. ABSENT TICKERS FROM MERGE-CSV.CSV AUDIT ---")
merge_csv_path = "merge-csv.csv"
if os.path.exists(merge_csv_path):
    df_merge = pd.read_csv(merge_csv_path)
    merge_tickers = set(df_merge["Ticker"].unique())
else:
    merge_tickers = set()

all_clean_files = [f for f in os.listdir(CLEAN_DIR) if f.endswith(".csv")]
all_clean_tickers = [f.replace(".csv", "") for f in all_clean_files]

absent_tickers = [t for t in all_clean_tickers if t not in merge_tickers]
print(f"Total tickers in price_data_clean/ absent from merge-csv.csv: {len(absent_tickers)}")

absent_summary = []
for t in absent_tickers:
    fpath = os.path.join(CLEAN_DIR, f"{t}.csv")
    df = pd.read_csv(fpath)
    rows = len(df)
    last_d = df["Date"].max() if "Date" in df.columns and not df.empty else "N/A"
    absent_summary.append((t, rows, last_d))

for t, r, d in sorted(absent_summary):
    print(f"Absent Ticker: {t:<15} | History Rows: {r:<6} | Last Date: {d}")

# 4. Filter & Produce eligible_tickers.json
print("\n--- 4. PRODUCING ELIGIBLE_TICKERS.JSON ---")
eligible = []
ineligible_reasons = {}

for f in sorted(all_clean_files):
    t = f.replace(".csv", "")
    fpath = os.path.join(CLEAN_DIR, f)
    df = pd.read_csv(fpath)
    if df.empty or "Date" not in df.columns or "Close" not in df.columns:
        ineligible_reasons[t] = "Empty/Invalid CSV"
        continue
    
    df["Date"] = pd.to_datetime(df["Date"])
    rows = len(df)
    last_date = df["Date"].max()
    days_old = (NOW - last_date).days
    
    df["pct_change"] = df["Close"].pct_change().abs()
    jumps_count = (df["pct_change"] >= 0.40).sum()
    
    reasons = []
    if rows < 750:
        reasons.append(f"Insufficient history ({rows} rows < 750)")
    if days_old > 5:
        reasons.append(f"Stale last date ({last_date.strftime('%Y-%m-%d')})")
    if jumps_count > 0:
        reasons.append(f"{jumps_count} unexplained jump(s) >= 40%")
        
    if not reasons:
        eligible.append(t)
    else:
        ineligible_reasons[t] = "; ".join(reasons)

print(f"Total Tickers Audited : {len(all_clean_files)}")
print(f"Eligible Tickers      : {len(eligible)}")
print(f"Ineligible Tickers    : {len(ineligible_reasons)}")

out_path = os.path.join(CLEAN_DIR, "eligible_tickers.json")
with open(out_path, "w") as f:
    json.dump({"eligible_tickers": eligible, "count": len(eligible), "reasons": ineligible_reasons}, f, indent=2)

print(f"Saved eligible tickers list to {out_path}")
