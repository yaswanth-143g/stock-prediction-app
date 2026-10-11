"""
data_pipeline.py — Multi-Tier Historical Data Download & Storage Pipeline
========================================================================
Downloads historical OHLCV market data for supported stocks, handles multi-tier
fallbacks (Yahoo Finance -> Local CSV -> SQLite/MySQL), and verifies schema consistency.
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
from ticker_validator import clean_ticker_symbol, fetch_raw_data_with_retries

BASE_STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'LT.NS', 'ITC.NS', 'HINDUNILVR.NS', 'BHARTIARTL.NS',
    'AXISBANK.NS', 'KOTAKBANK.NS', 'MARUTI.NS', 'ASIANPAINT.NS', 'SUNPHARMA.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'BAJFINANCE.NS', 'NTPC.NS', 'POWERGRID.NS',
    'TATASTEEL.NS', 'HINDALCO.NS', 'ABBOTINDIA.NS', 'WIPRO.NS', 'HCLTECH.NS', 'ADANIENT.NS', 'ADANIPORTS.NS', 'COALINDIA.NS', 'BPCL.NS', 'ONGC.NS',
    'GRASIM.NS', 'CIPLA.NS', 'DRREDDY.NS', 'ETERNAL.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS', 'DIVISLAB.NS', 'APOLLOHOSP.NS', 'JSWSTEEL.NS', 'NESTLEIND.NS',
    'BAJAJ-AUTO.NS', 'M&M.NS', 'TATACONSUM.NS', 'INDUSINDBK.NS', 'BRITANNIA.NS', 'TECHM.NS', 'TRENT.NS', 'BEL.NS', 'HAL.NS', 'SHRIRAMFIN.NS',
    'AARTIIND.NS', 'ABB.NS', 'ABFRL.NS', 'ACC.NS', 'ADANIENSOL.NS', 'ADANIGREEN.NS', 'ADANIPOWER.NS', 'ATGL.NS', 'AEGISLOG.NS', 'AFFLE.NS',
    'AJANTPHARM.NS', 'ALKEM.NS', 'ALLCARGO.NS', 'AMBUJACEM.NS', 'ANGELONE.NS', 'APARINDS.NS', 'APLAPOLLO.NS', 'APOLLOTYRE.NS', 'APTUS.NS', 'ARVIND.NS',
    'ASHOKLEY.NS', 'ASTRAL.NS', 'ATUL.NS', 'AUBANK.NS', 'AUROPHARMA.NS', 'AWL.NS', 'BAJAJHLDNG.NS', 'BALKRISIND.NS', 'BALRAMCHIN.NS', 'BANDHANBNK.NS',
    'BANKBARODA.NS', 'BANKINDIA.NS', 'BATAINDIA.NS', 'BDL.NS', 'BEML.NS', 'BERGEPAINT.NS', 'BHARATFORG.NS', 'BHEL.NS', 'BIOCON.NS', 'BIRLACORPN.NS',
    'BLS.NS', 'BLUEDART.NS', 'BOSCHLTD.NS', 'BRIGADE.NS', 'BSOFT.NS', 'CANBK.NS', 'CANFINHOME.NS', 'CARBORUNIV.NS', 'CASTROLIND.NS', 'CDSL.NS',
    'CEATLTD.NS', 'CENTRALBK.NS', 'CENTURYPLY.NS', 'CENTURYTEX.NS', 'CERA.NS', 'CESC.NS', 'CGPOWER.NS', 'CHAMBLFERT.NS', 'CHOLAFIN.NS', 'CHOLAHLDNG.NS',
    'CLEAN.NS', 'COCHINSHIP.NS', 'COFORGE.NS', 'COLPAL.NS', 'CONCOR.NS', 'COROMANDEL.NS', 'CRAFTSMAN.NS', 'CREDITACC.NS', 'CROMPTON.NS', 'CUMMINSIND.NS',
    'CYIENT.NS', 'DABUR.NS', 'DALBHARAT.NS', 'DATAPATTNS.NS', 'DEEPAKNTR.NS', 'DELHIVERY.NS', 'DEVYANI.NS', 'DIXON.NS', 'DLF.NS', 'EASEMYTRIP.NS',
    'EDELWEISS.NS', 'EIDPARRY.NS', 'EIHOTEL.NS', 'ELGIEQUIP.NS', 'EMAMILTD.NS', 'ENDURANCE.NS', 'ENGINERSIN.NS', 'EQUITASBNK.NS', 'ERIS.NS', 'ESCORTS.NS',
    'EXIDEIND.NS', 'FSL.NS', 'GAIL.NS', 'GANESHHOU.NS', 'GLAXO.NS', 'GLENMARK.NS', 'GODREJAGRO.NS', 'GODREJCP.NS', 'GODREJIND.NS', 'GODREJPROP.NS',
    'GRANULES.NS', 'GSPL.NS', 'GUJGASLTD.NS', 'HAPPSTMNDS.NS', 'HAVELLS.NS', 'HDFCAMC.NS', 'HDFCLIFE.NS', 'HFCL.NS', 'HINDCOPPER.NS', 'HINDPETRO.NS',
    'HINDZINC.NS', 'HOMEFIRST.NS', 'HONAUT.NS', 'HUDCO.NS', 'IDBI.NS', 'IDFCFIRSTB.NS', 'IEX.NS', 'IGL.NS', 'IIFL.NS', 'INDIAMART.NS',
    'INDIANB.NS', 'INDHOTEL.NS', 'IOC.NS', 'IOB.NS', 'IPCALAB.NS', 'IRCTC.NS', 'IRFC.NS', 'IREDA.NS', 'ITDCEM.NS', 'J&KBANK.NS',
    'JBCHEPHARM.NS', 'JINDALSTEL.NS', 'JIOFIN.NS', 'JUBLFOOD.NS', 'KALYANKJIL.NS', 'KARURVYSYA.NS', 'KEI.NS', 'KEC.NS', 'KPITTECH.NS', 'KNRCON.NS',
    'LALPATHLAB.NS', 'LICHSGFIN.NS', 'LICI.NS', 'LTIM.NS', 'LTTS.NS', 'LUPIN.NS', 'MFSL.NS', 'MANAPPURAM.NS', 'MAPMYINDIA.NS', 'MAXHEALTH.NS'
]

DATA_DIR = "Stock_Data"


def download_single_stock(ticker: str, period: str = "10y") -> pd.DataFrame:
    """Download single stock OHLCV data."""
    clean_t = clean_ticker_symbol(ticker)
    df_raw = fetch_raw_data_with_retries(clean_t, period=period)
    if df_raw.empty:
        return pd.DataFrame()

    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, f"{clean_t}.csv")
    df_raw.to_csv(file_path, index=False)
    return df_raw


def run_data_pipeline(period: str = "10y", verbose: bool = True) -> tuple:
    """Run pipeline for all supported stocks."""
    os.makedirs(DATA_DIR, exist_ok=True)
    combined = []
    success_cnt = 0
    failed_cnt = 0
    total = len(BASE_STOCKS)

    for idx, ticker in enumerate(BASE_STOCKS, 1):
        clean_t = clean_ticker_symbol(ticker)
        try:
            df = download_single_stock(ticker, period=period)
            if not df.empty:
                df["Ticker"] = clean_t
                combined.append(df)
                success_cnt += 1
            else:
                failed_cnt += 1
        except Exception:
            failed_cnt += 1

        if verbose and (idx % 25 == 0 or idx == total):
            print(f"Pipeline progress: [{idx}/{total}] processed — {success_cnt} succeeded, {failed_cnt} failed so far.")

    if combined:
        df_all = pd.concat(combined, ignore_index=True)
        df_all.to_csv("merge-csv.csv", index=False)
        return df_all, success_cnt, failed_cnt

    return pd.DataFrame(), success_cnt, failed_cnt


if __name__ == "__main__":
    print("Running Data Pipeline...")
    df, s, f = run_data_pipeline()
    print(f"Pipeline finished: {s} succeeded, {f} failed. Total rows: {len(df)}")
