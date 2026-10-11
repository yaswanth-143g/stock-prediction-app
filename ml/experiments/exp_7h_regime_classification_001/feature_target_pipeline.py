"""
feature_target_pipeline.py — Dual Feature Sets & Volatility-Aware Target Pipeline
===================================================================================
Constructs Feature Set A (Clean Canonical) and Feature Set B (Stationary Research)
along with 7-horizon volatility-aware ternary classification targets for top 20 stocks.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

# Top 20 Selected Liquid NSE Tickers
TOP_20_TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS",
    "BAJFINANCE.NS", "INFY.NS", "TCS.NS", "TATASTEEL.NS", "KOTAKBANK.NS",
    "BHARTIARTL.NS", "MARUTI.NS", "INDUSINDBK.NS", "ADANIENT.NS", "LT.NS",
    "ADANIPORTS.NS", "ITC.NS", "HINDUNILVR.NS", "IRCTC.NS", "SUNPHARMA.NS"
]

FORECAST_HORIZONS = [1, 2, 3, 4, 5, 6, 7]


def compute_volatility_aware_targets(df_stock: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 7-horizon ternary classification targets (+1 UP, 0 FLAT, -1 DOWN).

    Unit Alignment Verification:
    ----------------------------
    - Horizon return: R_{t, k} = (Close_{t+k} - Close_t) / Close_t  (fractional decimal, e.g. +0.02 = +2%)
    - Volatility proxy: NATR_{frac, t} = (ATR_14 / Close_t) (fractional decimal, e.g. 0.015 = 1.5%)
    - Threshold formula for horizon k:
        theta_{t, k} = 0.5 * NATR_{frac, t} * sqrt(k)
      Both R_{t, k} and theta_{t, k} are unitless fractional returns (decimal).

    Ternary Labeling Rule:
      Target_{t, k} = +1  if R_{t, k} > theta_{t, k}
      Target_{t, k} = -1  if R_{t, k} < -theta_{t, k}
      Target_{t, k} =  0  otherwise (-theta_{t, k} <= R_{t, k} <= theta_{t, k})
    """
    df_out = df_stock.copy()
    close = df_out["Close"]
    high = df_out["High"]
    low = df_out["Low"]

    # Compute 14-day True Range and ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    natr_frac = atr14 / (close + 1e-9)  # fractional ATR (e.g. 0.015)

    for k in FORECAST_HORIZONS:
        # Future fractional return from t to t+k
        future_close = close.shift(-k)
        ret_k = (future_close - close) / (close + 1e-9)

        # Volatility-aware threshold scaled by sqrt(k)
        theta_k = 0.5 * natr_frac * np.sqrt(k)

        target_col = f"Target_h{k}"
        # Ternary categorization
        cond_up = ret_k > theta_k
        cond_down = ret_k < -theta_k

        target_series = pd.Series(0, index=df_out.index, dtype=int)
        target_series[cond_up] = 1
        target_series[cond_down] = -1

        df_out[target_col] = target_series
        df_out[f"Return_h{k}"] = ret_k

    return df_out


def build_feature_set_a(df_stock: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Feature Set A — Clean Canonical Features
    Preserves original technical indicators, removes duplicate Close column,
    and guarantees exact single Close feature assertion.
    """
    df = df_stock.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_p = df["Open"]
    vol = df["Volume"]

    feat_df = pd.DataFrame(index=df.index)

    # Base price & volume (exactly ONE Close)
    feat_df["Close"] = close
    feat_df["Open"] = open_p
    feat_df["High"] = high
    feat_df["Low"] = low
    feat_df["Volume"] = vol

    # Moving Average Ratios
    for m in [5, 10, 20, 50, 200]:
        feat_df[f"Close_MA{m}_ratio"] = close / (close.rolling(m).mean() + 1e-9)

    feat_df["Close_EMA12_ratio"] = close / (close.ewm(span=12, adjust=False).mean() + 1e-9)
    feat_df["Close_EMA26_ratio"] = close / (close.ewm(span=26, adjust=False).mean() + 1e-9)

    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    feat_df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feat_df["MACD"] = macd
    feat_df["MACD_Signal"] = signal
    feat_df["MACD_Hist"] = macd - signal

    # Bollinger Bands
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_high = ma20 + 2 * std20
    bb_low = ma20 - 2 * std20
    feat_df["BB_High_ratio"] = close / (bb_high + 1e-9)
    feat_df["BB_Low_ratio"] = close / (bb_low + 1e-9)
    feat_df["BB_Width"] = (bb_high - bb_low) / (ma20 + 1e-9)

    # ATR & NATR
    high_low = high - low
    high_pc = (high - close.shift(1)).abs()
    low_pc = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    feat_df["ATR"] = tr.rolling(14).mean()
    feat_df["NATR"] = (feat_df["ATR"] / (close + 1e-9)) * 100.0

    # Stochastic
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    k = 100 * ((close - low14) / (high14 - low14 + 1e-9))
    feat_df["Stoch_K"] = k
    feat_df["Stoch_D"] = k.rolling(3).mean()

    # OBV
    obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    feat_df["OBV"] = obv
    feat_df["OBV_MA10_ratio"] = obv / (obv.rolling(10).mean() + 1e-9)

    # Historical Returns & Volatilities
    for r in [1, 2, 3, 5, 10, 20, 30]:
        feat_df[f"Return_{r}d"] = close.pct_change(r)

    feat_df["Vol_5d"] = close.pct_change().rolling(5).std()
    feat_df["Vol_20d"] = close.pct_change().rolling(20).std()

    feat_df["High_Low_ratio"] = high / (low + 1e-9)
    feat_df["Close_Open_ratio"] = close / (open_p + 1e-9)

    # Sanitize infinities
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

    feature_cols = feat_df.columns.tolist()

    # Explicit Safety Assertions
    assert feature_cols.count("Close") <= 1, f"Duplicate Close detected in Feature Set A! Count: {feature_cols.count('Close')}"
    assert len(feature_cols) == len(set(feature_cols)), "Duplicate feature column names detected in Set A!"

    return feat_df, feature_cols


def build_feature_set_b(df_stock: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Feature Set B — Stationary Research Features
    Focuses entirely on stationary, normalized indicators (ratios, normalized returns,
    oscillators, relative volumes) to avoid relying on raw unscaled price levels.
    """
    df = df_stock.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_p = df["Open"]
    vol = df["Volume"]

    feat_df = pd.DataFrame(index=df.index)

    # 1. Stationary Returns (1d to 30d)
    for r in [1, 2, 3, 5, 10, 20, 30]:
        feat_df[f"Return_{r}d"] = close.pct_change(r)

    # 2. Price / MA Normalized Deviations (Stationary)
    for m in [5, 10, 20, 50, 200]:
        sma = close.rolling(m).mean()
        feat_df[f"Close_SMA{m}_dev"] = (close - sma) / (sma + 1e-9)

    feat_df["Close_EMA12_dev"] = (close - close.ewm(span=12, adjust=False).mean()) / (close + 1e-9)
    feat_df["Close_EMA26_dev"] = (close - close.ewm(span=26, adjust=False).mean()) / (close + 1e-9)

    # 3. Volatility & Volatility-Normalized Returns
    vol_20d = close.pct_change().rolling(20).std()
    feat_df["Vol_5d"] = close.pct_change().rolling(5).std()
    feat_df["Vol_20d"] = vol_20d
    feat_df["Sharpe_Proxy_5d"] = feat_df["Return_5d"] / (vol_20d * np.sqrt(5) + 1e-6)
    feat_df["Sharpe_Proxy_20d"] = feat_df["Return_20d"] / (vol_20d * np.sqrt(20) + 1e-6)

    # ATR / NATR
    high_low = high - low
    high_pc = (high - close.shift(1)).abs()
    low_pc = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    feat_df["NATR"] = (atr / (close + 1e-9)) * 100.0

    # 4. Oscillators & Bounded Indicators
    # RSI (0-100 scaled to -0.5 to +0.5)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    feat_df["RSI_norm"] = (rsi - 50.0) / 50.0

    # MACD Normalized by Close
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    feat_df["MACD_norm"] = macd / (close + 1e-9)
    feat_df["MACD_Signal_norm"] = signal / (close + 1e-9)
    feat_df["MACD_Hist_norm"] = (macd - signal) / (close + 1e-9)

    # Bollinger Bands Position & Width
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_high = ma20 + 2 * std20
    bb_low = ma20 - 2 * std20
    feat_df["BB_Width"] = (bb_high - bb_low) / (ma20 + 1e-9)
    feat_df["BB_Pos"] = (close - bb_low) / (bb_high - bb_low + 1e-9)

    # Stochastic Oscillators
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    k = 100 * ((close - low14) / (high14 - low14 + 1e-9))
    feat_df["Stoch_K_norm"] = (k - 50.0) / 50.0
    feat_df["Stoch_D_norm"] = (k.rolling(3).mean() - 50.0) / 50.0

    # 5. Volume & Intraday Price Action Ratios
    vol_ma20 = vol.rolling(20).mean() + 1e-9
    feat_df["Relative_Vol"] = vol / vol_ma20
    feat_df["Vol_Change"] = vol.pct_change()

    day_range = (high - low) + 1e-9
    feat_df["Gap_Pct"] = (open_p - close.shift(1)) / (close.shift(1) + 1e-9)
    feat_df["Candle_Body_Ratio"] = (close - open_p) / day_range
    feat_df["Upper_Wick_Ratio"] = (high - pd.concat([open_p, close], axis=1).max(axis=1)) / day_range
    feat_df["Lower_Wick_Ratio"] = (pd.concat([open_p, close], axis=1).min(axis=1) - low) / day_range

    # Sanitize infinities
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)

    feature_cols = feat_df.columns.tolist()

    # Explicit Safety Assertions
    assert feature_cols.count("Close") <= 1, f"Duplicate Close detected in Feature Set B! Count: {feature_cols.count('Close')}"
    assert len(feature_cols) == len(set(feature_cols)), "Duplicate feature column names detected in Set B!"

    return feat_df, feature_cols
