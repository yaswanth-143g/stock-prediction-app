"""
tests/test_new_stocks_and_pipeline.py — Comprehensive Test Suite for New Stocks & Multi-Asset Pipeline
========================================================================================================
Validates data ingestion, ticker resolution, feature engineering, AI ensemble prediction,
portfolio optimization, backtesting, and API endpoints for a wide range of NEW stock symbols:
1. Indian Equities: TATAMOTORS.NS, SBIN.NS, ICICIBANK.NS, BHARTIARTL.NS, ITC.NS, WIPRO.NS
2. US Equities: AAPL, MSFT, NVDA, TSLA, GOOGL
3. Multi-Asset Instruments: BTC-USD, ETH-USD, GC=F, USDINR=X
4. Automatic Ticker Alias & Replacement Mapping: tatamotors -> TATAMOTORS.NS, zomato -> ETERNAL.NS
"""

import pytest
import json
import pandas as pd
from app import app
from ticker_validator import clean_ticker_symbol, fetch_raw_data_with_retries, validate_single_ticker
from predict import predict_stock, predict_live_intraday, predict_7days, get_model_manager
from services.market_data_service import get_market_data_service
from quant.analytics import optimize_portfolio_mvo, analyze_portfolio


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


NEW_INDIAN_STOCKS = ["TATAMOTORS.NS", "SBIN.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "ITC.NS", "WIPRO.NS"]
NEW_US_STOCKS = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]
MULTI_ASSETS = ["BTC-USD", "ETH-USD", "GC=F", "USDINR=X"]
ALIAS_INPUTS = [("tatamotors", "TATAMOTORS.NS"), ("zomato", "ETERNAL.NS"), ("sbin", "SBIN.NS"), ("aapl", "AAPL")]


def test_ticker_cleaning_and_alias_resolution():
    """Verify ticker cleaner normalizes new stocks, US equities, crypto, and legacy aliases."""
    for raw_inp, expected in ALIAS_INPUTS:
        cleaned = clean_ticker_symbol(raw_inp)
        assert cleaned == expected, f"Expected {expected} for input {raw_inp}, got {cleaned}"


def test_data_fetch_for_new_stocks():
    """Verify data fetcher retrieves valid non-empty DataFrame for all new stock symbols."""
    all_new = NEW_INDIAN_STOCKS + NEW_US_STOCKS + MULTI_ASSETS
    for stk in all_new:
        df = fetch_raw_data_with_retries(stk, period="1y")
        assert df is not None and not df.empty, f"Data fetch returned empty DataFrame for {stk}"
        assert len(df) >= 10, f"Insufficient rows ({len(df)}) returned for {stk}"
        assert "Close" in df.columns or "close" in df.columns, f"Missing Close price column for {stk}"


@pytest.mark.parametrize("ticker", NEW_INDIAN_STOCKS + NEW_US_STOCKS + MULTI_ASSETS)
def test_predict_stock_for_new_stocks(ticker):
    """Verify predict_stock generates valid 7-day AI forecast for new stock symbols."""
    res = predict_stock(ticker)
    assert res["status"] == "success"
    assert "current_price" in res and res["current_price"] > 0
    assert "predicted_prices" in res and len(res["predicted_prices"]) == 7
    assert "signal" in res
    assert "model_breakdown" in res
    assert "stop_loss" in res
    assert "take_profit" in res
    assert "risk_reward_ratio" in res
    assert "fundamentals" in res


@pytest.mark.parametrize("ticker", ["TATAMOTORS.NS", "SBIN.NS", "AAPL", "BTC-USD"])
def test_market_data_service_visualization_data(ticker):
    """Verify MarketDataService visualization payload generation for new stocks."""
    svc = get_market_data_service()
    res = svc.get_visualization_data(ticker, period="6m")
    assert res["status"] == "success"
    assert "live_quote" in res
    assert "ohlcv_series" in res and len(res["ohlcv_series"]) > 0
    assert "volume_summary" in res
    assert "analytics_summary" in res
    latest = res["ohlcv_series"][-1]
    assert "close" in latest
    assert "sma20" in latest
    assert "rsi" in latest
    assert "macd" in latest


def test_mvo_portfolio_optimization_with_new_stocks():
    """Verify Markowitz MVO & Risk Parity portfolio optimization with new stock tickers."""
    tickers = ["TATAMOTORS.NS", "SBIN.NS", "ICICIBANK.NS", "AAPL", "NVDA"]
    res = optimize_portfolio_mvo(tickers=tickers, period="1y")
    assert res["status"] == "success"
    assert "max_sharpe_portfolio" in res
    assert "min_volatility_portfolio" in res
    assert "risk_parity_portfolio" in res
    assert "mvo_weights" in res
    assert "min_volatility_weights" in res
    assert "risk_parity_weights" in res
    assert "efficient_frontier" in res
    assert len(res["mvo_weights"]) == len(tickers)


def test_api_endpoints_with_new_stocks(client):
    """Verify Flask API endpoints handle new stock ticker requests correctly."""
    # 1. Live intraday endpoint
    rv_live = client.get("/api/live/TATAMOTORS.NS")
    assert rv_live.status_code == 200
    data_live = json.loads(rv_live.data)
    assert data_live["status"] == "success"

    # 2. Prediction endpoint
    rv_pred = client.get("/api/prediction/SBIN.NS")
    assert rv_pred.status_code == 200
    data_pred = json.loads(rv_pred.data)
    assert data_pred["status"] == "success"

    # 3. Stock visualization endpoint
    rv_viz = client.get("/api/stock/AAPL/visualization?period=1y&include_prediction=true")
    assert rv_viz.status_code == 200
    data_viz = json.loads(rv_viz.data)
    assert data_viz["status"] == "success"

    # 4. CSV Export endpoint
    rv_csv = client.get("/api/export/csv/TATAMOTORS.NS")
    assert rv_csv.status_code == 200
    assert "text/csv" in rv_csv.content_type

    # 5. PDF Export endpoint
    rv_pdf = client.get("/api/export/pdf/SBIN.NS")
    assert rv_pdf.status_code == 200

    # 6. Portfolio Optimize API endpoint with new stocks
    payload = {"tickers": ["TATAMOTORS.NS", "SBIN.NS", "AAPL"], "target_return": 0.15}
    rv_mvo = client.post("/api/portfolio/optimize", json=payload)
    assert rv_mvo.status_code == 200
    data_mvo = json.loads(rv_mvo.data)
    assert data_mvo["status"] == "success"
    assert "mvo_weights" in data_mvo
    assert "risk_parity_weights" in data_mvo
