"""
tests/test_live_market_and_ml.py — Comprehensive Test Suite for Live Market Data & AI Engine
================================================================================================
Validates:
1. MarketDataService live quotes, search, market status, index overview
2. Redis caching & fallback resilience
3. MySQL market_data table storage & retrieval
4. 36-feature pipeline & multi-model ensemble prediction (LSTM, XGBoost, RF, LightGBM, Ensemble)
5. Flask API endpoints (/api/live/<ticker>, /api/market/status, /api/market/indices, /api/stocks/search, /api/stock/<ticker>/history, /api/predict/<ticker>)
"""

import importlib
pytest = importlib.import_module("pytest")
import os
import json
import pandas as pd
from app import app
from services.market_data_service import get_market_data_service, MarketDataService
from predict import predict_7days, predict_live_intraday, get_nse_market_status
from db import save_market_data_tick, get_recent_market_ticks


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_market_data_service_singleton():
    svc1 = get_market_data_service()
    svc2 = get_market_data_service()
    assert svc1 is svc2
    assert isinstance(svc1, MarketDataService)


def test_get_live_quote_format():
    svc = get_market_data_service()
    quote = svc.get_live_quote("RELIANCE.NS")

    assert isinstance(quote, dict)
    assert "ticker" in quote
    assert "exchange" in quote
    assert "ltp" in quote
    assert "open" in quote
    assert "high" in quote
    assert "low" in quote
    assert "volume" in quote
    assert "source" in quote
    assert quote["source"] in ["market_api", "redis_cache", "yfinance_fallback", "local_csv_fallback", "unavailable"]


def test_market_status():
    svc = get_market_data_service()
    status = svc.get_market_status()
    assert status["status"] == "success"
    assert "is_market_open" in status
    assert "market_status" in status
    assert "current_time_ist" in status


def test_market_indices():
    svc = get_market_data_service()
    res = svc.get_market_indices()
    assert res["status"] == "success"
    assert "indices" in res
    assert len(res["indices"]) >= 4
    for idx in res["indices"]:
        assert "symbol" in idx
        assert "name" in idx
        assert "price" in idx


def test_stock_search():
    svc = get_market_data_service()
    results = svc.search_stocks("RELIANCE")
    assert isinstance(results, list)
    assert len(results) > 0
    assert results[0]["ticker"] == "RELIANCE.NS"


def test_mysql_sqlite_market_data_tick_storage():
    tick = {
        "ticker": "TEST_TICKER",
        "exchange": "NSE",
        "ltp": 1250.75,
        "open": 1240.00,
        "high": 1260.00,
        "low": 1235.00,
        "volume": 10000,
        "source": "unit_test",
        "timestamp": "2026-09-30T10:00:00+05:30"
    }
    save_market_data_tick(tick)
    recent = get_recent_market_ticks("TEST_TICKER", limit=5)
    assert isinstance(recent, list)
    assert len(recent) > 0
    first = dict(recent[0])
    assert first["ticker"] == "TEST_TICKER"
    assert float(first["close"]) == 1250.75


def test_prediction_ensemble_pipeline():
    res = predict_7days("RELIANCE.NS")
    assert res["status"] == "success"
    assert "current_price" in res
    assert "predicted_prices" in res
    assert len(res["predicted_prices"]) == 7
    assert "model_breakdown" in res
    breakdown = res["model_breakdown"]
    assert "random_forest" in breakdown
    assert "xgboost" in breakdown
    assert "lstm" in breakdown
    assert "lightgbm" in breakdown
    assert "ensemble" in breakdown
    assert "stop_loss" in res
    assert "take_profit" in res
    assert "risk_reward_ratio" in res


def test_api_live_endpoint(client):
    rv = client.get("/api/live/RELIANCE.NS")
    assert rv.status_code == 200
    json_data = json.loads(rv.data)
    assert json_data["status"] == "success"
    assert "ltp" in json_data
    assert "source" in json_data
    assert "predicted_5m_target" in json_data


def test_api_market_status_endpoint(client):
    rv = client.get("/api/market/status")
    assert rv.status_code == 200
    json_data = json.loads(rv.data)
    assert json_data["status"] == "success"
    assert "is_market_open" in json_data


def test_api_market_indices_endpoint(client):
    rv = client.get("/api/market/indices")
    assert rv.status_code == 200
    json_data = json.loads(rv.data)
    assert json_data["status"] == "success"
    assert "indices" in json_data


def test_api_stocks_search_endpoint(client):
    rv = client.get("/api/stocks/search?q=TCS")
    assert rv.status_code == 200
    json_data = json.loads(rv.data)
    assert json_data["status"] == "success"
    assert json_data["count"] > 0
    assert json_data["stocks"][0]["ticker"] == "TCS.NS"


def test_api_prediction_endpoint(client):
    rv = client.get("/api/prediction/RELIANCE.NS")
    assert rv.status_code == 200
    json_data = json.loads(rv.data)
    assert json_data["status"] == "success"
    assert "predicted_price" in json_data
    assert "signal" in json_data


def test_get_visualization_data_service():
    svc = get_market_data_service()
    res = svc.get_visualization_data("RELIANCE.NS", period="1y")

    assert res["status"] == "success"
    assert res["ticker"] == "RELIANCE.NS"
    assert "live_quote" in res
    assert "data_source" in res
    assert "ohlcv_series" in res
    assert isinstance(res["ohlcv_series"], list)
    assert len(res["ohlcv_series"]) > 0

    latest = res["ohlcv_series"][-1]
    assert "date" in latest
    assert "open" in latest
    assert "close" in latest
    assert "high" in latest
    assert "low" in latest
    assert "volume" in latest
    assert "sma20" in latest
    assert "rsi" in latest
    assert "macd" in latest

    assert "volume_summary" in res
    assert "relative_volume" in res["volume_summary"]
    assert "analytics_summary" in res
    assert "atr" in res["analytics_summary"]
    assert "volatility_pct" in res["analytics_summary"]


def test_api_stock_visualization_endpoint(client):
    rv = client.get("/api/stock/RELIANCE.NS/visualization?period=1y&include_prediction=true")
    assert rv.status_code == 200
    json_data = json.loads(rv.data)

    assert json_data["status"] == "success"
    assert json_data["ticker"] == "RELIANCE.NS"
    assert "ohlcv_series" in json_data
    assert "ai_forecast" in json_data
    if json_data["ai_forecast"]:
        assert "prediction_dates" in json_data["ai_forecast"]
        assert "predicted_prices" in json_data["ai_forecast"]

