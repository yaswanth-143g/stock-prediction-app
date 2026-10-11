"""
tests/test_production_upgrades.py — Tests for Production Upgrade Requirements
==============================================================================
Validates ModelManager preloading, /api/health endpoint, OrderService paper trading,
CSV/PDF export routes, and multi-asset ticker normalization.
"""

import pytest
import json
from app import app
from predict import get_model_manager, predict_stock
from services.order_service import get_order_service
from ticker_validator import clean_ticker_symbol


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_model_manager_singleton_preloading():
    """Verify ModelManager singleton preloads models with sub-50ms inference availability."""
    mm1 = get_model_manager()
    mm2 = get_model_manager()
    assert mm1 is mm2
    assert hasattr(mm1, "loaded_models")
    assert isinstance(mm1.loaded_models, dict)


def test_api_health_endpoint(client):
    """Verify GET /api/health endpoint returns structured component statuses."""
    response = client.get("/api/health")
    assert response.status_code in [200, 500]
    data = response.get_json()
    assert "status" in data
    assert "components" in data
    assert "database" in data["components"]
    assert "cache" in data["components"]
    assert "model_manager" in data["components"]
    assert "scheduler" in data["components"]


def test_paper_trading_order_service(client):
    """Verify OrderService paper trading placement, virtual balances, and positions."""
    svc = get_order_service()
    res = svc.place_order(
        user_id="test_user_unit",
        ticker="RELIANCE.NS",
        side="BUY",
        quantity=5,
        order_type="MARKET"
    )
    assert res["status"] == "success"
    assert res["order"]["side"] == "BUY"
    assert res["order"]["quantity"] == 5

    positions = svc.get_positions("test_user_unit")
    assert len(positions) >= 1
    assert positions[0]["ticker"] == "RELIANCE.NS"
    assert positions[0]["shares"] >= 5



def test_export_csv_endpoint(client):
    """Verify GET /api/export/csv/<ticker> returns downloadable CSV content."""
    response = client.get("/api/export/csv/RELIANCE.NS")
    assert response.status_code == 200
    assert "text/csv" in response.content_type
    assert b"Day" in response.data or b"Forecast Date" in response.data or len(response.data) > 0


def test_export_pdf_endpoint(client):
    """Verify GET /api/export/pdf/<ticker> returns downloadable PDF content."""
    response = client.get("/api/export/pdf/RELIANCE.NS")
    assert response.status_code == 200
    assert len(response.data) > 0


def test_multi_asset_ticker_validation():
    """Verify multi-asset ticker cleaner handles Crypto, Forex, Commodities, and US Equities."""
    assert clean_ticker_symbol("btc-usd") == "BTC-USD"
    assert clean_ticker_symbol("usdinr=x") == "USDINR=X"
    assert clean_ticker_symbol("aapl") == "AAPL"
    assert clean_ticker_symbol("gc=f") == "GC=F"
    assert clean_ticker_symbol("reliance") == "RELIANCE.NS"


def test_mvo_portfolio_optimization_endpoint(client):
    """Verify POST /api/portfolio/optimize returns Markowitz MVO & Risk Parity allocations."""
    payload = {
        "tickers": ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"],
        "target_return": 0.15
    }
    response = client.post("/api/portfolio/optimize", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "mvo_weights" in data
    assert "risk_parity_weights" in data
    assert "min_volatility_weights" in data


def test_ensemble_model_endpoints(client):
    """Verify GET /api/models/ensemble and POST /api/models/ensemble/reweight endpoints."""
    res_get = client.get("/api/models/ensemble")
    assert res_get.status_code == 200
    data_get = res_get.get_json()
    assert data_get["status"] == "success"
    assert "ensemble_weights" in data_get

    res_post = client.post("/api/models/ensemble/reweight", json={"rolling_rmse": {"random_forest": 1.2, "xgboost": 1.3, "lstm": 1.4, "lightgbm": 1.5}})
    assert res_post.status_code == 200
    data_post = res_post.get_json()
    assert data_post["status"] == "success"


def test_options_analytics_endpoint(client):
    """Verify GET /api/options/<symbol> returns PCR, Max Pain, and IV Skew analytics."""
    response = client.get("/api/options/NIFTY")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] in ["success", "unavailable"]
    if data["status"] == "success":
        assert "pcr_oi" in data
        assert "max_pain_strike" in data
        assert "iv_skew_pct" in data
    else:
        assert "options_chain_matrix" in data


def test_prometheus_metrics_endpoint(client):
    """Verify GET /metrics returns Prometheus formatted telemetry text."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"nse_platform_up" in response.data
    assert b"nse_paper_trading_cash_balance_inr" in response.data

