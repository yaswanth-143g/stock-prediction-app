"""
tests/test_frontend_consolidation.py — Integration Tests for Consolidated Frontend Architecture
=============================================================================================
Verifies that all templates and static assets are cleanly served from the unified frontend/ directory.
"""
import os
import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    """Verify GET / renders frontend/templates/index.html correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"STOCK PRICE PREDICTION" in response.data or b"Stock Price" in response.data
    assert b"tvLightweightChartContainer" in response.data

def test_login_route(client):
    """Verify GET /login renders frontend/templates/login.html correctly."""
    response = client.get("/login")
    assert response.status_code == 200
    assert b"Sign In" in response.data
    assert b"NSE QUANT PLATFORM" in response.data

def test_register_route(client):
    """Verify GET /register renders frontend/templates/register.html correctly."""
    response = client.get("/register")
    assert response.status_code == 200
    assert b"Create Account" in response.data
    assert b"NSE QUANT PLATFORM" in response.data

def test_static_css_served(client):
    """Verify GET /static/css/style.css serves stylesheet from frontend/static/css/style.css."""
    response = client.get("/static/css/style.css")
    assert response.status_code == 200
    assert b":root" in response.data or b"body" in response.data

def test_static_dashboard_js_served(client):
    """Verify GET /static/js/dashboard.js serves dashboard script from frontend/static/js/."""
    response = client.get("/static/js/dashboard.js")
    assert response.status_code == 200
    assert len(response.data) > 0

def test_static_tradingview_chart_js_served(client):
    """Verify GET /static/js/tradingview_chart.js serves chart script from frontend/static/js/."""
    response = client.get("/static/js/tradingview_chart.js")
    assert response.status_code == 200
    assert len(response.data) > 0

def test_frontend_directory_inventory():
    """Verify all 6 web frontend files and streamlit_app.py exist in frontend/."""
    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
    expected_files = [
        os.path.join(frontend_dir, "templates", "index.html"),
        os.path.join(frontend_dir, "templates", "login.html"),
        os.path.join(frontend_dir, "templates", "register.html"),
        os.path.join(frontend_dir, "static", "css", "style.css"),
        os.path.join(frontend_dir, "static", "js", "dashboard.js"),
        os.path.join(frontend_dir, "static", "js", "tradingview_chart.js"),
        os.path.join(frontend_dir, "streamlit_app.py"),
    ]
    for fp in expected_files:
        assert os.path.isfile(fp), f"Expected file not found: {fp}"
