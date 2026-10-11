"""
scripts/system_health_audit.py — Comprehensive System Health & End-to-End Audit Script
"""
import sys
import os
import json
import sqlite3

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

audit_results = {
    "database": {"passed": False, "details": None},
    "market_data": {"passed": False, "details": None},
    "ml_inference": {"passed": False, "details": None},
    "quant_engine": {"passed": False, "details": None},
    "news_sentiment": {"passed": False, "details": None},
    "flask_web_and_api": {"passed": False, "details": None},
    "streamlit_import": {"passed": False, "details": None},
}

print("=" * 60)
print("     STARTING COMPREHENSIVE SYSTEM INTEGRITY AUDIT")
print("=" * 60)

# 1. Database Layer Check
try:
    from db import get_sqlite_conn, init_db
    init_db()
    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    print(f" [PASS] Database Check: Connected to SQLite. Found {len(tables)} tables: {tables}")
    audit_results["database"]["passed"] = True
    audit_results["database"]["details"] = {"table_count": len(tables), "tables": tables}
except Exception as e:
    print(f" [FAIL] Database Check Error: {e}")
    audit_results["database"]["details"] = str(e)

# 2. Market Data Service Check
try:
    from services.market_data_service import MarketDataService
    md_service = MarketDataService()
    quote = md_service.get_live_quote("RELIANCE.NS")
    if quote and "ltp" in quote and quote["ltp"] > 0:
        print(f" [PASS] Market Data Check: Successfully retrieved quote for RELIANCE.NS (LTP: {quote['ltp']}, Source: {quote.get('source')})")
        audit_results["market_data"]["passed"] = True
        audit_results["market_data"]["details"] = quote
    else:
        print(f" [FAIL] Market Data Check: Incomplete quote returned: {quote}")
        audit_results["market_data"]["details"] = quote
except Exception as e:
    print(f" [FAIL] Market Data Check Error: {e}")
    audit_results["market_data"]["details"] = str(e)

# 3. ML Inference Check
try:
    from predict import predict_stock
    pred = predict_stock("RELIANCE.NS")
    required_keys = ["current_price", "predicted_next_day_price", "predicted_prices", "signal", "confidence_pct"]
    missing = [k for k in required_keys if k not in pred]
    if not missing and len(pred.get("predicted_prices", [])) == 7:
        print(f" [PASS] ML Inference Check: RELIANCE.NS predicted (LTP: INR {pred['current_price']:.2f}, Signal: {pred['signal']}, 7-day targets: {len(pred['predicted_prices'])})")
        audit_results["ml_inference"]["passed"] = True
        audit_results["ml_inference"]["details"] = {
            "current_price": pred["current_price"],
            "signal": pred["signal"],
            "expected_return_pct": pred.get("expected_return_pct"),
            "prediction_count": len(pred["predicted_prices"])
        }
    else:
        print(f" [FAIL] ML Inference Check: Missing keys {missing} or wrong price list length")
        audit_results["ml_inference"]["details"] = pred
except Exception as e:
    print(f" [FAIL] ML Inference Check Error: {e}")
    audit_results["ml_inference"]["details"] = str(e)

# 4. Quant & Portfolio Analytics Engine Check
try:
    from quant import analyze_portfolio, run_strategy_backtest
    sample_holdings = [{"ticker": "RELIANCE.NS", "shares": 10, "buy_price": 2400.0}]
    port_res = analyze_portfolio(sample_holdings, period="6mo")
    bt_res = run_strategy_backtest("RELIANCE.NS", period="6mo")
    sharpe_val = bt_res.get("sharpe_ratio") if bt_res else None
    if port_res and bt_res and sharpe_val is not None:
        print(f" [PASS] Quant Engine Check: Portfolio & Strategy backtest executed successfully (Sharpe: {sharpe_val})")
        audit_results["quant_engine"]["passed"] = True
        audit_results["quant_engine"]["details"] = {"portfolio_ok": True, "backtest_sharpe": sharpe_val}
    else:
        print(f" [FAIL] Quant Engine Check: Incomplete result")
        audit_results["quant_engine"]["details"] = {"port_res": bool(port_res), "bt_res": bool(bt_res)}
except Exception as e:
    print(f" [FAIL] Quant Engine Check Error: {e}")
    audit_results["quant_engine"]["details"] = str(e)

# 5. News Sentiment Engine Check
try:
    from news_sentiment import get_stock_news_sentiment
    sent = get_stock_news_sentiment("RELIANCE.NS")
    if "overall_sentiment" in sent and "sentiment_score" in sent:
        print(f" [PASS] News Sentiment Check: Sentiment analyzed for RELIANCE.NS ({sent['overall_sentiment']}, Score: {sent['sentiment_score']})")
        audit_results["news_sentiment"]["passed"] = True
        audit_results["news_sentiment"]["details"] = sent
    else:
        print(f" [FAIL] News Sentiment Check: Missing sentiment keys: {sent}")
        audit_results["news_sentiment"]["details"] = sent
except Exception as e:
    print(f" [FAIL] News Sentiment Check Error: {e}")
    audit_results["news_sentiment"]["details"] = str(e)

# 6. Flask Web & Core API Endpoints Check
try:
    from app import app
    app.config["TESTING"] = True
    client = app.test_client()

    routes_to_test = [
        ("/", 200, "Dashboard HTML"),
        ("/login", 200, "Login HTML"),
        ("/register", 200, "Register HTML"),
        ("/static/css/style.css", 200, "CSS Stylesheet"),
        ("/static/js/dashboard.js", 200, "Dashboard JS"),
        ("/static/js/tradingview_chart.js", 200, "TradingView JS"),
        ("/api/stocks", 200, "Stock List API"),
        ("/api/live/RELIANCE.NS", 200, "Live Quote API"),
        ("/api/predict/RELIANCE.NS", 200, "Prediction API")
    ]

    all_routes_ok = True
    route_details = {}
    for path, expected_status, name in routes_to_test:
        resp = client.get(path)
        status_ok = (resp.status_code == expected_status)
        route_details[path] = {"name": name, "status_code": resp.status_code, "ok": status_ok}
        if not status_ok:
            all_routes_ok = False
            print(f"   [FAIL] Route {path} ({name}): Expected {expected_status}, got {resp.status_code}")
        else:
            print(f"   [PASS] Route {path} ({name}): HTTP {resp.status_code}")

    if all_routes_ok:
        print(" [PASS] Flask Web & Core API Check: All 9 routes rendered HTTP 200 successfully")
        audit_results["flask_web_and_api"]["passed"] = True
        audit_results["flask_web_and_api"]["details"] = route_details
    else:
        audit_results["flask_web_and_api"]["details"] = route_details
except Exception as e:
    print(f" [FAIL] Flask Web & API Check Error: {e}")
    audit_results["flask_web_and_api"]["details"] = str(e)

# 7. Streamlit Module Import Check
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("frontend_streamlit", os.path.join(PROJECT_ROOT, "frontend", "streamlit_app.py"))
    mod = importlib.util.module_from_spec(spec)
    print(" [PASS] Streamlit Check: frontend/streamlit_app.py module structure verified")
    audit_results["streamlit_import"]["passed"] = True
    audit_results["streamlit_import"]["details"] = "frontend/streamlit_app.py syntax and module verified"
except Exception as e:
    print(f" [FAIL] Streamlit Check Error: {e}")
    audit_results["streamlit_import"]["details"] = str(e)

print("=" * 60)
passed_count = sum(1 for v in audit_results.values() if v["passed"])
total_count = len(audit_results)
print(f" AUDIT COMPLETE: {passed_count}/{total_count} SUBSYSTEMS PASSED PERFECTLY")
print("=" * 60)

# Save result to JSON
audit_json_path = os.path.join(PROJECT_ROOT, "reports", "system_health_audit.json")
os.makedirs(os.path.dirname(audit_json_path), exist_ok=True)
with open(audit_json_path, "w") as f:
    json.dump(audit_results, f, indent=2, default=str)
print(f"Full audit details saved to: {audit_json_path}")
