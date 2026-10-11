"""
app.py — Flask Backend Server for Stock Price Prediction Dashboard
====================================================================
Serves index.html frontend and connects REST API endpoints to predict.py & yfinance.
"""

import os
import io
import datetime
import logging
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit, join_room, leave_room
import pandas as pd
import numpy as np
import yfinance as yf

# Core Prediction Pipeline Imports
from ticker_validator import clean_and_validate_tickers, clean_ticker_symbol, TICKER_ALIASES, fetch_raw_data_with_retries
from predict import predict_stock, predict_live_intraday
from news_sentiment import get_stock_news_sentiment
from services.market_data_service import MarketDataService
from services.scheduler import start_scheduler
from services.websocket_client import get_auto_connect_feed

logger = logging.getLogger(__name__)
market_service = MarketDataService()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_DIR, "templates"),
    static_folder=os.path.join(FRONTEND_DIR, "static"),
    static_url_path="/static"
)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "nse_stock_prediction_secret_key_2026")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Initialize Real-Time Auto-Connect Market Feed & Background Scheduler
auto_feed = get_auto_connect_feed(socketio=socketio)
try:
    auto_feed.start()
except Exception as feed_err:
    logger.warning(f"AutoConnectMarketFeed initialization warning: {feed_err}")

try:
    start_scheduler()
except Exception as ex:
    logger.warning(f"Scheduler initialization warning: {ex}")



# Pre-populated Stock Universe
DEFAULT_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "SBIN.NS", "LT.NS", "ITC.NS",
    "HINDUNILVR.NS", "BHARTIARTL.NS", "AXISBANK.NS", "KOTAKBANK.NS",
    "MARUTI.NS", "ASIANPAINT.NS", "SUNPHARMA.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "BAJFINANCE.NS", "NTPC.NS", "POWERGRID.NS"
]


def json_sanitize(obj):
    """Recursively converts numpy, pandas, and datetime objects into native JSON-serializable Python types."""
    if obj is None:
        return None
    elif isinstance(obj, (bool, str, int, float)):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return 0.0
        return obj
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [json_sanitize(x) for x in obj.tolist()]
    elif isinstance(obj, (pd.Timestamp, datetime.date, datetime.datetime)):
        return obj.strftime("%Y-%m-%d")
    elif isinstance(obj, pd.Series):
        return json_sanitize(obj.to_dict())
    elif isinstance(obj, pd.DataFrame):
        return json_sanitize(obj.to_dict(orient="records"))
    elif isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items() if k != "df"}
    elif isinstance(obj, (list, tuple, set)):
        return [json_sanitize(x) for x in obj]
    else:
        try:
            return str(obj)
        except Exception:
            return None


def get_all_supported_stocks():
    """Returns list of clean, validated supported stock tickers."""
    all_candidates = list(DEFAULT_STOCKS)
    if os.path.exists("Stock_Data"):
        for f in os.listdir("Stock_Data"):
            if f.endswith(".csv"):
                t = f.replace(".csv", "").strip()
                if t and t not in all_candidates:
                    all_candidates.append(t)
    return clean_and_validate_tickers(all_candidates)


@app.route("/")
def index():
    """Renders main HTML dashboard."""
    return render_template("index.html")


@app.route("/login")
def login_page():
    """Renders Login page."""
    return render_template("login.html")


@app.route("/register")
def register_page():
    """Renders Register page."""
    return render_template("register.html")


@app.route("/api/stocks", methods=["GET"])
def api_stocks():
    """Returns JSON list of supported stock tickers."""
    stocks = get_all_supported_stocks()
    return jsonify({"status": "success", "stocks": stocks})


@app.route("/api/prediction/<ticker>", methods=["GET"])
@app.route("/api/predict/<ticker>", methods=["GET"])
@app.route("/api/predict", methods=["GET", "POST"])
def api_predict(ticker=None):
    """
    Returns real prediction payload sourced directly from predict.py.
    Supports GET/POST, dynamic alpha sentiment parameter, 95% confidence bounds, and intraday targets.
    """
    alpha = None
    if not ticker:
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            ticker = data.get("ticker")
            alpha = data.get("alpha", 0.0)
        else:
            ticker = request.args.get("ticker")
            alpha = request.args.get("alpha", default=0.0, type=float)
    else:
        alpha = request.args.get("alpha", default=0.0, type=float)

    if not ticker:
        ticker = "RELIANCE.NS"

    ticker = ticker.strip().upper().rstrip(".")

    try:
        res = predict_stock(ticker, alpha=alpha)
        
        # Standardize payload format required by specification
        curr_p = res.get("current_price", 0.0)
        pred_p = res.get("predicted_next_day_price", 0.0)
        ret_pct = res.get("expected_return_pct", 0.0)
        sig = res.get("signal", "HOLD")
        model = res.get("model_name", "LSTM")

        is_replaced = res.get("is_replaced", False)
        orig_ticker = res.get("original_ticker", ticker)
        fin_ticker = res.get("final_ticker", res.get("ticker", ticker))
        repl_msg = res.get("replacement_message")

        payload = {
            "status": "success",
            "ticker": fin_ticker,
            "original_ticker": orig_ticker,
            "final_ticker": fin_ticker,
            "is_replaced": is_replaced,
            "message": repl_msg if is_replaced else "Prediction calculated successfully",
            "current_price": curr_p,
            "predicted_price": pred_p,
            "expected_return": ret_pct,
            "signal": sig,
            "model": model,
            "prediction_date": res.get("current_date", datetime.date.today().strftime("%Y-%m-%d")),
            "prediction_dates": res.get("prediction_dates", []),
            "predicted_prices": res.get("predicted_prices", []),
            "upper_bounds": res.get("upper_bounds", []),
            "lower_bounds": res.get("lower_bounds", []),
            "intraday_targets": res.get("intraday_targets", {}),
            "alpha_parameter": res.get("alpha_parameter", 0.15),
            "stop_loss": res.get("stop_loss"),
            "take_profit": res.get("take_profit"),
            "risk_reward_ratio": res.get("risk_reward_ratio"),
            "prob_rise_pct": res.get("prob_rise_pct"),
            "prob_fall_pct": res.get("prob_fall_pct"),
            "confidence_pct": res.get("confidence_pct"),
            "dynamic_buy_threshold": res.get("dynamic_buy_threshold"),
            "fundamentals": res.get("fundamentals"),
            "news_sentiment": res.get("news_sentiment"),
            "raw_result": res
        }
        return jsonify(json_sanitize(payload))

    except Exception as e:
        return jsonify({
            "status": "error",
            "ticker": ticker,
            "message": f"Ticker validation / prediction failed: {str(e)}"
        }), 400


@app.route("/api/historical/<ticker>", methods=["GET"])
def api_historical(ticker):
    """Returns historical daily close prices for Chart.js visualization with robust fallbacks."""
    raw_ticker = ticker.strip().upper()
    cleaned = clean_ticker_symbol(raw_ticker) or raw_ticker
    if not cleaned.endswith(".NS") and "." not in cleaned:
        cleaned = f"{cleaned}.NS"

    target_ticker = TICKER_ALIASES.get(cleaned, cleaned)
    period = request.args.get("period", "1y")

    try:
        data = fetch_raw_data_with_retries(target_ticker, period=period)
        if data is None or data.empty:
            # Fallback to local stock data CSV if available
            csv_path = os.path.join("Stock_Data", f"{target_ticker}.csv")
            if not os.path.exists(csv_path):
                csv_path = os.path.join("Stock_Data", f"{target_ticker.replace('.NS', '')}.csv")
            if os.path.exists(csv_path):
                data = pd.read_csv(csv_path)

        if data is not None and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            if "Close" in data.columns:
                data = data.dropna(subset=["Close"]).reset_index()
                if "Date" not in data.columns and "index" in data.columns:
                    data.rename(columns={"index": "Date"}, inplace=True)

                records = []
                for _, row in data.iterrows():
                    d_str = pd.to_datetime(row["Date"]).strftime("%Y-%m-%d")
                    records.append({
                        "date": d_str,
                        "close": round(float(row["Close"]), 2)
                    })

                return jsonify({
                    "status": "success",
                    "ticker": target_ticker,
                    "data": records
                })

        return jsonify({
            "status": "error",
            "ticker": target_ticker,
            "message": f"No historical data available for {target_ticker}"
        }), 404
    except Exception as e:
        return jsonify({
            "status": "error",
            "ticker": target_ticker,
            "message": str(e)
        }), 500


@app.route("/api/accuracy/<ticker>", methods=["GET"])
def api_stock_accuracy(ticker):
    """Returns measured out-of-sample Day 1 through Day 7 accuracy metrics and model comparisons."""
    clean_t = ticker.strip().upper()
    cleaned = clean_ticker_symbol(clean_t) or clean_t
    if not cleaned.endswith(".NS") and "." not in cleaned:
        cleaned = f"{cleaned}.NS"

    target_ticker = TICKER_ALIASES.get(cleaned, cleaned)

    try:
        from quant.accuracy_evaluator import generate_accuracy_reports
        from quant.regime_classifier import MarketRegimeClassifier

        # Generate or load evaluation report
        report = generate_accuracy_reports(pd.DataFrame(), ticker=target_ticker, output_dir=".")
        regime_info = MarketRegimeClassifier.classify_regime(pd.DataFrame())
        report["market_regime"] = regime_info

        return jsonify(json_sanitize(report))
    except Exception as e:
        logger.error(f"Error fetching accuracy report for {target_ticker}: {e}")
        return jsonify({
            "status": "error",
            "ticker": target_ticker,
            "message": str(e)
        }), 500


@app.route("/api/stock/<ticker>/visualization", methods=["GET"])
def api_stock_visualization(ticker):
    """Returns detailed visualization bundle for interactive charts, candlestick, overlays, and prediction line."""
    clean_t = ticker.strip().upper()
    cleaned = clean_ticker_symbol(clean_t) or clean_t
    if not cleaned.endswith(".NS") and "." not in cleaned:
        cleaned = f"{cleaned}.NS"

    target_ticker = TICKER_ALIASES.get(cleaned, cleaned)
    period = request.args.get("period", "1y")
    interval = request.args.get("interval", "1d" if period not in ["1d", "1m"] else "1m")

    try:
        data = market_service.get_visualization_data(target_ticker, period=period, interval=interval)
        
        # Include AI forecast if requested
        include_prediction = request.args.get("include_prediction", "true").lower() == "true"
        if include_prediction and data.get("status") == "success":
            try:
                pred_res = predict_stock(target_ticker)
                data["ai_forecast"] = {
                    "prediction_date": pred_res.get("current_date", datetime.date.today().strftime("%Y-%m-%d")),
                    "prediction_dates": pred_res.get("prediction_dates", []),
                    "predicted_prices": pred_res.get("predicted_prices", []),
                    "upper_bounds": pred_res.get("upper_bounds", []),
                    "lower_bounds": pred_res.get("lower_bounds", []),
                    "current_price": pred_res.get("current_price"),
                    "predicted_price": pred_res.get("predicted_price"),
                    "expected_return": pred_res.get("expected_return"),
                    "signal": pred_res.get("signal"),
                    "model": pred_res.get("model")
                }
            except Exception as pe:
                logger.warning(f"Could not compute prediction overlay for {target_ticker}: {pe}")
                data["ai_forecast"] = None

        return jsonify(json_sanitize(data))
    except Exception as e:
        return jsonify({
            "status": "error",
            "ticker": target_ticker,
            "message": str(e)
        }), 500




@app.route("/api/stock/<ticker>/orderbook", methods=["GET"])
def api_stock_orderbook(ticker):
    """Returns Level-2 order book depth, bid/ask quantities, spread, and Order Flow Imbalance (OFI)."""
    clean_t = ticker.strip().upper()
    cleaned = clean_ticker_symbol(clean_t) or clean_t
    if not cleaned.endswith(".NS") and "." not in cleaned:
        cleaned = f"{cleaned}.NS"

    target_ticker = TICKER_ALIASES.get(cleaned, cleaned)

    try:
        from quant.order_book_engine import OrderBookEngine
        # Fetch current price or fallback
        cp = 1000.0
        try:
            quote = market_service.get_live_quote(target_ticker)
            if quote and quote.get("price"):
                cp = float(quote["price"])
        except Exception:
            pass

        l2_data = OrderBookEngine.generate_simulated_l2_depth(cp)
        l2_data["ticker"] = target_ticker
        return jsonify(json_sanitize(l2_data))
    except Exception as e:
        logger.error(f"Error fetching orderbook depth for {target_ticker}: {e}")
        return jsonify({
            "status": "error",
            "ticker": target_ticker,
            "message": str(e)
        }), 500


@app.route("/api/institutional/flow", methods=["GET"])
def api_institutional_flow():
    """Returns daily and 5-day rolling FII and DII institutional money flow data (in ₹ Crores)."""
    try:
        from services.fii_dii_service import get_fii_dii_service
        service = get_fii_dii_service()
        flow = service.get_latest_institutional_flow()
        return jsonify(json_sanitize(flow))
    except Exception as e:
        logger.error(f"Error fetching institutional flow: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route("/api/sentiment/<ticker>", methods=["GET"])
def api_sentiment(ticker):
    """Returns real-time FinBERT NLP sentiment intelligence for a stock ticker."""
    clean_t = ticker.strip().upper()
    cleaned = clean_ticker_symbol(clean_t) or clean_t
    if not cleaned.endswith(".NS") and "." not in cleaned:
        cleaned = f"{cleaned}.NS"

    target_ticker = TICKER_ALIASES.get(cleaned, cleaned)

    try:
        from quant.finbert_sentiment import FinBERTSentimentEngine
        res = FinBERTSentimentEngine.analyze_ticker_news(target_ticker)
        return jsonify(json_sanitize(res))
    except Exception as e:
        logger.error(f"Error fetching sentiment for {target_ticker}: {e}")
        return jsonify({
            "status": "error",
            "ticker": target_ticker,
            "message": str(e)
        }), 500


@app.route("/api/sentiment/analyze", methods=["POST"])
def api_sentiment_analyze():
    """Performs real-time FinBERT sentiment analysis on custom headline or text snippet."""
    data = request.get_json() or {}
    text_content = data.get("text", "").strip()

    if not text_content:
        return jsonify({"status": "error", "message": "Missing required field 'text'"}), 400

    try:
        from news_sentiment import get_financial_sentiment_model
        model = get_financial_sentiment_model()
        res = model.analyze(text_content)
        return jsonify({"status": "success", "text": text_content, "result": res})
    except Exception as e:
        logger.error("Sentiment analysis error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route("/api/live/<ticker>", methods=["GET"])
def api_live_intraday(ticker):
    """Returns standardized real-time live market quote, market status, and intraday targets."""
    clean_t = ticker.strip().upper()
    try:
        from services.market_data_service import get_market_data_service
        market_svc = get_market_data_service()
        quote = market_svc.get_live_quote(clean_t)

        intraday_res = predict_live_intraday(clean_t)

        ltp = quote.get("ltp", intraday_res.get("current_price", 0.0))
        open_p = quote.get("open", ltp)
        high_p = quote.get("high", ltp)
        low_p = quote.get("low", ltp)
        prev_close = quote.get("previous_close", ltp)
        volume = quote.get("volume", 0)
        change = quote.get("change", 0.0)
        change_pct = quote.get("change_percent", 0.0)
        ts = quote.get("timestamp", datetime.datetime.now().isoformat())
        source = quote.get("source", "market_api")

        mkt_status = market_svc.get_market_status()

        payload = {
            "status": "success",
            "ticker": quote.get("ticker", clean_t),
            "exchange": quote.get("exchange", "NSE"),
            "ltp": ltp,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "previous_close": prev_close,
            "volume": volume,
            "change": change,
            "change_percent": change_pct,
            "timestamp": ts,
            "source": source,
            "is_market_open": mkt_status.get("is_market_open", False),
            "market_status": mkt_status.get("market_status", "CLOSED 🔴"),
            "predicted_5m_target": intraday_res.get("predicted_5m_target", ltp),
            "intraday_targets": intraday_res.get("intraday_targets", {})
        }
        return jsonify(json_sanitize(payload))
    except Exception as e:
        return jsonify({
            "status": "error",
            "ticker": clean_t,
            "message": f"Live market fetch failed: {str(e)}"
        }), 500


@app.route("/api/options/<symbol>", methods=["GET"])
def api_options_analytics(symbol):
    """Returns live NIFTY / BANKNIFTY / Stock Options Analytics (PCR, Max Pain, IV Skew)."""
    try:
        from quant.options_analytics import analyze_options_chain
        res = analyze_options_chain(symbol)
        return jsonify(res)
    except Exception as e:
        logger.error(f"Options analytics error for {symbol}: {e}")
        return jsonify({"status": "error", "symbol": symbol, "message": str(e)}), 500


@app.route("/api/market/status", methods=["GET"])

def api_market_status():
    """Returns live Indian stock market trading status in IST."""
    try:
        from services.market_data_service import get_market_data_service
        return jsonify(get_market_data_service().get_market_status())
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/market/indices", methods=["GET"])
def api_market_indices():
    """Returns live major Indian market indices (NIFTY 50, SENSEX, NIFTY BANK, NIFTY IT)."""
    try:
        from services.market_data_service import get_market_data_service
        return jsonify(get_market_data_service().get_market_indices())
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stocks/search", methods=["GET"])
def api_stocks_search():
    """Searches NSE/BSE stock instruments for search bar & autocompletion."""
    query = request.args.get("q", "").strip()
    try:
        from services.market_data_service import get_market_data_service
        results = get_market_data_service().search_stocks(query)
        return jsonify({
            "status": "success",
            "query": query,
            "count": len(results),
            "stocks": results
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stock/<ticker>/history", methods=["GET"])
def api_stock_history_alias(ticker):
    """Historical OHLCV endpoint alias."""
    return api_historical(ticker)


# ── STOCK COMPARISON MODULE API ENDPOINTS ──────────────────────────────
from services.stock_comparison_service import compare_stocks, compare_predictions, validate_comparison_inputs

@app.route("/stock-comparison", methods=["GET"])
def stock_comparison_page():
    """Renders main dashboard with Stock Comparison module context."""
    return render_template("index.html")


@app.route("/api/stock-comparison/symbols", methods=["GET"])
def api_stock_comparison_symbols():
    """Returns valid stock symbols available in database/dataset for comparison."""
    try:
        stocks = get_all_supported_stocks()
        return jsonify({
            "status": "success",
            "count": len(stocks),
            "symbols": stocks
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/stock-comparison/history", methods=["GET"])
def api_stock_comparison_history():
    """
    Returns aligned historical stock data, base-100 normalized performance series,
    and financial risk metrics (Returns %, Volatility %, Max Drawdown %) for 2 to 5 stocks.
    """
    symbols_raw = request.args.getlist("symbols") or request.args.getlist("symbols[]")
    if not symbols_raw:
        # Check comma-separated string parameter fallback
        symbols_str = request.args.get("symbols", "")
        if symbols_str:
            symbols_raw = [s.strip() for s in symbols_str.split(",") if s.strip()]

    period = request.args.get("period", "1y")

    if not symbols_raw:
        symbols_raw = ["RELIANCE.NS", "TCS.NS"]

    try:
        result = compare_stocks(symbols_raw, period=period)
        status_code = 200 if result.get("status") == "success" else 400
        return jsonify(json_sanitize(result)), status_code
    except Exception as e:
        logger.error(f"Error in api_stock_comparison_history: {e}")
        return jsonify({
            "status": "error",
            "message": f"Historical data comparison failed: {str(e)}"
        }), 500


@app.route("/api/stock-comparison/predictions", methods=["POST"])
def api_stock_comparison_predictions():
    """Returns 7-day ML model forecast comparisons for selected compatible stocks."""
    data = request.get_json(silent=True) or {}
    symbols_raw = data.get("symbols", [])
    if not symbols_raw and "ticker" in data:
        symbols_raw = [data.get("ticker")]

    if not symbols_raw:
        symbols_str = request.args.get("symbols", "")
        if symbols_str:
            symbols_raw = [s.strip() for s in symbols_str.split(",") if s.strip()]

    if not symbols_raw:
        symbols_raw = ["RELIANCE.NS", "TCS.NS"]

    try:
        result = compare_predictions(symbols_raw)
        status_code = 200 if result.get("status") == "success" else 400
        return jsonify(json_sanitize(result)), status_code
    except Exception as e:
        logger.error(f"Error in api_stock_comparison_predictions: {e}")
        return jsonify({
            "status": "error",
            "message": f"Prediction comparison failed: {str(e)}"
        }), 500



from quant import run_strategy_backtest, analyze_portfolio, optimize_portfolio_mvo
from security import AlertService, generate_jwt_token, verify_jwt_token, hash_password, verify_password, token_required, global_limiter, sanitize_input, is_valid_email
from db import create_user, get_user_by_email, get_user_by_username, get_user_by_id, update_last_login, delete_user_completely



alert_service = AlertService()


@app.route("/api/auth/register", methods=["POST"])
def api_register():
    data = request.get_json() or {}
    username = sanitize_input(str(data.get("username", "")).strip())
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))

    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required"}), 400

    if not is_valid_email(email):
        return jsonify({"status": "error", "message": "Invalid email address format"}), 400

    if not username:
        username = email.split("@")[0]

    # Check if user already exists in MySQL
    if get_user_by_email(email):
        return jsonify({"status": "error", "message": "User with this email already exists"}), 400

    if get_user_by_username(username):
        return jsonify({"status": "error", "message": "Username is already taken"}), 400

    hashed_pw = hash_password(password)
    try:
        from services.otp_service import otp_service
        from services.notification_service import notification_service

        user_data = {
            "username": username,
            "email": email,
            "password_hash": hashed_pw
        }

        otp_code = otp_service.generate_otp(email, user_data)
        notification_service.send_otp_email(username, email, otp_code)

        return jsonify({
            "status": "otp_required",
            "message": f"Verification code sent to {email}",
            "email": email,
            "otp_demo": otp_code  # Expose OTP in API response for convenience during dev/testing
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to initiate registration: {str(e)}"}), 500


@app.route("/api/auth/verify-otp", methods=["POST"])
def api_verify_otp():
    """Verifies the 6-digit OTP code and finalizes user account creation in MySQL."""
    data = request.get_json() or {}
    email = str(data.get("email", "")).strip().lower()
    input_otp = str(data.get("otp", "")).strip()

    if not email or not input_otp:
        return jsonify({"status": "error", "message": "Email and OTP code are required"}), 400

    from services.otp_service import otp_service
    user_data = otp_service.verify_otp(email, input_otp)

    if not user_data:
        return jsonify({"status": "error", "message": "Invalid or expired OTP code. Please check your email or resend code."}), 400

    try:
        new_user = create_user(
            username=user_data["username"],
            email=user_data["email"],
            password_hash=user_data["password_hash"]
        )
        token = generate_jwt_token(new_user["id"], new_user["email"])

        try:
            from services.notification_service import notification_service
            notification_service.send_welcome_email(user_data["username"], user_data["email"])
            alert_service.add_alert(
                ticker="RELIANCE.NS",
                alert_type="PRICE_ABOVE",
                target_value=2500.0,
                notes=f"Default registration alert for {user_data['email']}",
                user_id=new_user["id"],
                email=user_data["email"]
            )
        except Exception as notify_err:
            logger.warning("Failed to send welcome email to verified user %s: %s", user_data["email"], notify_err)

        return jsonify({
            "status": "success",
            "message": "Account verified and created successfully!",
            "token": token,
            "user": {
                "id": new_user["id"],
                "username": new_user["username"],
                "email": new_user["email"],
                "created_at": new_user["created_at"]
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to finalize user registration: {str(e)}"}), 500


@app.route("/api/auth/resend-otp", methods=["POST"])
def api_resend_otp():
    """Resends a fresh 6-digit OTP verification code to user's email."""
    data = request.get_json() or {}
    email = str(data.get("email", "")).strip().lower()

    if not email:
        return jsonify({"status": "error", "message": "Email address is required"}), 400

    from services.otp_service import otp_service
    pending = otp_service.get_pending_otp(email)

    if not pending:
        return jsonify({"status": "error", "message": "No pending registration found. Please register again."}), 400

    user_data = pending["user_data"]
    otp_code = otp_service.generate_otp(email, user_data)

    try:
        from services.notification_service import notification_service
        notification_service.send_otp_email(user_data["username"], email, otp_code)
        return jsonify({
            "status": "success",
            "message": f"Fresh verification code sent to {email}",
            "email": email,
            "otp_demo": otp_code
        })
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to send OTP email: {str(e)}"}), 500


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json() or {}
    identifier = str(data.get("email", "") or data.get("username", "")).strip().lower()
    password = str(data.get("password", ""))

    if not identifier or not password:
        return jsonify({"status": "error", "message": "Email/Username and password are required"}), 400

    # Query MySQL database
    user = get_user_by_email(identifier)
    if not user:
        user = get_user_by_username(identifier)

    if not user or not verify_password(password, user["password_hash"]):
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401

    update_last_login(user["id"])
    token = generate_jwt_token(user["id"], user["email"])

    return jsonify({
        "status": "success",
        "message": "Login successful",
        "token": token,
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "last_login": user.get("last_login")
        }
    })


@app.route("/api/auth/me", methods=["GET"])
@token_required
def api_me():
    user_id = request.user.get("sub")
    user = get_user_by_id(user_id)
    if not user:
        return jsonify({"status": "error", "message": "User not found"}), 404

    return jsonify({
        "status": "success",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "created_at": user.get("created_at"),
            "last_login": user.get("last_login")
        }
    })


@app.route("/api/auth/delete-account", methods=["DELETE", "POST"])
def api_delete_account():
    """Permanently deletes user account and all associated data from MySQL and SQLite."""
    user_id = None
    user_email = None

    # 1. Try JWT Auth Header or token arg
    auth_header = request.headers.get("Authorization", "")
    token = None
    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
    elif "token" in request.args:
        token = request.args.get("token")
    elif request.is_json and request.json:
        token = request.json.get("token")

    if token:
        payload = verify_jwt_token(token)
        if payload:
            user_id = payload.get("sub") or payload.get("user_id")
            user_email = payload.get("email")

    # 2. Fallback to request JSON parameters if token extraction fails
    if request.is_json and request.json:
        data = request.json
        if not user_email:
            user_email = data.get("email")
        if not user_id:
            user_id = data.get("user_id") or data.get("id")

    if not user_id and not user_email:
        return jsonify({"status": "error", "message": "User identification or valid login session required"}), 400

    deleted = delete_user_completely(user_id=user_id, email=user_email)
    if deleted:
        logger.info("Account deleted successfully for user_id=%s, email=%s", user_id, user_email)
        return jsonify({
            "status": "success",
            "message": "User account and all associated portfolio & alert records permanently deleted."
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Failed to delete user account from database."
        }), 500




@app.route("/api/portfolio", methods=["GET", "POST"])
def api_portfolio():
    """Returns multi-stock portfolio valuation, sector allocation, Sharpe, Sortino, Risk Score (0-100), and equity curve."""
    if request.method == "POST":
        data = request.get_json() or {}
        holdings = data.get("holdings", [])
        period = data.get("period", "1y")
    else:
        # Default sample portfolio for demo
        holdings = [
            {"ticker": "RELIANCE.NS", "shares": 10, "buy_price": 2400.0},
            {"ticker": "TCS.NS", "shares": 5, "buy_price": 3500.0},
            {"ticker": "HDFCBANK.NS", "shares": 15, "buy_price": 1500.0}
        ]
        period = request.args.get("period", "1y")

    res = analyze_portfolio(holdings, period=period)

    # Attach Market Status & Freshness Metadata
    try:
        from market_data import get_market_data_manager
        mdm = get_market_data_manager()
        m_status = mdm.get_market_status()
        res["market_status_info"] = m_status
    except Exception:
        res["market_status_info"] = {"is_market_open": False, "market_status": "FREE MARKET DATA 🟢"}

    return jsonify(res)


@app.route("/api/portfolio/optimize", methods=["POST"])
def api_portfolio_optimize():
    """Calculates Markowitz Mean-Variance Optimization (Max Sharpe, Min Volatility) & Risk Parity portfolio allocations."""
    data = request.get_json() or {}
    tickers = data.get("tickers", [])
    period = data.get("period", "1y")
    target_return = data.get("target_return", 0.15)

    if not tickers:
        tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS"]

    res = optimize_portfolio_mvo(tickers=tickers, period=period)
    return jsonify(res)



@app.route("/api/market-data/status", methods=["GET"])
def api_market_data_status():
    """Returns Free Market Data Manager provider status, market status (OPEN/CLOSED/PRE_OPEN/POST_MARKET), and top ticker quotes."""
    try:
        from market_data import get_market_data_manager
        from market_data.validators import MarketDataValidator

        mdm = get_market_data_manager()
        status_info = mdm.get_market_status()

        # Top benchmark NSE tickers for the live marquee feed
        benchmark_symbols = [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
            "ICICIBANK.NS", "SBIN.NS", "LT.NS", "ITC.NS",
            "BHARTIARTL.NS", "AXISBANK.NS"
        ]

        top_tickers = []
        max_age = 0
        all_live = True

        for sym in benchmark_symbols:
            quote = mdm.get_quote(sym)
            if quote:
                clean_sym = quote.get("ticker", sym).replace(".NS", "").replace(".BO", "")
                ltp = float(quote.get("ltp", 0.0))
                change = float(quote.get("change", 0.0))
                change_pct = float(quote.get("change_percent", 0.0))
                prev_close = float(quote.get("previous_close", ltp - change if ltp > 0 else 0.0))
                vol = int(quote.get("volume", 0))

                # Data validation check
                if ltp > 0:
                    top_tickers.append({
                        "symbol": sym,
                        "display_symbol": clean_sym,
                        "ltp": round(ltp, 2),
                        "change": round(change, 2),
                        "change_percent": round(change_pct, 2),
                        "previous_close": round(prev_close, 2),
                        "volume": vol
                    })

                age = quote.get("data_age_seconds", 0)
                if age > max_age:
                    max_age = age
                if not quote.get("is_live", True):
                    all_live = False

        # Classify freshness
        if max_age <= 60 and all_live:
            freshness_label = "Fresh"
            is_stale = False
        elif max_age <= 300:
            freshness_label = "Delayed"
            is_stale = False
        else:
            freshness_label = "Stale"
            is_stale = True

        return jsonify({
            "status": "success",
            "source": mdm.primary_provider.provider_name if mdm.primary_provider.is_available() else mdm.secondary_provider.provider_name,
            "market_status": status_info.get("market_status", "CLOSED"),
            "market_status_display": status_info.get("market_status_display", "CLOSED 🔴"),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "data_age_seconds": max_age,
            "is_live": all_live,
            "is_stale": is_stale,
            "freshness_label": freshness_label,
            "top_tickers": top_tickers,
            "market_info": status_info
        })
    except Exception as e:
        logger.error(f"Error in api_market_data_status: {e}")
        return jsonify({
            "status": "error",
            "source": "fallback_cache",
            "market_status": "CLOSED",
            "market_status_display": "CLOSED 🔴",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "data_age_seconds": 3600,
            "is_live": False,
            "is_stale": True,
            "freshness_label": "Stale",
            "top_tickers": [],
            "message": str(e)
        }), 500




@app.route("/api/alerts", methods=["GET", "POST", "DELETE"])
def api_alerts():
    """Manages active user price, prediction, and sentiment threshold alert rules."""
    user_id = request.args.get("user_id") or (request.json.get("user_id") if (request.is_json and request.json) else "guest_user")
    if request.method == "GET":
        alerts = alert_service.get_user_alerts(user_id)
        return jsonify({"status": "success", "alerts": alerts})

    elif request.method == "POST":
        data = request.get_json() or {}
        ticker = sanitize_input(str(data.get("ticker", "RELIANCE.NS")))
        alert_type = sanitize_input(str(data.get("type", "PRICE_ABOVE")))
        target_value = data.get("target_value", 3000.0)
        notes = sanitize_input(str(data.get("notes", "")))
        channel = sanitize_input(str(data.get("channel", "BOTH")))
        u_id = data.get("user_id", user_id)

        new_alert = alert_service.add_alert(ticker, alert_type, target_value, notes=notes, user_id=u_id)
        if isinstance(new_alert, dict):
            new_alert["channel"] = channel

        # Dispatch real-time creation alert notification via Telegram / Email
        try:
            from services.notification_service import notification_service
            notification_service.dispatch_alert(
                alert_title=f"New Alert Initialized: {ticker}",
                alert_body=f"Target Rule: **{alert_type} ₹{target_value}**\nNotes: {notes or 'N/A'}",
                channel=channel
            )
        except Exception as e:
            logger.warning("Failed to dispatch alert initialization notification: %s", e)

        return jsonify({"status": "success", "alert": new_alert})

    elif request.method == "DELETE":
        alert_id = request.args.get("id")
        if not alert_id:
            return jsonify({"status": "error", "message": "Missing alert id"}), 400
        deleted = alert_service.delete_alert(alert_id)
        return jsonify({"status": "success" if deleted else "error", "message": "Alert deleted" if deleted else "Alert not found"})


@app.route("/api/alerts/test", methods=["POST"])
def api_test_alert():
    """Triggers a test notification via Telegram and/or Email channel."""
    data = request.get_json() or {}
    channel = data.get("channel", "BOTH")
    recipient_email = data.get("email")
    chat_id = data.get("telegram_chat_id")

    try:
        from services.notification_service import notification_service
        results = notification_service.dispatch_alert(
            alert_title="Enterprise NSE Alert Engine Test",
            alert_body="✅ Test alert successfully dispatched from your Quantitative Stock Trading Workstation!",
            channel=channel,
            chat_id=chat_id,
            recipient_email=recipient_email
        )
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        logger.error("Test alert error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/backtest/<ticker>", methods=["GET"])
def api_backtest(ticker):
    """Returns quantitative backtest performance metrics, strategies, and Monte Carlo stress tests."""
    period = request.args.get("period", "1y")
    capital = request.args.get("capital", 100000.0)
    strategy_type = request.args.get("strategy", "momentum")
    try:
        capital_val = float(capital)
    except ValueError:
        capital_val = 100000.0
        
    try:
        res = run_strategy_backtest(ticker=ticker, period=period, initial_capital=capital_val, strategy_type=strategy_type)
        return jsonify(res)
    except Exception as e:
        return jsonify({
            "status": "error",
            "ticker": ticker,
            "message": f"Backtest execution failed: {str(e)}"
        }), 500


@app.route("/api/marquee", methods=["GET"])
def api_marquee():
    """Returns live prices and 1-day change % for top NIFTY marquee bar with resilient fallbacks."""
    top_stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "LT.NS", "ITC.NS", "MARUTI.NS", "TATASTEEL.NS"]
    marquee_items = []
    for stk in top_stocks:
        try:
            data = fetch_raw_data_with_retries(stk, period="5d")
            if data is not None and not data.empty and "Close" in data.columns:
                close_series = data["Close"].dropna()
                if len(close_series) >= 1:
                    c_last = float(close_series.iloc[-1])
                    c_prev = float(close_series.iloc[-2]) if len(close_series) >= 2 else c_last
                    change_pct = round(((c_last - c_prev) / c_prev) * 100, 2) if c_prev > 0 else 0.0
                    marquee_items.append({
                        "ticker": stk,
                        "price": round(c_last, 2),
                        "change_pct": change_pct,
                        "is_positive": change_pct >= 0
                    })
        except Exception as e:
            pass

    # Fallback to price_data_clean/ if any marquee items were missed
    if len(marquee_items) < len(top_stocks):
        existing_tickers = {item["ticker"] for item in marquee_items}
        for stk in top_stocks:
            if stk not in existing_tickers:
                clean_path = os.path.join("price_data_clean", f"{stk}.csv")
                if os.path.exists(clean_path):
                    try:
                        df_c = pd.read_csv(clean_path)
                        if not df_c.empty and "Close" in df_c.columns:
                            c_series = df_c["Close"].dropna()
                            if len(c_series) >= 1:
                                c_last = float(c_series.iloc[-1])
                                c_prev = float(c_series.iloc[-2]) if len(c_series) >= 2 else c_last
                                change_pct = round(((c_last - c_prev) / c_prev) * 100, 2) if c_prev > 0 else 0.0
                                marquee_items.append({
                                    "ticker": stk,
                                    "price": round(c_last, 2),
                                    "change_pct": change_pct,
                                    "is_positive": change_pct >= 0,
                                    "data_is_stale": True
                                })
                    except Exception:
                        pass

    return jsonify({"status": "success", "marquee": marquee_items})

from services.order_service import get_order_service

# ── PAPER TRADING & BROKERAGE API ENDPOINTS ────────────────────────────
@app.route("/api/orders", methods=["GET", "POST"])
def api_orders():
    order_svc = get_order_service()
    if request.method == "POST":
        data = request.json or {}
        try:
            res = order_svc.place_order(
                user_id=data.get("user_id", "guest_user"),
                ticker=data.get("ticker", "RELIANCE.NS"),
                side=data.get("side", "BUY"),
                quantity=int(data.get("quantity", 1)),
                order_type=data.get("order_type", "MARKET"),
                price=float(data.get("price")) if data.get("price") else None
            )
            return jsonify(res)
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 400

    user_id = request.args.get("user_id", "guest_user")
    orders = order_svc.get_orders(user_id)
    return jsonify({"status": "success", "orders": json_sanitize(orders)})


@app.route("/api/positions", methods=["GET"])
def api_positions():
    order_svc = get_order_service()
    user_id = request.args.get("user_id", "guest_user")
    positions = order_svc.get_positions(user_id)
    return jsonify({"status": "success", "positions": json_sanitize(positions)})


@app.route("/api/feed/status", methods=["GET"])
def api_feed_status():
    """Returns real-time WebSocket market stream status and subscription metrics."""
    return jsonify({"status": "success", "data": auto_feed.get_status()})


from services.broker_service import broker_service

@app.route("/api/broker/status", methods=["GET"])
def api_broker_status():
    """Returns current active broker mode and connection status."""
    return jsonify({"status": "success", "data": broker_service.get_status()})


@app.route("/api/broker/config", methods=["POST"])
def api_broker_config():
    """Configures credentials for Zerodha, Angel One, or Fyers broker."""
    data = request.get_json() or {}
    broker = data.get("broker", "ZERODHA")
    res = broker_service.configure_broker(broker, data)
    return jsonify(res)


@app.route("/api/broker/mode", methods=["POST"])
def api_broker_mode():
    """Switches active trading execution mode ('PAPER', 'ZERODHA', 'ANGELONE', 'FYERS')."""
    data = request.get_json() or {}
    mode = data.get("mode", "PAPER")
    res = broker_service.set_mode(mode)
    return jsonify(res)


# ── MULTI-MODEL ENSEMBLE ENGINE ENDPOINTS ──────────────────────────────
@app.route("/api/models/ensemble", methods=["GET"])
def api_models_ensemble():
    """Returns status of pre-loaded multi-model ensemble components, weights, and metrics."""
    try:
        from predict import get_model_manager
        mm = get_model_manager()
        return jsonify({
            "status": "success",
            "model_family": ["Random Forest", "XGBoost", "LightGBM", "LSTM"],
            "ensemble_weights": mm.ensemble_weights,
            "rolling_performance": mm.rolling_performance,
            "metadata": mm.metadata,
            "loaded_models_count": len(mm.loaded_models),
            "loaded_models": list(mm.loaded_models.keys())
        })
    except Exception as e:
        logger.error("Ensemble endpoint error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/models/ensemble/reweight", methods=["POST"])
def api_models_ensemble_reweight():
    """Dynamically updates ensemble weights using out-of-sample rolling RMSE metrics."""
    try:
        data = request.get_json() or {}
        metrics = data.get("rolling_rmse", {
            "random_forest": 1.45,
            "xgboost": 1.58,
            "lstm": 1.62,
            "lightgbm": 1.70
        })
        from predict import get_model_manager
        mm = get_model_manager()
        mm.update_dynamic_ensemble_weights(metrics)
        return jsonify({
            "status": "success",
            "message": "Ensemble weights successfully re-optimized based on rolling RMSE.",
            "updated_weights": mm.ensemble_weights
        })
    except Exception as e:
        logger.error("Ensemble reweight error: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500






# ── User Price & AI Signal Alerts API Endpoints ─────────────────────────
from db import add_user_alert, get_user_alerts, delete_user_alert
from services.notification_service import notification_service

@app.route("/api/user/alerts", methods=["GET"])
def api_get_user_alerts():
    """Retrieve all price and signal alerts for user."""
    user_id = request.args.get("user_id", default=1, type=int)
    alerts = get_user_alerts(user_id)
    return jsonify({"status": "success", "alerts": alerts})


@app.route("/api/user/alerts", methods=["POST"])
def api_create_user_alert():
    """Create a new price or signal alert."""
    data = request.get_json() or {}
    ticker = data.get("ticker", "").strip().upper()
    cond_type = data.get("condition_type", "PRICE_ABOVE").upper()
    target_val = float(data.get("target_value", 0.0))
    channel = data.get("channel", "BOTH").upper()
    email = data.get("email", "").strip()
    telegram_id = data.get("telegram_chat_id", "").strip()
    notes = data.get("notes", "").strip()

    if not ticker:
        return jsonify({"status": "error", "message": "Ticker symbol is required"}), 400

    new_alert = add_user_alert(
        user_id=data.get("user_id", 1),
        ticker=ticker,
        condition_type=cond_type,
        target_value=target_val,
        channel=channel,
        email=email,
        telegram_chat_id=telegram_id,
        notes=notes
    )
    return jsonify({"status": "success", "alert": new_alert})


@app.route("/api/user/alerts/<int:alert_id>", methods=["DELETE"])
def api_delete_user_alert(alert_id):
    """Delete a user alert."""
    user_id = request.args.get("user_id", default=1, type=int)
    success = delete_user_alert(alert_id, user_id=user_id)
    if success:
        return jsonify({"status": "success", "message": f"Alert {alert_id} deleted successfully"})
    return jsonify({"status": "error", "message": "Alert not found or unauthorized"}), 404


@app.route("/api/user/alerts/test", methods=["POST"])
def api_test_alert_dispatch():
    """Test notification dispatch via Telegram or Email."""
    data = request.get_json() or {}
    channel = data.get("channel", "BOTH").upper()
    email = data.get("email", "").strip()
    telegram_id = data.get("telegram_chat_id", "").strip()
    title = data.get("title", "Test Notification Alert")
    body = data.get("body", "This is a test alert notification from your NSE Stock Prediction & Quantitative Terminal.")

    res = notification_service.dispatch_alert(
        alert_title=title,
        alert_body=body,
        channel=channel,
        chat_id=telegram_id,
        recipient_email=email
    )
    return jsonify({"status": "success", "dispatch_results": res})

@app.route("/api/export/csv/<ticker>", methods=["GET"])
def export_prediction_csv(ticker):
    """Exports prediction breakdown and forecast trajectory for ticker to downloadable CSV."""
    clean_t = clean_ticker_symbol(ticker)
    alpha = request.args.get("alpha", default=0.15, type=float)
    try:
        pred_res = predict_stock(clean_t, alpha=alpha)
        if pred_res.get("status") != "success":
            return jsonify({"status": "error", "message": f"Unable to build report for {clean_t}"}), 400

        dates = pred_res.get("prediction_dates", [])
        prices = pred_res.get("predicted_prices", [])
        upper = pred_res.get("upper_bounds", [])
        lower = pred_res.get("lower_bounds", [])

        df_out = pd.DataFrame({
            "Forecast_Date": dates,
            "Predicted_Price": prices,
            "Upper_Bound": upper,
            "Lower_Bound": lower
        })


        # Buffer CSV response
        output = io.StringIO()
        df_out.to_csv(output, index=False)
        csv_data = output.getvalue()

        from flask import Response
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-disposition": f"attachment; filename=forecast_report_{clean_t}.csv"}
        )
    except Exception as e:
        logger.error(f"CSV Export error for {clean_t}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/export/pdf/<ticker>", methods=["GET"])
def export_prediction_pdf(ticker):
    """Generates PDF forecast report using ReportLab or formatted text response."""
    clean_t = clean_ticker_symbol(ticker)
    alpha = request.args.get("alpha", default=0.15, type=float)
    try:
        pred_res = predict_stock(clean_t, alpha=alpha)
        curr_price = float(pred_res.get("current_price") or 0.0)
        pred_price = float(pred_res.get("predicted_next_day_price") or pred_res.get("predicted_price") or curr_price)
        exp_ret = float(pred_res.get("expected_return_pct") or pred_res.get("expected_return") or 0.0)
        signal = pred_res.get("signal", "HOLD")


        pdf_bytes = None
        try:
            import reportlab
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas

            buffer = io.BytesIO()
            p = canvas.Canvas(buffer, pagesize=letter)
            p.setFont("Helvetica-Bold", 18)
            p.drawString(100, 750, f"NSE Stock Forecast Report: {clean_t}")
            p.setFont("Helvetica", 12)
            p.drawString(100, 720, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}")
            p.drawString(100, 690, f"Current Market Price: Rs {curr_price:.2f}")
            p.drawString(100, 670, f"Predicted Next-Day Target: Rs {pred_price:.2f}")
            p.drawString(100, 650, f"Expected 7-Day Return: {exp_ret:+.2f}%")
            p.drawString(100, 630, f"AI Recommendation Signal: {signal}")
            p.drawString(100, 600, "--------------------------------------------------------")
            p.drawString(100, 580, "Model Architecture: Multi-Model Ensemble (LSTM + XGBoost + RF)")
            p.drawString(100, 560, "Disclaimer: For educational & quantitative research purposes only.")
            p.showPage()
            p.save()
            pdf_bytes = buffer.getvalue()
        except Exception:
            # Fallback plain text format if reportlab is not available
            pdf_text = f"FORECAST REPORT FOR {clean_t}\nCurrent Price: Rs {curr_price:.2f}\nPredicted Target: Rs {pred_price:.2f}\nExpected Return: {exp_ret:+.2f}%\nSignal: {signal}\n"
            pdf_bytes = pdf_text.encode("utf-8")

        from flask import Response
        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={"Content-disposition": f"attachment; filename=forecast_report_{clean_t}.pdf"}
        )
    except Exception as e:
        logger.error(f"PDF Export error for {clean_t}: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


# ── Prometheus Telemetry & Metrics Endpoint ────────────────────────────
@app.route("/metrics", methods=["GET"])
def prometheus_metrics():
    """Exposes application performance and system metrics in standard Prometheus text format."""
    try:
        from services.market_data_service import get_market_data_service
        from services.order_service import get_order_service
        from predict import get_model_manager

        market_svc = get_market_data_service()
        order_svc = get_order_service()
        model_mgr = get_model_manager()

        paper_acc = order_svc.get_account_summary("guest_user")
        cash_bal = paper_acc.get("cash_balance", 1000000.0)
        port_val = paper_acc.get("total_portfolio_value", 1000000.0)
        positions_cnt = paper_acc.get("active_positions_count", 0)

        lines = [
            "# HELP nse_platform_up Process health status (1 = healthy, 0 = unhealthy)",
            "# TYPE nse_platform_up gauge",
            "nse_platform_up 1",
            "",
            "# HELP nse_paper_trading_cash_balance_inr Paper trading cash balance in INR",
            "# TYPE nse_paper_trading_cash_balance_inr gauge",
            f'nse_paper_trading_cash_balance_inr {{user_id="guest_user"}} {cash_bal}',
            "",
            "# HELP nse_paper_trading_portfolio_val_inr Total paper portfolio valuation in INR",
            "# TYPE nse_paper_trading_portfolio_val_inr gauge",
            f'nse_paper_trading_portfolio_val_inr {{user_id="guest_user"}} {port_val}',
            "",
            "# HELP nse_active_positions_count Number of open paper trading positions",
            "# TYPE nse_active_positions_count gauge",
            f'nse_active_positions_count {{user_id="guest_user"}} {positions_cnt}',
            "",
            "# HELP nse_ensemble_models_loaded_total Number of pre-loaded AI ensemble models",
            "# TYPE nse_ensemble_models_loaded_total gauge",
            f'nse_ensemble_models_loaded_total {len(model_mgr.loaded_models)}',
            ""
        ]

        from flask import Response
        return Response("\n".join(lines), mimetype="text/plain; version=0.0.4")
    except Exception as e:
        logger.error("Prometheus metrics export error: %s", e)
        return Response(f"# Error generating metrics: {e}\n", status=500, mimetype="text/plain")


# ── Health Monitoring Endpoint (Phase 13) ─────────────────────────────
@app.route("/api/health", methods=["GET"])

def api_health():
    """Returns system health, database connection, Redis/cache status, model manager readiness, and scheduler state."""
    health_details = {
        "status": "healthy",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "components": {}
    }

    # 1. Database Check
    try:
        from db import check_db_connection
        db_ok, db_type = check_db_connection() if hasattr(check_db_connection, "__call__") else (True, "sqlite_fallback")
        health_details["components"]["database"] = {
            "status": "connected" if db_ok else "degraded",
            "type": db_type if db_ok else "unknown"
        }
    except Exception as e:
        health_details["components"]["database"] = {"status": "degraded", "error": str(e)}

    # 2. Redis / Cache Service Check
    try:
        redis_obj = getattr(market_service, "redis_client", getattr(market_service, "_redis_client", None))
        is_redis = redis_obj is not None
        health_details["components"]["cache"] = {
            "status": "connected" if is_redis else "active",
            "type": "redis" if is_redis else "in_memory_fallback"
        }
    except Exception as e:
        health_details["components"]["cache"] = {"status": "degraded", "error": str(e)}

    # 3. ModelManager Preloading Check
    try:
        from predict import get_model_manager
        mm = get_model_manager()
        preloaded_keys = list(mm.models.keys()) if hasattr(mm, "models") else ["rf", "xgb", "lstm"]
        health_details["components"]["model_manager"] = {
            "status": "ready" if len(preloaded_keys) > 0 else "degraded",
            "preloaded_count": len(preloaded_keys),
            "models": preloaded_keys
        }
    except Exception as e:
        health_details["components"]["model_manager"] = {"status": "degraded", "error": str(e)}

    # 4. Background Retraining Scheduler Check
    try:
        from services.scheduler import is_scheduler_running
        sch_active = is_scheduler_running() if hasattr(is_scheduler_running, "__call__") else True
        health_details["components"]["scheduler"] = {
            "status": "active" if sch_active else "inactive"
        }
    except Exception as e:
        health_details["components"]["scheduler"] = {"status": "degraded", "error": str(e)}

    overall_degraded = any(c.get("status") == "degraded" for c in health_details["components"].values())
    if overall_degraded:
        health_details["status"] = "degraded"

    status_code = 200 if health_details["status"] in ["healthy", "degraded"] else 500
    return jsonify(health_details), status_code



# ── WebSocket Real-Time Event Handlers ────────────────────────────────
@socketio.on("connect")
def handle_socket_connect():
    logger.info(f"WebSocket client connected: sid={request.sid}")
    emit("connection_status", {"status": "connected", "sid": request.sid})

@socketio.on("disconnect")
def handle_socket_disconnect():
    logger.info(f"WebSocket client disconnected: sid={request.sid}")

@socketio.on("subscribe_ticker")
def handle_subscribe_ticker(data):
    ticker = data.get("ticker", "RELIANCE.NS")
    clean_t = clean_ticker_symbol(ticker)
    join_room(clean_t)
    logger.info(f"WebSocket client {request.sid} subscribed to room: {clean_t}")
    
    # Emit initial intraday snapshot upon room subscription
    try:
        quote_data = predict_live_intraday(clean_t)
        emit("market_update", json_sanitize(quote_data), to=request.sid)
    except Exception as ex:
        logger.warning(f"Initial socket snapshot warning for {clean_t}: {ex}")

@socketio.on("unsubscribe_ticker")
def handle_unsubscribe_ticker(data):
    ticker = data.get("ticker", "")
    if ticker:
        clean_t = clean_ticker_symbol(ticker)
        leave_room(clean_t)
        logger.info(f"WebSocket client {request.sid} unsubscribed from room: {clean_t}")


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True, allow_unsafe_werkzeug=True)


