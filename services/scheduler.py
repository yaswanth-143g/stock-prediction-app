"""
services/scheduler.py — Automated Daily Model Retraining Scheduler
===================================================================
Executes background retraining jobs for ML models after market close (3:30 PM IST).
Flow: train -> validate -> temporary save -> verify -> atomic promotion -> reload.
"""

import os
import shutil
import logging
import time
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import joblib

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger("SchedulerService")
logger.setLevel(logging.INFO)

IST = timezone(timedelta(hours=5, minutes=30))

RETRAIN_HOUR = int(os.environ.get("MODEL_RETRAIN_HOUR", 16))
RETRAIN_MINUTE = int(os.environ.get("MODEL_RETRAIN_MINUTE", 0))
RETRAIN_TZ = os.environ.get("MODEL_RETRAIN_TIMEZONE", "Asia/Kolkata")

_scheduler = None
_scheduler_status = {"status": "stopped", "last_run": None, "last_result": None}


def run_model_retraining_flow() -> bool:
    """
    Executes atomic model retraining pipeline:
    1. Fetch latest historical data
    2. Build 36-feature set
    3. Train updated RandomForest / XGBoost model
    4. Save to temporary model file
    5. Verify model output validity
    6. Atomically promote temporary model -> production model
    7. Reload into in-memory ModelManager singleton
    """
    global _scheduler_status
    logger.info("RETRAIN FLOW STARTED: Initiating daily post-market model retraining...")
    start_t = time.time()
    _scheduler_status["last_run"] = datetime.now(IST).isoformat()

    MODELS_DIR = "models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    temp_model_path = os.path.join(MODELS_DIR, "temp_global_model.pkl")
    prod_model_path = os.path.join(MODELS_DIR, "global_model.pkl")

    try:
        from services.market_data_service import get_market_data_service
        from feature_engineering import build_canonical_feature_set, clean_ohlcv, CANONICAL_FEATURE_COLS
        from sklearn.ensemble import RandomForestRegressor
        from predict import get_model_manager

        market_svc = get_market_data_service()
        raw_df = market_svc.get_historical_data("RELIANCE.NS", period="2y")
        if raw_df.empty:
            csv_path = os.path.join("Stock_Data", "RELIANCE.NS.csv")
            if os.path.exists(csv_path):
                raw_df = pd.read_csv(csv_path)

        df_clean = clean_ohlcv(raw_df, ticker="RELIANCE.NS")
        if df_clean.empty or len(df_clean) < 60:
            raise ValueError("Insufficient data to perform retraining flow.")

        df_feat = build_canonical_feature_set(df_clean, ticker="RELIANCE.NS")
        df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

        X = df_feat[CANONICAL_FEATURE_COLS].values
        # Create multi-day target returns
        y = (df_clean["Close"].shift(-7) - df_clean["Close"]) / df_clean["Close"]
        y = y.iloc[:len(X)].dropna()
        X = X[:len(y)]

        if len(X) < 30:
            raise ValueError("Insufficient feature rows after target alignment.")

        # Train candidate model
        candidate_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        candidate_model.fit(X, y)

        # 4. Save to temporary model file
        joblib.dump(candidate_model, temp_model_path)
        logger.info(f"RETRAIN FLOW: Temporary model saved to {temp_model_path}")

        # 5. Verify candidate model output
        test_pred = candidate_model.predict(X[-1:])
        if test_pred is None or len(test_pred) == 0 or np.isnan(test_pred[0]):
            raise ValueError("Candidate model verification failed: Output contains NaN.")

        # 6. Atomic promotion
        shutil.move(temp_model_path, prod_model_path)
        logger.info(f"RETRAIN FLOW: Atomic promotion successful ({temp_model_path} -> {prod_model_path})")

        # 7. Reload in-memory ModelManager
        model_mgr = get_model_manager()
        model_mgr.load_all_models()
        model_mgr.metadata["last_retrained"] = datetime.now(IST).isoformat()

        elapsed = round(time.time() - start_t, 2)
        _scheduler_status["last_result"] = f"Success ({elapsed}s)"
        logger.info(f"RETRAIN FLOW COMPLETED successfully in {elapsed} seconds.")

        # Dispatch Gmail notification for daily retrain status & portfolio summary
        try:
            from services.notification_service import notification_service
            from services.order_service import get_order_service
            order_svc = get_order_service()
            summary = order_svc.get_account_summary("guest_user")
            
            notification_service.dispatch_alert(
                alert_title="Daily Model Retraining & Post-Market Digest",
                alert_body=f"✅ Model retraining completed successfully in **{elapsed}s**.\n"
                           f"📊 Current Portfolio Value: **₹{summary.get('net_worth', 0.0):,.2f}**\n"
                           f"💰 Cash Balance: **₹{summary.get('cash', 0.0):,.2f}**\n"
                           f"📈 Total PnL: **₹{summary.get('total_pnl', 0.0):,.2f}**",
                channel="BOTH"
            )
        except Exception as notify_err:
            logger.warning("Failed to dispatch daily retrain Gmail alert: %s", notify_err)

        return True

    except Exception as e:
        logger.error(f"RETRAIN FLOW ERROR: Model retraining failed: {e}. Preserving existing production model.")
        if os.path.exists(temp_model_path):
            try:
                os.remove(temp_model_path)
            except Exception:
                pass
        _scheduler_status["last_result"] = f"Failed: {str(e)}"
        return False


def run_background_news_scraping_job():
    """
    Background job: Scrapes live headlines for top 50 tickers every 3 hours on weekdays and once on weekends.
    Enforces random 1-3s delays, timeouts, retries with exponential backoff, and fallback safety.
    Logs run health metrics to scraper_runs table.
    """
    import random
    import json
    from news_logger import log_scraper_run

    started_at = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    logger.info("BACKGROUND NEWS SCRAPER STARTED: Scraper job triggered for top 50 tickers...")

    config_path = os.path.join("config", "top50_tickers.json")
    tickers = []
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
                tickers = data.get("top_50_tickers", [])
        except Exception as e:
            logger.warning(f"Failed to read {config_path}: {e}")

    if not tickers:
        from news_sentiment import STOCK_INFO_MAP
        tickers = list(STOCK_INFO_MAP.keys())[:50]

    from news_sentiment import fetch_multi_source_headlines, get_financial_sentiment_model
    from news_logger import log_raw_headline

    finbert_model = get_financial_sentiment_model()
    scorer_used = "finbert" if (finbert_model._pipeline and not finbert_model._using_fallback) else "lexicon"

    scraped_count = 0
    logged_count = 0
    tickers_ok = 0
    tickers_failed = 0
    source_failures = {}

    for idx, ticker in enumerate(tickers):
        # Random 1-3s delay between requests to avoid rate limits
        delay = random.uniform(1.0, 3.0)
        time.sleep(delay)

        max_retries = 2
        success_for_ticker = False

        for attempt in range(max_retries + 1):
            try:
                headlines = fetch_multi_source_headlines(ticker, max_total=6)
                if headlines:
                    scraped_count += len(headlines)
                    for art in headlines:
                        if not art.get("is_fallback", False):
                            success = log_raw_headline(
                                ticker=ticker,
                                headline=art.get("title", ""),
                                sentiment_score=art.get("score", 0.0),
                                confidence=art.get("confidence", 0.70) / 100.0 if "confidence" in art else 0.70,
                                source=art.get("source", "RSS"),
                                published_date_str=art.get("date", None),
                                scorer_used=scorer_used,
                                finbert_score=art.get("score") if scorer_used == "finbert" else None,
                                lexicon_score=art.get("score") if scorer_used == "lexicon" else None,
                                is_fallback=False
                            )
                            if success:
                                logged_count += 1
                success_for_ticker = True
                break  # Success, break retry loop
            except Exception as ex:
                src_name = "Unknown"
                source_failures[src_name] = source_failures.get(src_name, 0) + 1
                if attempt < max_retries:
                    backoff = (2 ** attempt) * 1.5
                    logger.debug(f"Retry {attempt+1} for {ticker} after error: {ex}. Waiting {backoff}s...")
                    time.sleep(backoff)
                else:
                    logger.warning(f"News scraper failed for {ticker} after {max_retries} retries: {ex}")

        if success_for_ticker:
            tickers_ok += 1
        else:
            tickers_failed += 1

    finished_at = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")
    run_id = log_scraper_run(started_at, finished_at, tickers_ok, tickers_failed, logged_count)

    _scheduler_status["last_news_scrape"] = finished_at
    _scheduler_status["last_news_result"] = f"Run #{run_id}: Scraped {scraped_count} headlines, inserted {logged_count} rows across {tickers_ok} tickers (Failed: {tickers_failed})"
    logger.info(f"BACKGROUND NEWS SCRAPER COMPLETED: Run #{run_id} inserted {logged_count} raw headlines ({tickers_ok} OK, {tickers_failed} failed) at {finished_at}.")


def start_scheduler() -> BackgroundScheduler:
    """Starts APScheduler background service."""
    global _scheduler, _scheduler_status
    if _scheduler is not None and _scheduler.running:
        return _scheduler

    try:
        _scheduler = BackgroundScheduler(timezone=RETRAIN_TZ)
        
        # Job 1: Daily Model Retraining (16:00 IST)
        retrain_trigger = CronTrigger(hour=RETRAIN_HOUR, minute=RETRAIN_MINUTE, day_of_week="mon-fri")
        _scheduler.add_job(run_model_retraining_flow, retrain_trigger, id="daily_model_retrain", replace_existing=True)
        
        # Job 2: Background News Scraping (Every 3 hours on weekdays mon-fri)
        weekday_news_trigger = CronTrigger(hour="0,3,6,9,12,15,18,21", minute=15, day_of_week="mon-fri")
        _scheduler.add_job(run_background_news_scraping_job, weekday_news_trigger, id="weekday_news_scrape", replace_existing=True)

        # Job 3: Weekend News Scraping (Once per day on sat-sun at 10:00 AM)
        weekend_news_trigger = CronTrigger(hour=10, minute=0, day_of_week="sat,sun")
        _scheduler.add_job(run_background_news_scraping_job, weekend_news_trigger, id="weekend_news_scrape", replace_existing=True)

        _scheduler.start()
        _scheduler_status["status"] = "running"
        logger.info(f"Scheduler started. Daily retrain at {RETRAIN_HOUR:02d}:{RETRAIN_MINUTE:02d}, News scraping active (3h weekdays / daily weekends).")
    except Exception as e:
        logger.error(f"Failed to start scheduler: {e}")
        _scheduler_status["status"] = f"error: {str(e)}"

    return _scheduler


def stop_scheduler():
    """Stops background scheduler service."""
    global _scheduler, _scheduler_status
    if _scheduler is not None and _scheduler.running:
        _scheduler.shutdown()
        _scheduler_status["status"] = "stopped"
        logger.info("Scheduler stopped.")


def is_scheduler_running() -> bool:
    """Return True if background scheduler is active."""
    global _scheduler
    return _scheduler is not None and _scheduler.running


def get_scheduler_status() -> dict:
    """Returns current scheduler operational status."""
    global _scheduler, _scheduler_status
    is_running = _scheduler.running if (_scheduler is not None) else False
    return {
        "status": "running" if is_running else _scheduler_status["status"],
        "retrain_hour": RETRAIN_HOUR,
        "retrain_minute": RETRAIN_MINUTE,
        "timezone": RETRAIN_TZ,
        "last_run": _scheduler_status.get("last_run"),
        "last_result": _scheduler_status.get("last_result"),
        "last_news_scrape": _scheduler_status.get("last_news_scrape"),
        "last_news_result": _scheduler_status.get("last_news_result")
    }

