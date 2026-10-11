"""
services/notification_service.py
================================
Production-Grade Real-Time Notification Engine for Telegram & Email Alerts.
Supports non-blocking async dispatch of stock price alerts, stop-loss breaches,
AI prediction signals, and automated portfolio reports.
"""

import os
import time
import logging
import threading
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import urllib.request
import urllib.parse
import json
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Asynchronous thread pool executor for non-blocking notification delivery
_executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="NotificationWorker")


class NotificationService:
    """
    Singleton notification dispatch engine for Telegram Bot and SMTP Email channels.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(NotificationService, cls).__new__(cls)
                cls._instance._init_config()
            return cls._instance

    def _init_config(self):
        # SMTP Email Configuration
        self.smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.environ.get('SMTP_PORT', 587))
        self.smtp_username = os.environ.get('SMTP_USERNAME', 'yaswanthgangavarapu208@gmail.com')
        self.smtp_password = os.environ.get('SMTP_PASSWORD', 'vxprqtpkovyiftvf')
        self.smtp_sender = os.environ.get('SMTP_SENDER_EMAIL', '23se02ml230@ppsu.ac.in')
        self.smtp_use_tls = os.environ.get('SMTP_USE_TLS', 'true').lower() == 'true'

        logger.info("NotificationService initialized. SMTP Email configured: %s",
                    bool(self.smtp_username and self.smtp_password))

    def send_email_notification(self, subject: str, body_text: str, body_html: str = None, recipient_email: str = None) -> dict:
        """
        Synchronously dispatches email notification via SMTP.
        Includes automatic fallback to direct Gmail SMTP IPs if host DNS resolution (getaddrinfo) times out.
        """
        target_recipient = str(recipient_email).strip() if recipient_email else self.smtp_username
        if not self.smtp_username or not self.smtp_password or not target_recipient:
            logger.warning("SMTP email dispatch skipped: SMTP_USERNAME or SMTP_PASSWORD or recipient missing.")
            return {"status": "skipped", "message": "SMTP credentials or recipient not configured"}

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        from_email = self.smtp_username
        msg["From"] = f"NSE Quantitative Cockpit <{from_email}>"
        msg["To"] = target_recipient
        if self.smtp_sender:
            msg["Reply-To"] = self.smtp_sender

        msg.attach(MIMEText(body_text, "plain"))
        if body_html:
            msg.attach(MIMEText(body_html, "html"))

        # Primary host connection attempt
        hosts_to_try = [self.smtp_server]
        if self.smtp_server.lower() in ("smtp.gmail.com", "gmail.com"):
            # Direct IP fallbacks if local Windows DNS resolution (getaddrinfo) fails or times out
            hosts_to_try.extend(["142.250.27.108", "142.250.153.108", "74.125.200.108", "173.194.77.108"])

        last_error = None
        for host in hosts_to_try:
            try:
                is_ip = host.replace(".", "").isdigit()
                if self.smtp_port == 465:
                    ssl_ctx = None
                    if is_ip:
                        ssl_ctx = ssl.create_default_context()
                        ssl_ctx.check_hostname = False
                        ssl_ctx.verify_mode = ssl.CERT_NONE

                    with smtplib.SMTP_SSL(host, self.smtp_port, timeout=10, context=ssl_ctx) as server:
                        server.login(self.smtp_username, self.smtp_password)
                        server.sendmail(from_email, [target_recipient], msg.as_string())
                else:
                    with smtplib.SMTP(host, self.smtp_port, timeout=10) as server:
                        server.ehlo("smtp.gmail.com")
                        if self.smtp_use_tls:
                            if is_ip:
                                tls_ctx = ssl.create_default_context()
                                tls_ctx.check_hostname = False
                                tls_ctx.verify_mode = ssl.CERT_NONE
                                server.starttls(context=tls_ctx)
                            else:
                                server.starttls()
                            server.ehlo("smtp.gmail.com")
                        server.login(self.smtp_username, self.smtp_password)
                        server.sendmail(from_email, [target_recipient], msg.as_string())

                logger.info("Email alert '%s' sent successfully to %s (via %s)", subject, target_recipient, host)
                return {"status": "success", "recipient": target_recipient, "server": host}
            except Exception as e:
                last_error = e
                logger.warning("SMTP connection to %s failed (%s), attempting fallback...", host, str(e))

        logger.error("Failed to send email notification to %s after trying all hosts: %s", target_recipient, str(last_error), exc_info=True)
        return {"status": "error", "recipient": target_recipient, "message": str(last_error)}

    def send_email_async(self, subject: str, body_text: str, body_html: str = None, recipient_email: str = None):
        """Dispatches email notification asynchronously in a background thread worker with error tracking."""
        def _worker_done_callback(fut):
            try:
                res = fut.result()
                if isinstance(res, dict) and res.get("status") == "error":
                    logger.error("Async email dispatch failed for subject '%s': %s", subject, res.get("message"))
            except Exception as ex:
                logger.error("Unhandled exception in async email dispatch worker: %s", str(ex), exc_info=True)

        future = _executor.submit(self.send_email_notification, subject, body_text, body_html, recipient_email)
        future.add_done_callback(_worker_done_callback)

    def dispatch_alert(self, alert_title: str, alert_body: str, channel: str = "EMAIL", chat_id: str = None, recipient_email: str = None) -> dict:
        """
        Dispatches real-time stock alert via Gmail SMTP Email.
        """
        results = {}
        formatted_email_html = f"""
        <div style="font-family: Arial, sans-serif; background-color: #0F172A; color: #F8FAFC; padding: 24px; border-radius: 12px;">
            <h2 style="color: #38BDF8; margin-top: 0;">🚨 {alert_title}</h2>
            <div style="font-size: 15px; line-height: 1.6; color: #E2E8F0; white-space: pre-wrap;">{alert_body}</div>
            <hr style="border: 0; border-top: 1px solid #334155; margin: 20px 0;" />
            <p style="font-size: 12px; color: #94A3B8;">Enterprise NSE Quantitative Stock Prediction Platform Alert Engine</p>
        </div>
        """

        results["email"] = self.send_email_notification(
            subject=f"NSE Alert: {alert_title}",
            body_text=alert_body,
            body_html=formatted_email_html,
            recipient_email=recipient_email
        )

        return results


    def send_welcome_email(self, username: str, recipient_email: str):
        """Dispatches a rich welcome email notification to a newly registered user account."""
        subject = f"🎉 Welcome to NSE Quantitative Cockpit, {username}!"
        body_text = f"Hello {username},\n\nWelcome to Enterprise NSE Quantitative Trading Terminal! Your account ({recipient_email}) has been successfully created.\n\nYou will receive real-time stock price alerts and AI prediction triggers directly in your inbox."
        
        body_html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; background-color: #0B1120; color: #F8FAFC; padding: 32px; border-radius: 16px; max-width: 600px; margin: 0 auto; border: 1px solid #1E293B;">
            <div style="text-align: center; margin-bottom: 24px;">
                <h1 style="color: #38BDF8; font-size: 24px; margin: 0;">🚀 Welcome to NSE Quant Cockpit</h1>
                <p style="color: #94A3B8; font-size: 14px; margin-top: 6px;">AI-Powered Stock Predictions & Quantitative Terminal</p>
            </div>
            <div style="background-color: #0F172A; border-radius: 12px; padding: 20px; border: 1px solid #334155; margin-bottom: 24px;">
                <h3 style="color: #34D399; margin-top: 0;">Hello {username},</h3>
                <p style="font-size: 14px; line-height: 1.6; color: #CBD5E1;">Your account registered under <strong style="color: #38BDF8;">{recipient_email}</strong> has been successfully created and activated!</p>
                <ul style="font-size: 13.5px; color: #94A3B8; padding-left: 20px; line-height: 1.8;">
                    <li>⚡ Real-time stock price alerts & threshold notifications</li>
                    <li>🔮 AI-driven 7-day price predictions & confidence bands</li>
                    <li>📊 Multi-asset portfolio analytics & Markowitz MVO</li>
                </ul>
            </div>
            <div style="text-align: center; color: #64748B; font-size: 12px; border-top: 1px solid #1E293B; padding-top: 16px;">
                Enterprise NSE Stock Prediction Platform • Powered by Gmail SMTP Engine
            </div>
        </div>
        """

        self.send_email_async(
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            recipient_email=recipient_email
        )

    def send_otp_email(self, username: str, recipient_email: str, otp_code: str):
        """Dispatches a 6-digit OTP verification email for account registration."""
        subject = f"🔐 Your Account Verification Code: {otp_code} — NSE Quant Cockpit"
        body_text = f"Hello {username},\n\nYour 6-digit verification code is: {otp_code}\n\nThis code will expire in 10 minutes.\nIf you did not request this account creation, please ignore this email."

        body_html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; background-color: #0B1120; color: #F8FAFC; padding: 32px; border-radius: 16px; max-width: 550px; margin: 0 auto; border: 1px solid #1E293B;">
            <div style="text-align: center; margin-bottom: 24px;">
                <h1 style="color: #38BDF8; font-size: 22px; margin: 0;">🔐 Email Verification Code</h1>
                <p style="color: #94A3B8; font-size: 13.5px; margin-top: 6px;">NSE Quantitative Cockpit Account Registration</p>
            </div>
            <div style="background-color: #0F172A; border-radius: 12px; padding: 24px; text-align: center; border: 1px solid #334155; margin-bottom: 24px;">
                <p style="font-size: 14px; color: #CBD5E1; margin-bottom: 16px;">Hello <strong style="color: #F8FAFC;">{username}</strong>, use the following code to complete your registration:</p>
                <div style="font-family: monospace; font-size: 36px; font-weight: 800; letter-spacing: 8px; color: #34D399; background: #1E293B; padding: 14px 20px; border-radius: 10px; display: inline-block; border: 1px solid #059669;">
                    {otp_code}
                </div>
                <p style="font-size: 12px; color: #94A3B8; margin-top: 16px;">⏱️ Code expires in <strong>10 minutes</strong>.</p>
            </div>
            <div style="text-align: center; color: #64748B; font-size: 12px; border-top: 1px solid #1E293B; padding-top: 16px;">
                If you did not attempt to register, no further action is required.
            </div>
        </div>
        """

        self.send_email_async(
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            recipient_email=recipient_email
        )

    def send_order_execution_email(self, order_record: dict, recipient_email: str = None):
        """Dispatches a rich Gmail SMTP trade execution receipt for paper trading orders."""
        target_email = recipient_email or order_record.get("email") or order_record.get("recipient_email")
        if not target_email and order_record.get("user_id"):
            try:
                from db import get_user_by_id
                u = get_user_by_id(order_record.get("user_id"))
                if u and u.get("email"):
                    target_email = u["email"]
            except Exception:
                pass

        order_id = order_record.get("order_id", "N/A")
        side = str(order_record.get("side", "BUY")).upper()
        ticker = order_record.get("ticker", "N/A")
        quantity = order_record.get("quantity", 0)
        price = float(order_record.get("price", 0.0))
        total_cost = float(order_record.get("total_cost", price * quantity))
        est_fees = float(order_record.get("estimated_fees", 0.0))
        timestamp = order_record.get("timestamp", "")

        side_color = "#34D399" if side == "BUY" else "#F87171"
        side_badge = "🟢 BOUGHT" if side == "BUY" else "🔴 SOLD"

        subject = f"📈 Trade Receipt: {side} {quantity} {ticker} @ ₹{price:,.2f} — NSE Cockpit"
        body_text = f"Trade Execution Receipt [{order_id}]\nSide: {side}\nTicker: {ticker}\nQuantity: {quantity}\nFill Price: ₹{price:,.2f}\nTotal Cost: ₹{total_cost:,.2f}\nFees: ₹{est_fees:,.2f}\nTimestamp: {timestamp}"

        body_html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; background-color: #0B1120; color: #F8FAFC; padding: 32px; border-radius: 16px; max-width: 580px; margin: 0 auto; border: 1px solid #1E293B;">
            <div style="text-align: center; margin-bottom: 24px;">
                <h1 style="color: #38BDF8; font-size: 22px; margin: 0;">🧾 Trade Execution Receipt</h1>
                <p style="color: #94A3B8; font-size: 13px; margin-top: 4px;">Paper Trading Engine • Gmail SMTP Notification</p>
            </div>
            <div style="background-color: #0F172A; border-radius: 12px; padding: 20px; border: 1px solid #334155; margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1E293B; padding-bottom: 12px; margin-bottom: 16px;">
                    <span style="font-size: 18px; font-weight: 700; color: {side_color};">{side_badge} {ticker}</span>
                    <span style="font-family: monospace; font-size: 13px; color: #94A3B8;">{order_id}</span>
                </div>
                <table style="width: 100%; font-size: 14px; border-collapse: collapse; color: #CBD5E1;">
                    <tr><td style="padding: 6px 0; color: #94A3B8;">Action Side:</td><td style="text-align: right; font-weight: 600; color: {side_color};">{side}</td></tr>
                    <tr><td style="padding: 6px 0; color: #94A3B8;">Quantity:</td><td style="text-align: right; font-weight: 600;">{quantity:,} shares</td></tr>
                    <tr><td style="padding: 6px 0; color: #94A3B8;">Fill Price:</td><td style="text-align: right; font-weight: 600; color: #38BDF8;">₹{price:,.2f}</td></tr>
                    <tr><td style="padding: 6px 0; color: #94A3B8;">Nominal Cost:</td><td style="text-align: right; font-weight: 600;">₹{total_cost:,.2f}</td></tr>
                    <tr><td style="padding: 6px 0; color: #94A3B8;">Estimated Fees:</td><td style="text-align: right; font-weight: 600;">₹{est_fees:,.2f}</td></tr>
                    <tr style="border-top: 1px solid #334155;"><td style="padding: 10px 0 0 0; color: #F8FAFC; font-weight: 700;">Total Settlement:</td><td style="padding: 10px 0 0 0; text-align: right; font-weight: 700; color: #34D399; font-size: 16px;">₹{(total_cost + est_fees if side == "BUY" else total_cost - est_fees):,.2f}</td></tr>
                </table>
            </div>
            <div style="text-align: center; color: #64748B; font-size: 12px; border-top: 1px solid #1E293B; padding-top: 14px;">
                Executed at {timestamp} • Enterprise NSE Quantitative Cockpit
            </div>
        </div>
        """

        self.send_email_async(
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            recipient_email=target_email
        )

    def send_portfolio_digest_email(self, summary_data: dict, recipient_email: str = None):
        """Dispatches a daily portfolio digest and performance summary via Gmail SMTP."""
        target_email = recipient_email or summary_data.get("email") or summary_data.get("recipient_email")
        if not target_email and summary_data.get("user_id"):
            try:
                from db import get_user_by_id
                u = get_user_by_id(summary_data.get("user_id"))
                if u and u.get("email"):
                    target_email = u["email"]
            except Exception:
                pass
        net_worth = float(summary_data.get("net_worth", 0.0))
        cash = float(summary_data.get("cash", 0.0))
        total_pnl = float(summary_data.get("total_pnl", 0.0))
        pnl_pct = float(summary_data.get("total_pnl_pct", 0.0))
        pos_count = summary_data.get("active_positions_count", 0)

        pnl_color = "#34D399" if total_pnl >= 0 else "#F87171"
        pnl_sign = "+" if total_pnl >= 0 else ""

        subject = f"📊 Daily Portfolio Digest: Net Worth ₹{net_worth:,.2f} ({pnl_sign}{pnl_pct:.2f}%) — NSE Cockpit"
        body_text = f"Daily Portfolio Digest\nNet Worth: ₹{net_worth:,.2f}\nCash Balance: ₹{cash:,.2f}\nTotal PnL: ₹{total_pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)\nActive Positions: {pos_count}"

        body_html = f"""
        <div style="font-family: 'Segoe UI', Arial, sans-serif; background-color: #0B1120; color: #F8FAFC; padding: 32px; border-radius: 16px; max-width: 600px; margin: 0 auto; border: 1px solid #1E293B;">
            <div style="text-align: center; margin-bottom: 24px;">
                <h1 style="color: #38BDF8; font-size: 22px; margin: 0;">📊 Daily Portfolio Summary Digest</h1>
                <p style="color: #94A3B8; font-size: 13px; margin-top: 4px;">Enterprise NSE Quantitative Cockpit Report</p>
            </div>
            <div style="background-color: #0F172A; border-radius: 12px; padding: 20px; border: 1px solid #334155; margin-bottom: 20px;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
                    <div style="background: #1E293B; padding: 14px; border-radius: 8px;">
                        <div style="font-size: 12px; color: #94A3B8;">Total Portfolio Valuation</div>
                        <div style="font-size: 20px; font-weight: 700; color: #F8FAFC; margin-top: 4px;">₹{net_worth:,.2f}</div>
                    </div>
                    <div style="background: #1E293B; padding: 14px; border-radius: 8px;">
                        <div style="font-size: 12px; color: #94A3B8;">Unrealized PnL</div>
                        <div style="font-size: 20px; font-weight: 700; color: {pnl_color}; margin-top: 4px;">{pnl_sign}₹{total_pnl:,.2f} ({pnl_sign}{pnl_pct:.2f}%)</div>
                    </div>
                </div>
                <div style="font-size: 13.5px; color: #CBD5E1; line-height: 1.6;">
                    • <strong>Available Cash:</strong> ₹{cash:,.2f}<br/>
                    • <strong>Active Holdings:</strong> {pos_count} tickers currently open
                </div>
            </div>
            <div style="text-align: center; color: #64748B; font-size: 12px; border-top: 1px solid #1E293B; padding-top: 14px;">
                Dispatched via Gmail SMTP Engine • Enterprise NSE Quantitative Cockpit
            </div>
        </div>
        """

        self.send_email_async(
            subject=subject,
            body_text=body_text,
            body_html=body_html,
            recipient_email=recipient_email
        )

    def evaluate_and_trigger_alerts(self, active_alerts: list, ticker_prices: dict, predictions: dict = None, sentiments: dict = None, cooldown_seconds: int = 300) -> list:
        """
        Evaluates a list of alert rules against live market data, AI predictions, and sentiment scores.
        Applies a minimum cooldown period (default 300s / 5 mins) between repeat triggers for the same rule.
        Returns list of triggered alerts.
        """
        if not hasattr(self, "_alert_cooldowns"):
            self._alert_cooldowns = {}

        if not hasattr(self, "_alert_history"):
            self._alert_history = []

        now_ts = time.time()
        triggered = []

        for alert in active_alerts:
            rule_id = str(alert.get("id") or alert.get("rule_id") or alert.get("ticker", "GENERIC"))
            last_trig = self._alert_cooldowns.get(rule_id, 0.0)

            # 1. Enforce Cooldown
            if now_ts - last_trig < cooldown_seconds:
                continue

            ticker = str(alert.get("ticker", "")).strip().upper()
            curr_price = ticker_prices.get(ticker) if ticker_prices else None
            cond = str(alert.get("condition_type") or alert.get("type", "ABOVE")).upper()
            target_val = float(alert.get("target_value", 0.0))

            is_triggered = False
            trigger_reason = ""

            # Price Threshold Rules
            if curr_price is not None and curr_price > 0:
                if cond in ("ABOVE", "PRICE_ABOVE", "TAKE_PROFIT") and curr_price >= target_val:
                    is_triggered = True
                    trigger_reason = f"Live Price ₹{curr_price:,.2f} crossed Target ABOVE ₹{target_val:,.2f}"
                elif cond in ("BELOW", "PRICE_BELOW", "STOP_LOSS") and curr_price <= target_val:
                    is_triggered = True
                    trigger_reason = f"Live Price ₹{curr_price:,.2f} crossed Target BELOW ₹{target_val:,.2f}"
                elif cond == "PCT_CHANGE":
                    ref_price = float(alert.get("reference_price", curr_price))
                    pct = ((curr_price - ref_price) / ref_price) * 100.0 if ref_price > 0 else 0.0
                    if abs(pct) >= target_val:
                        is_triggered = True
                        trigger_reason = f"Price Change {pct:+.2f}% reached threshold ±{target_val:.2f}%"

            # AI Prediction Rules (PRED_ABOVE / PRED_BELOW)
            if not is_triggered and predictions and ticker in predictions:
                pred_data = predictions[ticker]
                pred_price = float(pred_data.get("predicted_price", 0.0))
                confidence = float(pred_data.get("confidence", 0.0))
                min_conf = float(alert.get("min_confidence", 70.0))

                if confidence >= min_conf and pred_price > 0:
                    if cond == "PRED_ABOVE" and pred_price >= target_val:
                        is_triggered = True
                        trigger_reason = f"AI Predicted Price ₹{pred_price:,.2f} crossed target ₹{target_val:,.2f} (Confidence: {confidence:.0f}%)"
                    elif cond == "PRED_BELOW" and pred_price <= target_val:
                        is_triggered = True
                        trigger_reason = f"AI Predicted Price ₹{pred_price:,.2f} dropped below target ₹{target_val:,.2f} (Confidence: {confidence:.0f}%)"

            # News Sentiment Rules (SENTIMENT_ABOVE / SENTIMENT_BELOW / SENTIMENT_CHANGE)
            if not is_triggered and sentiments and ticker in sentiments:
                sent_score = float(sentiments[ticker].get("score", 0.0))
                if cond == "SENTIMENT_ABOVE" and sent_score >= target_val:
                    is_triggered = True
                    trigger_reason = f"FinBERT Sentiment Score {sent_score:+.2f} crossed threshold ABOVE {target_val:+.2f}"
                elif cond == "SENTIMENT_BELOW" and sent_score <= target_val:
                    is_triggered = True
                    trigger_reason = f"FinBERT Sentiment Score {sent_score:+.2f} dropped BELOW threshold {target_val:+.2f}"

            if is_triggered:
                # Update Cooldown Timestamp
                self._alert_cooldowns[rule_id] = now_ts

                alert["status"] = "TRIGGERED"
                alert["triggered_price"] = curr_price if curr_price else target_val
                alert["triggered_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                alert["trigger_reason"] = trigger_reason

                # Record in Alert Execution History
                history_entry = dict(alert)
                self._alert_history.append(history_entry)

                title = f"🔔 Stock Alert Triggered: {ticker}"
                body = f"Stock **{ticker}** triggered rule **{cond}**.\n\nReason: **{trigger_reason}**"
                channel = alert.get("channel", "BOTH")
                alert_recipient = alert.get("email") or alert.get("recipient_email") or alert.get("user_email")
                self.dispatch_alert(title, body, channel=channel, recipient_email=alert_recipient)
                triggered.append(alert)

        return triggered

    def get_alert_history(self, user_id: int = None) -> list:
        """Retrieves executed alert trigger history."""
        if not hasattr(self, "_alert_history"):
            return []
        if user_id is not None:
            return [a for a in self._alert_history if a.get("user_id") == user_id]
        return list(self._alert_history)


# Global Singleton Instance
notification_service = NotificationService()


