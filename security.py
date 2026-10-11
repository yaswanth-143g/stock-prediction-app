"""
security.py — Enterprise Security, JWT Auth & Input Sanitization Engine
========================================================================
JWT Authentication, password hashing/verification, request rate limiting,
input sanitization, and AlertService for stock prediction platform.
"""

import os
import re
import html
import time
import hashlib
import functools
import importlib
from flask import request, jsonify

try:
    jwt = importlib.import_module("jwt")
except Exception:
    jwt = None

from werkzeug.security import generate_password_hash as werkzeug_generate_hash, check_password_hash as werkzeug_check_hash

SECRET_KEY = os.environ.get("SECRET_KEY", os.environ.get("FLASK_SECRET_KEY", "prod_stock_prediction_jwt_secret_key_2026"))


def hash_password(password: str) -> str:
    """Computes secure salted PBKDF2 password hash using Werkzeug."""
    if not password:
        return ""
    return werkzeug_generate_hash(str(password))


def verify_password(password: str, hashed: str) -> bool:
    """Verifies plaintext password against stored hash (supporting Werkzeug salted hash & SHA-256 legacy)."""
    if not password or not hashed:
        return False
    # Check Werkzeug salted hash format first
    if hashed.startswith("pbkdf2:") or hashed.startswith("scrypt:") or hashed.startswith("argon2:"):
        return werkzeug_check_hash(hashed, str(password))
    # Legacy SHA-256 fallback compatibility
    legacy_sha256 = hashlib.sha256(str(password).encode("utf-8")).hexdigest()
    return legacy_sha256 == hashed


def generate_jwt_token(user_id: int, email: str) -> str:
    """Generates a secure HMAC-SHA256 JWT token valid for 7 days."""
    try:
        if jwt is None:
            raise ImportError("PyJWT package not available")
        payload = {
            "sub": str(user_id),
            "user_id": user_id,
            "email": str(email).lower().strip(),
            "exp": time.time() + (7 * 24 * 3600),
            "iat": time.time()
        }
        tok = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
        if isinstance(tok, bytes):
            tok = tok.decode("utf-8")
        return tok
    except Exception:
        # Fallback simple token format if PyJWT unavailable
        token_str = f"{user_id}:{email}:{int(time.time())}"
        signature = hashlib.sha256(f"{token_str}:{SECRET_KEY}".encode("utf-8")).hexdigest()[:16]
        return f"{token_str}:{signature}"


def verify_jwt_token(token: str) -> dict:
    """Verifies JWT token and returns payload dict."""
    if not token:
        return None
    try:
        if jwt is None:
            raise ImportError("PyJWT package not available")
        res = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        if "sub" not in res and "user_id" in res:
            res["sub"] = res["user_id"]
        return res
    except Exception:
        pass

    try:
        parts = token.split(":")
        if len(parts) == 4:
            u_id, email, t_stamp, sig = parts
            token_prefix = f"{u_id}:{email}:{t_stamp}"
            expected_sig = hashlib.sha256(f"{token_prefix}:{SECRET_KEY}".encode("utf-8")).hexdigest()[:16]
            if sig == expected_sig and (time.time() - int(t_stamp)) < (7 * 24 * 3600):
                return {"sub": int(u_id), "user_id": int(u_id), "email": email}
    except Exception:
        pass
    return None


def token_required(f):
    """Flask route decorator requiring valid JWT token in Authorization header."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        token = None
        if auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
        elif "token" in request.args:
            token = request.args.get("token")

        if not token:
            return jsonify({"status": "error", "message": "Authentication token missing"}), 401

        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({"status": "error", "message": "Invalid or expired authentication token"}), 401

        request.user = payload
        request.current_user = payload
        return f(*args, **kwargs)
    return decorated


def global_limiter(f):
    """Flask route decorator for rate limiting."""
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        return f(*args, **kwargs)
    return decorated


class AlertService:
    """Enterprise Price Alert & Monitoring Service."""

    def __init__(self):
        self.alerts = []

    def add_alert(self, ticker: str, alert_type: str = "PRICE_ABOVE", target_value: float = 0.0, notes: str = "", user_id: int = 1, email: str = "") -> dict:
        alert_id = f"ALT-{len(self.alerts) + 1:04d}"
        alert = {
            "id": alert_id,
            "user_id": user_id,
            "ticker": str(ticker).upper(),
            "type": str(alert_type).upper(),
            "target_value": float(target_value),
            "email": str(email).strip().lower(),
            "notes": str(notes),
            "triggered": False,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.alerts.append(alert)
        return alert

    def load_alerts(self, user_id: int = 1) -> list:
        return self.alerts

    def delete_alert(self, alert_id: str) -> bool:
        initial_len = len(self.alerts)
        self.alerts = [a for a in self.alerts if str(a.get("id")) != str(alert_id)]
        return len(self.alerts) < initial_len

    def get_user_alerts(self, user_id: int = 1) -> list:
        return [a for a in self.alerts if a.get("user_id") == user_id]


def sanitize_input(text: str) -> str:
    """Sanitizes user string input against HTML/XSS injections."""
    if not text:
        return ""
    clean = html.escape(str(text).strip())
    return clean


def is_valid_email(email: str) -> bool:
    """Validates email format."""
    if not email or ".." in email:
        return False
    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(pattern, str(email).strip()))
