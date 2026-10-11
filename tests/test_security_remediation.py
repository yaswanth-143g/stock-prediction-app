"""
test_security_remediation.py — Automated Security & Quality Verification Test Suite
===================================================================================
Verifies salted password hashing, JWT security, email validation, XSS sanitization,
and IST timezone accuracy.
"""

import sys
import unittest
import os
import hashlib
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from security import (
    hash_password,
    verify_password,
    generate_jwt_token,
    verify_jwt_token,
    sanitize_input,
    is_valid_email
)
from predict import get_nse_market_status
from app import app


class TestSecurityRemediation(unittest.TestCase):

    def setUp(self):
        self.app_client = app.test_client()

    def test_01_password_hashing(self):
        """Verify Werkzeug salted password hashing & verification."""
        password = "SecurePassword123!"
        hashed = hash_password(password)

        self.assertTrue(hashed.startswith("pbkdf2:") or hashed.startswith("scrypt:") or hashed.startswith("argon2:"), f"Expected salted hash format, got {hashed[:20]}")
        self.assertTrue(verify_password(password, hashed))
        self.assertFalse(verify_password("WrongPassword!", hashed))

    def test_02_legacy_sha256_compatibility(self):
        """Verify fallback compatibility for legacy SHA-256 password hashes."""
        password = "LegacyPassword456"
        legacy_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()

        self.assertTrue(verify_password(password, legacy_hash))
        self.assertFalse(verify_password("WrongPassword", legacy_hash))

    def test_03_email_validation(self):
        """Verify email format validation rules."""
        valid_emails = ["user@example.com", "john.doe@company.co.in", "test+tag@domain.org"]
        invalid_emails = ["invalid-email", "user@.com", "@domain.com", "plainaddress", "user@domain..com"]

        for email in valid_emails:
            self.assertTrue(is_valid_email(email), f"Expected valid: {email}")

        for email in invalid_emails:
            self.assertFalse(is_valid_email(email), f"Expected invalid: {email}")

    def test_04_xss_input_sanitization(self):
        """Verify HTML/XSS input sanitization."""
        malicious_input = "<script>alert('XSS')</script> & 'quoted'"
        sanitized = sanitize_input(malicious_input)

        self.assertNotIn("<script>", sanitized)
        self.assertIn("&lt;script&gt;", sanitized)
        self.assertIn("&amp;", sanitized)

    def test_05_jwt_token_security(self):
        """Verify JWT token encoding, decoding, and tamper resistance."""
        user_id = 42
        email = "testuser@example.com"
        token = generate_jwt_token(user_id, email)

        self.assertIsNotNone(token)
        payload = verify_jwt_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload.get("email"), email)

        # Test tampered token rejection
        tampered_token = token[:-4] + "XXXX"
        self.assertIsNone(verify_jwt_token(tampered_token))

    def test_06_ist_market_status(self):
        """Verify Indian Standard Time (IST, UTC+5:30) market status calculation."""
        is_open, status_str = get_nse_market_status()
        self.assertIn(status_str, ["OPEN 🟢", "CLOSED 🔴"])

        IST = timezone(timedelta(hours=5, minutes=30))
        now_ist = datetime.now(IST)
        expected_open = (now_ist.weekday() < 5) and (
            datetime.strptime("09:15:00", "%H:%M:%S").time() <= now_ist.time() <= datetime.strptime("15:30:00", "%H:%M:%S").time()
        )
        self.assertEqual(is_open, expected_open)

    def test_07_api_register_invalid_email(self):
        """Verify registration API rejects invalid email format."""
        resp = self.app_client.post("/api/auth/register", json={
            "username": "testuser_invalid",
            "email": "not-an-email",
            "password": "Password123"
        })
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertIn("Invalid email address format", data.get("message"))


if __name__ == "__main__":
    unittest.main()
