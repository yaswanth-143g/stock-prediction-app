"""
test_auth_mysql.py — Automated Test Script for MySQL Auth Flow
==============================================================
"""

import os
import sys
import unittest
import json

# Ensure parent path is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
from db import get_user_by_email, get_user_by_username

class TestMySQLAuth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_email = "testuser_suite@quant.com"
        cls.test_username = "testuser_suite"
        cls.test_password = "Password@123"
        try:
            from db import USE_MYSQL, get_mysql_connection, get_sqlite_connection
            if USE_MYSQL:
                try:
                    conn = get_mysql_connection()
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM users WHERE email = %s OR username = %s", (cls.test_email, cls.test_username))
                    cursor.close()
                    conn.close()
                except Exception:
                    pass

            conn = get_sqlite_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE email = ? OR username = ?", (cls.test_email, cls.test_username))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def setUp(self):
        self.client = app.test_client()
        self.test_email = self.__class__.test_email
        self.test_username = self.__class__.test_username
        self.test_password = self.__class__.test_password

    def test_01_registration(self):
        try:
            from db import delete_user_by_email
            delete_user_by_email(self.test_email)
        except Exception:
            pass

        payload = {
            "username": self.test_username,
            "email": self.test_email,
            "password": self.test_password
        }
        response = self.client.post('/api/auth/register', data=json.dumps(payload), content_type='application/json')
        data = json.loads(response.data)
        if response.status_code != 200:
            print("REGISTRATION RESPONSE ERROR:", data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data.get("status"), "otp_required")
        self.assertIn("otp_demo", data)

        # Step 2: Verify OTP
        otp_code = data.get("otp_demo")
        verify_resp = self.client.post('/api/auth/verify-otp', data=json.dumps({
            "email": self.test_email,
            "otp": otp_code
        }), content_type='application/json')
        verify_data = json.loads(verify_resp.data)
        self.assertEqual(verify_resp.status_code, 200)
        self.assertEqual(verify_data.get("status"), "success")
        self.assertIn("token", verify_data)

    def test_02_login(self):
        # Register user if not existing
        reg_resp = self.client.post('/api/auth/register', data=json.dumps({
            "username": self.test_username,
            "email": self.test_email,
            "password": self.test_password
        }), content_type='application/json')
        reg_data = json.loads(reg_resp.data)
        if reg_data.get("status") == "otp_required":
            self.client.post('/api/auth/verify-otp', data=json.dumps({
                "email": self.test_email,
                "otp": reg_data.get("otp_demo")
            }), content_type='application/json')

        payload = {
            "email": self.test_email,
            "password": self.test_password
        }
        response = self.client.post('/api/auth/login', data=json.dumps(payload), content_type='application/json')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data.get("status"), "success")
        self.assertIn("token", data)

        token = data.get("token")
        headers = {"Authorization": f"Bearer {token}"}
        me_resp = self.client.get('/api/auth/me', headers=headers)
        me_data = json.loads(me_resp.data)
        self.assertEqual(me_resp.status_code, 200)
        self.assertEqual(me_data.get("user", {}).get("email"), self.test_email)

if __name__ == "__main__":
    unittest.main()
