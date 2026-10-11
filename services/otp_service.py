"""
services/otp_service.py
========================
In-Memory OTP Store & Registration Verification Engine.
Handles generation, storage, expiration, and validation of 6-digit OTP codes.
"""

import time
import random
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default OTP expiration time: 10 minutes (600 seconds)
OTP_EXPIRATION_SECONDS = 600


class OTPService:
    """
    Singleton service to generate and verify 6-digit OTP codes for account registration.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OTPService, cls).__new__(cls)
            cls._instance.pending_otps: Dict[str, Dict[str, Any]] = {}
        return cls._instance

    def generate_otp(self, email: str, user_data: Dict[str, Any]) -> str:
        """
        Generates a 6-digit numeric OTP for the given email, saves pending registration data,
        and sets expiration timestamp.
        """
        clean_email = email.strip().lower()
        otp_code = f"{random.randint(100000, 999999)}"
        expires_at = time.time() + OTP_EXPIRATION_SECONDS

        self.pending_otps[clean_email] = {
            "otp": otp_code,
            "expires_at": expires_at,
            "user_data": user_data,
            "created_at": time.time()
        }

        logger.info("Generated OTP %s for email %s (Expires in %ds)", otp_code, clean_email, OTP_EXPIRATION_SECONDS)
        return otp_code

    def verify_otp(self, email: str, input_otp: str) -> Optional[Dict[str, Any]]:
        """
        Validates input OTP code against stored record.
        Returns user_data if valid and not expired, else None.
        """
        clean_email = email.strip().lower()
        clean_otp = str(input_otp).strip()

        record = self.pending_otps.get(clean_email)
        if not record:
            logger.warning("OTP verification failed: No pending OTP found for %s", clean_email)
            return None

        if time.time() > record["expires_at"]:
            logger.warning("OTP verification failed: OTP expired for %s", clean_email)
            self.pending_otps.pop(clean_email, None)
            return None

        if record["otp"] != clean_otp:
            logger.warning("OTP verification failed: Incorrect OTP for %s", clean_email)
            return None

        # Verification successful -> Consume OTP record
        user_data = record["user_data"]
        self.pending_otps.pop(clean_email, None)
        logger.info("OTP verification successful for %s", clean_email)
        return user_data

    def get_pending_otp(self, email: str) -> Optional[Dict[str, Any]]:
        """Retrieves pending OTP record if valid."""
        clean_email = email.strip().lower()
        record = self.pending_otps.get(clean_email)
        if record and time.time() <= record["expires_at"]:
            return record
        return None


# Global singleton instance
otp_service = OTPService()
