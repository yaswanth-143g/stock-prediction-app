"""
tests/conftest.py — Global Pytest Fixtures for Test Isolation
==============================================================
Enforces SQLite database isolation so all unit tests execute against a
temporary, isolated SQLite database (tmp_path) and never touch the production DB.
"""

import os
import sqlite3
import pytest

import db
import news_logger


@pytest.fixture(autouse=True)
def isolate_test_database(tmp_path, monkeypatch):
    """
    Autouse fixture that redirects all database connection calls in tests
    to an isolated, temporary SQLite database.
    """
    test_db_path = str(tmp_path / "isolated_test_app_auth.db")

    def test_get_sqlite_conn():
        conn = sqlite3.connect(test_db_path, timeout=15.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.row_factory = sqlite3.Row
        return conn

    monkeypatch.setattr(db, "get_sqlite_conn", test_get_sqlite_conn)
    monkeypatch.setattr(db, "get_sqlite_connection", test_get_sqlite_conn)
    monkeypatch.setattr(db, "get_mysql_connection", test_get_sqlite_conn)
    monkeypatch.setattr(news_logger, "get_sqlite_conn", test_get_sqlite_conn)

    # Initialize schema in the temporary test database
    conn = test_get_sqlite_conn()
    news_logger.init_news_tables(conn=conn)
    conn.close()

    yield test_db_path
