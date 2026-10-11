"""
db.py — Database Integration Engine for Stock Prediction Platform
===================================================================
Manages MySQL connection pooling with SQLite automatic fallback.
Provides CRUD operations for user authentication and user profiles.
"""

import os
import sqlite3
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("DatabaseEngine")
logger.setLevel(logging.INFO)

USE_MYSQL = False
mysql_conn_pool = None

DB_HOST = os.environ.get("MYSQL_HOST") or os.environ.get("DB_HOST", "localhost")
DB_USER = os.environ.get("MYSQL_USER") or os.environ.get("DB_USER", "root")
DB_PASS = os.environ.get("MYSQL_PASSWORD") or os.environ.get("DB_PASS", "Yaswanth@123")
DB_NAME = os.environ.get("MYSQL_DATABASE") or os.environ.get("DB_NAME", "stock_predictions")
DB_PORT = int(os.environ.get("MYSQL_PORT") or os.environ.get("DB_PORT", 3306))



DEFAULT_SQLITE_PATH = os.path.join("Stock_Data", "app_auth.db")

def get_sqlite_path() -> str:
    """Returns absolute or configured path to SQLite database via STOCK_DB_PATH environment variable."""
    return os.environ.get("STOCK_DB_PATH") or os.environ.get("SQLITE_DB_PATH") or DEFAULT_SQLITE_PATH

def get_sqlite_conn():
    """Connect to local SQLite fallback database, enable WAL mode, and ensure schema exists."""
    db_path = get_sqlite_path()
    db_dir = os.path.dirname(os.path.abspath(db_path))
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=15.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            last_login TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            exchange TEXT DEFAULT 'NSE',
            timestamp TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            ticker TEXT NOT NULL,
            condition_type TEXT NOT NULL,
            target_value REAL NOT NULL,
            channel TEXT DEFAULT 'EMAIL',
            email TEXT,
            notes TEXT,
            is_active INTEGER DEFAULT 1,
            triggered INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            ticker TEXT NOT NULL,
            company_name TEXT,
            sector TEXT,
            shares REAL NOT NULL,
            buy_price REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_paper_account (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE DEFAULT 1,
            virtual_cash REAL DEFAULT 100000.0,
            net_worth REAL DEFAULT 100000.0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_watchlists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER DEFAULT 1,
            ticker TEXT NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_ticker ON market_data (ticker, timestamp);")
    conn.commit()
    return conn


def get_sqlite_connection():
    """Alias for get_sqlite_conn."""
    return get_sqlite_conn()


def check_db_connection() -> tuple:
    """Check database health status and return (is_healthy, details_dict)."""
    if USE_MYSQL and mysql_conn_pool:
        try:
            conn = mysql_conn_pool.get_connection()
            conn.close()
            return True, {"type": "mysql", "database": DB_NAME, "host": DB_HOST}
        except Exception as e:
            return False, {"type": "mysql", "error": str(e)}
    try:
        conn = get_sqlite_conn()
        conn.close()
        return True, {"type": "sqlite", "file": get_sqlite_path()}
    except Exception as e:
        return False, {"type": "sqlite", "error": str(e)}


def get_mysql_connection():
    """Connect to MySQL if available, else SQLite fallback."""
    if USE_MYSQL and mysql_conn_pool:
        try:
            return mysql_conn_pool.get_connection()
        except Exception:
            pass
    return get_sqlite_conn()


def init_db():
    """Initialize database schema for MySQL or SQLite fallback."""
    global USE_MYSQL, mysql_conn_pool
    try:
        import mysql.connector
        from mysql.connector import pooling

        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT,
            autocommit=True
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;")
        cursor.close()
        conn.close()

        mysql_conn_pool = pooling.MySQLConnectionPool(
            pool_name="app_pool",
            pool_size=5,
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            port=DB_PORT
        )
        USE_MYSQL = True
        logger.info(f"MySQL Primary Database connected successfully! (host={DB_HOST}:{DB_PORT}, db={DB_NAME}, pool=app_pool)")
    except Exception as err:
        logger.warning(f"MySQL connection initialization notice: {err}. Falling back to SQLite.")
        USE_MYSQL = False

    if USE_MYSQL and mysql_conn_pool:
        try:
            conn = mysql_conn_pool.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) UNIQUE,
                    email VARCHAR(255) NOT NULL UNIQUE,
                    password_hash VARCHAR(255) NOT NULL,
                    full_name VARCHAR(255),
                    last_login TIMESTAMP NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Auto-migrate schema: ensure full_name column exists if users table was created previously
            try:
                cursor.execute("ALTER TABLE users ADD COLUMN full_name VARCHAR(255);")
            except Exception:
                pass  # Column already exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    ticker VARCHAR(50) NOT NULL,
                    exchange VARCHAR(20) DEFAULT 'NSE',
                    timestamp VARCHAR(100) NOT NULL,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT,
                    source VARCHAR(50),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_ticker_ts (ticker, timestamp)
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_alerts (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT DEFAULT 1,
                    ticker VARCHAR(50) NOT NULL,
                    condition_type VARCHAR(50) NOT NULL,
                    target_value DOUBLE NOT NULL,
                    channel VARCHAR(20) DEFAULT 'BOTH',
                    email VARCHAR(255),
                    telegram_chat_id VARCHAR(100),
                    notes TEXT,
                    is_active TINYINT DEFAULT 1,
                    triggered TINYINT DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_portfolios (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT DEFAULT 1,
                    ticker VARCHAR(50) NOT NULL,
                    company_name VARCHAR(255),
                    sector VARCHAR(100),
                    shares DOUBLE NOT NULL,
                    buy_price DOUBLE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_paper_account (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT UNIQUE DEFAULT 1,
                    virtual_cash DOUBLE DEFAULT 100000.0,
                    net_worth DOUBLE DEFAULT 100000.0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_watchlists (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT DEFAULT 1,
                    ticker VARCHAR(50) NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            cursor.close()
            conn.close()
        except Exception:
            USE_MYSQL = False

    if not USE_MYSQL:
        conn = get_sqlite_conn()
        conn.close()


init_db()


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Retrieve user record by email."""
    if not email:
        return None
    email_clean = email.lower().strip()
    if USE_MYSQL and mysql_conn_pool:
        try:
            conn = mysql_conn_pool.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE email = %s", (email_clean,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            return user
        except Exception as e:
            logger.warning(f"MySQL query error in get_user_by_email, falling back to SQLite: {e}")

    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = ?", (email_clean,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Retrieve user record by username."""
    if not username:
        return None
    usr_clean = username.strip()
    if USE_MYSQL and mysql_conn_pool:
        try:
            conn = mysql_conn_pool.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s", (usr_clean,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            return user
        except Exception as e:
            logger.warning(f"MySQL query error in get_user_by_username, falling back to SQLite: {e}")

    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (usr_clean,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve user record by ID."""
    try:
        u_id = int(user_id)
    except (ValueError, TypeError):
        return None

    if USE_MYSQL and mysql_conn_pool:
        try:
            conn = mysql_conn_pool.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE id = %s", (u_id,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            return user
        except Exception as e:
            logger.warning(f"MySQL query error in get_user_by_id, falling back to SQLite: {e}")

    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (u_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def delete_user_completely(user_id: int = None, email: str = None) -> bool:
    """
    Completely and permanently delete a user account and all associated user data
    (portfolios, alerts, paper trading account, watchlists) from both MySQL primary
    and SQLite fallback databases.
    """
    if not user_id and not email:
        return False

    email_clean = email.lower().strip() if email else None
    deleted_success = False

    # Extract valid numeric user_id if passed as string or int
    parsed_uid = None
    if user_id is not None:
        try:
            parsed_uid = int(user_id)
        except (ValueError, TypeError):
            parsed_uid = None

    if USE_MYSQL and mysql_conn_pool:
        try:
            conn = mysql_conn_pool.get_connection()
            cursor = conn.cursor()

            target_uids = set()
            if parsed_uid is not None:
                target_uids.add(parsed_uid)

            # Always resolve numeric ID from email if provided
            if email_clean:
                cursor.execute("SELECT id FROM users WHERE email = %s", (email_clean,))
                rows = cursor.fetchall()
                for r in rows:
                    if r and r[0] is not None:
                        target_uids.add(int(r[0]))

            for uid in target_uids:
                cursor.execute("DELETE FROM user_portfolios WHERE user_id = %s", (uid,))
                cursor.execute("DELETE FROM user_alerts WHERE user_id = %s", (uid,))
                cursor.execute("DELETE FROM user_paper_account WHERE user_id = %s", (uid,))
                cursor.execute("DELETE FROM user_watchlists WHERE user_id = %s", (uid,))
                cursor.execute("DELETE FROM users WHERE id = %s", (uid,))

            if email_clean:
                cursor.execute("DELETE FROM user_alerts WHERE email = %s", (email_clean,))
                cursor.execute("DELETE FROM users WHERE email = %s", (email_clean,))

            conn.commit()
            cursor.close()
            conn.close()
            deleted_success = True
            logger.info("Successfully deleted user IDs %s / Email %s from MySQL database.", list(target_uids), email_clean)
        except Exception as e:
            logger.error("MySQL delete_user_completely error: %s", e)

    try:
        conn = get_sqlite_conn()
        cursor = conn.cursor()

        target_uids = set()
        if parsed_uid is not None:
            target_uids.add(parsed_uid)

        if email_clean:
            cursor.execute("SELECT id FROM users WHERE email = ?", (email_clean,))
            rows = cursor.fetchall()
            for r in rows:
                val = r["id"] if isinstance(r, sqlite3.Row) else r[0]
                if val is not None:
                    target_uids.add(int(val))

        for uid in target_uids:
            cursor.execute("DELETE FROM user_portfolios WHERE user_id = ?", (uid,))
            cursor.execute("DELETE FROM user_alerts WHERE user_id = ?", (uid,))
            cursor.execute("DELETE FROM user_paper_account WHERE user_id = ?", (uid,))
            cursor.execute("DELETE FROM user_watchlists WHERE user_id = ?", (uid,))
            cursor.execute("DELETE FROM users WHERE id = ?", (uid,))

        if email_clean:
            cursor.execute("DELETE FROM user_alerts WHERE email = ?", (email_clean,))
            cursor.execute("DELETE FROM users WHERE email = ?", (email_clean,))

        conn.commit()
        conn.close()
        deleted_success = True
        logger.info("Successfully deleted user IDs %s / Email %s from SQLite fallback database.", list(target_uids), email_clean)
    except Exception as e:
        logger.error("SQLite delete_user_completely error: %s", e)

    return deleted_success


def delete_user_by_email(email: str):
    """Delete user by email from both MySQL and SQLite databases."""
    return delete_user_completely(email=email)



from security import hash_password, verify_password


def create_user(username: str = "", email: str = "", password_hash: str = "", full_name: str = "") -> Dict[str, Any]:
    """Create new user record."""
    email_clean = email.lower().strip()
    usr_clean = username.strip() or email_clean.split("@")[0]

    if USE_MYSQL and mysql_conn_pool:
        try:
            conn = mysql_conn_pool.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (username, email, password_hash, full_name) VALUES (%s, %s, %s, %s)",
                (usr_clean, email_clean, password_hash, full_name)
            )
            conn.commit()
            new_id = cursor.lastrowid
            cursor.close()
            conn.close()
            return {
                "id": new_id,
                "username": usr_clean,
                "email": email_clean,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        except Exception:
            pass

    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO users (username, email, password_hash, full_name) VALUES (?, ?, ?, ?)",
        (usr_clean, email_clean, password_hash, full_name)
    )
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return {
        "id": new_id,
        "username": usr_clean,
        "email": email_clean,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


def update_last_login(user_id: int):
    """Update last_login timestamp for user."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if USE_MYSQL and mysql_conn_pool:
        try:
            conn = mysql_conn_pool.get_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET last_login = %s WHERE id = %s", (now_str, user_id))
            conn.commit()
            cursor.close()
            conn.close()
            return
        except Exception:
            pass

    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET last_login = ? WHERE id = ?", (now_str, user_id))
    conn.commit()
    conn.close()


def verify_user_credentials(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify user login credentials."""
    user = get_user_by_email(email)
    if user and verify_password(password, user["password_hash"]):
        return user
    return None


def save_market_data_tick(tick_data: Dict[str, Any]):
    """Insert live market data record into MySQL or SQLite database."""
    if not tick_data:
        return

    ticker = str(tick_data.get("ticker", "")).strip().upper()
    exchange = str(tick_data.get("exchange", "NSE")).strip().upper()
    ts = str(tick_data.get("timestamp", datetime.now().isoformat()))
    open_p = float(tick_data.get("open", 0.0))
    high_p = float(tick_data.get("high", 0.0))
    low_p = float(tick_data.get("low", 0.0))
    close_p = float(tick_data.get("ltp", tick_data.get("close", 0.0)))
    vol = int(tick_data.get("volume", 0))
    source = str(tick_data.get("source", "market_api"))

    if USE_MYSQL and mysql_conn_pool:
        try:
            conn = mysql_conn_pool.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO market_data (ticker, exchange, timestamp, open, high, low, close, volume, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (ticker, exchange, ts, open_p, high_p, low_p, close_p, vol, source))
            conn.commit()
            cursor.close()
            conn.close()
            return
        except Exception:
            pass

    try:
        conn = get_sqlite_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO market_data (ticker, exchange, timestamp, open, high, low, close, volume, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticker, exchange, ts, open_p, high_p, low_p, close_p, vol, source))
        conn.commit()
        conn.close()
    except Exception:
        pass


def get_recent_market_ticks(ticker: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent market data ticks for a ticker from database."""
    clean_t = ticker.strip().upper().replace(".NS", "").replace(".BO", "")
    records = []

    if USE_MYSQL and mysql_conn_pool:
        try:
            conn = mysql_conn_pool.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM market_data WHERE ticker = %s ORDER BY id DESC LIMIT %s",
                (clean_t, limit)
            )
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            if rows:
                return rows
        except Exception:
            pass

    try:
        conn = get_sqlite_conn()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM market_data WHERE ticker = ? ORDER BY id DESC LIMIT ?",
            (clean_t, limit)
        )
        rows = cursor.fetchall()
        conn.close()
        records = [dict(r) for r in rows]
    except Exception:
        pass

    return records


# =========================================================================
# USER ALERTS CRUD ENGINE (MySQL & SQLite)
# =========================================================================

def add_user_alert(user_id: int, ticker: str, condition_type: str, target_value: float,
                   channel: str = "BOTH", email: str = "", telegram_chat_id: str = "", notes: str = "") -> Dict[str, Any]:
    """Insert a new price or signal alert for a user."""
    t_clean = str(ticker).strip().upper()
    c_clean = str(condition_type).strip().upper()
    val = float(target_value)
    chan = str(channel).strip().upper()

    if USE_MYSQL and mysql_conn_pool:
        try:
            conn = mysql_conn_pool.get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_alerts (user_id, ticker, condition_type, target_value, channel, email, telegram_chat_id, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (user_id, t_clean, c_clean, val, chan, email, telegram_chat_id, notes))
            conn.commit()
            new_id = cursor.lastrowid
            cursor.close()
            conn.close()
            return {"id": new_id, "user_id": user_id, "ticker": t_clean, "condition_type": c_clean, "target_value": val, "channel": chan, "email": email, "notes": notes, "is_active": 1, "triggered": 0}
        except Exception as e:
            logger.warning(f"MySQL alert insert failed, fallback to SQLite: {e}")

    conn = get_sqlite_conn()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_alerts (user_id, ticker, condition_type, target_value, channel, email, telegram_chat_id, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, t_clean, c_clean, val, chan, email, telegram_chat_id, notes))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    return {"id": new_id, "user_id": user_id, "ticker": t_clean, "condition_type": c_clean, "target_value": val, "channel": chan, "email": email, "notes": notes, "is_active": 1, "triggered": 0}


def get_user_alerts(user_id: int = 1) -> List[Dict[str, Any]]:
    """Retrieve all alerts for a user."""
    if USE_MYSQL and mysql_conn_pool:
        try:
            conn = mysql_conn_pool.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM user_alerts WHERE user_id = %s ORDER BY id DESC", (user_id,))
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            if rows:
                return rows
        except Exception:
            pass

    try:
        conn = get_sqlite_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_alerts WHERE user_id = ? ORDER BY id DESC", (user_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def delete_user_alert(alert_id: int, user_id: int = 1) -> bool:
    """Delete a user alert by ID."""
    if USE_MYSQL and mysql_conn_pool:
        try:
            conn = mysql_conn_pool.get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM user_alerts WHERE id = %s AND user_id = %s", (alert_id, user_id))
            conn.commit()
            affected = cursor.rowcount
            cursor.close()
            conn.close()
            return affected > 0
        except Exception:
            pass

    try:
        conn = get_sqlite_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_alerts WHERE id = ? AND user_id = ?", (alert_id, user_id))
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        return affected > 0
    except Exception:
        return False


