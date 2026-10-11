"""
run_news_collector.py — Standalone Locked News Scraping Collector
===================================================================
Executes multi-source financial news scraping job as a standalone process (no Flask required).
Prevents concurrent execution via file lock protection (`news_collector.lock`).

Windows Task Scheduler Configuration Instructions:
---------------------------------------------------
1. Open Task Scheduler (`taskschd.msc`).
2. Action -> Create Task...
3. General Tab:
   - Name: StockNewsCollectorJob
   - Security options: "Run whether user is logged on or not" or "Run only when user is logged on".
4. Triggers Tab:
   - Click "New..."
   - Trigger 1 (Every 3 Hours):
     - Begin the task: On a schedule (Daily)
     - Repeat task every: 3 hours for a duration of: Indefinitely
   - Trigger 2 (At Startup):
     - Begin the task: At startup
5. Actions Tab:
   - Action: Start a program
   - Program/script: python.exe (or full path C:\\Program Files\\Python313\\python.exe)
   - Add arguments: "c:\\Users\\yaswa\\OneDrive\\Desktop\\stock price pridiction\\run_news_collector.py"
   - Start in: c:\\Users\\yaswa\\OneDrive\\Desktop\\stock price pridiction
6. Conditions / Settings:
   - Uncheck "Stop if the computer switches to battery power".
   - Select "Do not start a new instance" under "If the task is already running, then the following rule applies".
"""

import os
import sys
import time
import atexit

# Ensure project root in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOCK_FILE_PATH = os.path.join(PROJECT_ROOT, "news_collector.lock")
_lock_file_handle = None


def is_pid_running(pid: int) -> bool:
    """Checks if a process ID is actively running on the OS."""
    if pid <= 0:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        if os.name == "nt":
            import ctypes
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            return False
        else:
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False


def clean_stale_lock():
    """Checks if news_collector.lock is stale (>2 hours old or process PID dead) and removes it."""
    if not os.path.exists(LOCK_FILE_PATH):
        return

    try:
        # Check file age (2 hours = 7200 seconds)
        file_age = time.time() - os.path.getmtime(LOCK_FILE_PATH)
        stale_by_age = file_age > 7200

        # Check PID from lock file
        stale_by_pid = False
        pid = None
        with open(LOCK_FILE_PATH, "r") as f:
            content = f.read()
            import re
            m = re.search(r"PID:\s*(\d+)", content)
            if m:
                pid = int(m.group(1))
                if not is_pid_running(pid):
                    stale_by_pid = True

        if stale_by_age or stale_by_pid:
            reason = f"age={file_age:.0f}s (>7200s)" if stale_by_age else f"dead PID={pid}"
            print(f"[STALE LOCK] Cleaning stale news_collector.lock ({reason})...")
            os.remove(LOCK_FILE_PATH)
    except Exception as ex:
        print(f"[WARN] Error inspecting/cleaning lock file: {ex}")


def acquire_process_lock() -> bool:
    """
    Acquires an exclusive file lock on `news_collector.lock`.
    Automatically cleans stale locks (>2 hours old or dead PID).
    Returns True if lock acquired, False if another process is running.
    """
    global _lock_file_handle

    # First attempt stale lock cleanup
    clean_stale_lock()

    try:
        _lock_file_handle = open(LOCK_FILE_PATH, "a+")
        if os.name == "nt":
            import msvcrt
            _lock_file_handle.seek(0)
            msvcrt.locking(_lock_file_handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(_lock_file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)

        # Write current PID to lockfile
        _lock_file_handle.seek(0)
        _lock_file_handle.truncate()
        _lock_file_handle.write(f"PID: {os.getpid()}\nStarted: {time.ctime()}\n")
        _lock_file_handle.flush()

        def release_lock():
            global _lock_file_handle
            if _lock_file_handle:
                try:
                    if os.name == "nt":
                        import msvcrt
                        _lock_file_handle.seek(0)
                        msvcrt.locking(_lock_file_handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(_lock_file_handle, fcntl.LOCK_UN)
                    _lock_file_handle.close()
                except Exception:
                    pass
                try:
                    if os.path.exists(LOCK_FILE_PATH):
                        os.remove(LOCK_FILE_PATH)
                except Exception:
                    pass

        atexit.register(release_lock)
        return True
    except (IOError, OSError):
        return False


import argparse

def main():
    parser = argparse.ArgumentParser(description="Standalone Stock News Collector Job")
    parser.add_argument("--once", action="store_true", help="Run a single scraping iteration and exit cleanly.")
    parser.add_argument("--loop", action="store_true", help="Run continuously in a loop.")
    args = parser.parse_args()

    print("==========================================================")
    print("         STANDALONE STOCK NEWS COLLECTOR JOB             ")
    print("==========================================================")
    print("Windows Task Scheduler Command (Every 3 Hours):")
    print('schtasks /Create /TN "StockNewsCollector" /TR "\\"C:\\Program Files\\Python313\\python.exe\\" \\"c:\\Users\\yaswa\\OneDrive\\Desktop\\stock price pridiction\\run_news_collector.py\\" --once" /SC HOURLY /MO 3 /F')
    print("==========================================================")

    if not acquire_process_lock():
        print(f"[LOCK EXCLUSION] Another instance of run_news_collector.py is currently executing (Lockfile: {LOCK_FILE_PATH}). Exiting cleanly.")
        sys.exit(0)

    print("[LOCK ACQUIRED] Exclusive process lock successfully established.")
    
    from services.scheduler import run_background_news_scraping_job

    start_time = time.time()
    try:
        run_background_news_scraping_job()
        duration = round(time.time() - start_time, 2)
        print(f"\n[JOB SUCCESS] News collection job completed successfully in {duration} seconds.")
    except Exception as ex:
        duration = round(time.time() - start_time, 2)
        print(f"\n[JOB ERROR] Scraping job encountered an unhandled error after {duration} seconds: {ex}")
        sys.exit(1)


if __name__ == "__main__":
    main()
