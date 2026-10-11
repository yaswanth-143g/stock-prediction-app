@echo off
REM run_collector.bat — Windows Task Scheduler batch runner for Stock News Collector
REM Scheduled every 3 hours with exclusive file locking protection.

cd /d "C:\Users\yaswa\OneDrive\Desktop\stock price pridiction"
"C:\Program Files\Python313\python.exe" "c:\Users\yaswa\OneDrive\Desktop\stock price pridiction\run_news_collector.py" --once >> collector_log.txt 2>&1
