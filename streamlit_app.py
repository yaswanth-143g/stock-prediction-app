"""
streamlit_app.py — Root launcher shim for frontend/streamlit_app.py
===================================================================
Forwards execution to the centralized frontend Streamlit application.
"""
import os
import sys

# Add project root and frontend directories to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if FRONTEND_DIR not in sys.path:
    sys.path.insert(0, FRONTEND_DIR)

from frontend.streamlit_app import main

if __name__ == "__main__":
    main()
