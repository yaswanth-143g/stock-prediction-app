import os
import sys

from app import app, socketio

if __name__ == "__main__":
    print("Starting Stock Prediction Server on http://127.0.0.1:5000 ...", flush=True)
    socketio.run(app, host="127.0.0.1", port=5000, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
