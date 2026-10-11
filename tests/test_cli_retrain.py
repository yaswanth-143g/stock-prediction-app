"""
test_cli_retrain.py — Unit Tests for retrain.py CLI
=====================================================
Verifies that retrain.py CLI help and execution methods work as intended.
"""

import subprocess
import sys


def test_retrain_cli_help():
    """Verifies python retrain.py --help runs cleanly with exit code 0."""
    res = subprocess.run([sys.executable, "retrain.py", "--help"], capture_output=True, text=True)
    assert res.returncode == 0
    assert "Scheduled Retrain CLI" in res.stdout
