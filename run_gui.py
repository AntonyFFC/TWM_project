"""Launcher script for the GUI application.

Usage:
    python run_gui.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from gui.main_window import main  # noqa: E402


if __name__ == "__main__":
    main()
