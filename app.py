#!/usr/bin/env python3
"""
Launcher for the Urban Change Detection GUI.

    python app.py

The application itself lives in gui/app.py (single source of truth;
v1 had two diverging copies of the GUI).
"""

from gui.app import main


if __name__ == "__main__":
    main()
