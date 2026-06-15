"""
Single source of truth for the app's writable data folder.

When running normally (from source) APP_ROOT is the project folder. When frozen
into a PyInstaller .exe, it is the folder next to the executable — so models,
datasets, reports, settings, etc. live beside the program (and stay writable),
not inside the read-only bundle.
"""

import os
import sys


def _app_root():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


APP_ROOT = _app_root()
