"""
AI bootstrap — IMPORT THIS FIRST, before PyQt5 / the user interface.

Why this file exists:
On Windows, PyQt5 and PyTorch each ship their own copies of low-level
runtime DLLs. Whichever is imported first wins; if PyQt5 loads first,
PyTorch fails with "WinError 1114: a DLL initialization routine failed".

The whole app avoids that by importing the AI engine before any UI code.
At the very top of main.py, before importing PyQt5, just do:

    import brain.aiboot   # loads torch + ultralytics safely first

That single line guarantees the AI engine initialises cleanly every time.
"""

# Load the AI engine BEFORE anything pulls in PyQt5.
import torch          # noqa: F401  (imported for its side effect: load order)
import ultralytics    # noqa: F401

TORCH_VERSION = torch.__version__
ULTRALYTICS_VERSION = ultralytics.__version__
