"""
Dr. Roy App - setup self-check.
Confirms every tool the app needs is installed and working.
Run any time by double-clicking, or it runs automatically if the app fails to start.
"""
import sys

# NOTE: AI engine (torch/ultralytics) is checked FIRST, before PyQt5.
# On Windows, PyQt5 must not load before torch or torch fails to init.
CHECKS = [
    ("torch",           "AI engine (PyTorch)"),
    ("ultralytics",     "AI model training & prediction (YOLO)"),
    ("PyQt5.QtWidgets", "Window & buttons (user interface)"),
    ("reportlab",       "PDF report export"),
    ("PIL",             "Image loading/saving"),
    ("cv2",             "Image analysis (OpenCV)"),
    ("numpy",           "Numerical engine"),
    ("pandas",          "Tables / CSV / Excel export"),
    ("openpyxl",        "Excel (.xlsx) export"),
    ("matplotlib",      "Charts & graphs"),
    ("fitz",            "Live PDF preview (PyMuPDF)"),
    ("googleapiclient", "Google Drive / Colab integration"),
    ("google_auth_oauthlib", "Google sign-in"),
]

print("=" * 56)
print("  Dr. Roy App - Setup Check")
print("  Python:", sys.version.split()[0], "(", sys.executable, ")")
print("=" * 56)

failed = []
for module, friendly in CHECKS:
    try:
        __import__(module)
        print(f"  [OK]  {friendly}")
    except Exception as e:
        print(f"  [!!]  {friendly}  --  {type(e).__name__}: {str(e)[:60]}")
        failed.append(friendly)

print("=" * 56)
if failed:
    print(f"  {len(failed)} tool(s) need attention. Tell Claude and it will fix them.")
else:
    print("  All tools working. Workspace is ready.")
print("=" * 56)
