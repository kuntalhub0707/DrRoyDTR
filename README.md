# Dr. Roy — Data Training & Reporting (DrRoyDTR)

**AI-powered histopathology and hematology training and reporting tool.**
A desktop application that lets a pathologist train their own AI models on slide
images, run predictions on new cases, and turn the results into clinical PDF
reports — entirely offline, with optional free cloud-GPU training via Google Colab.

- **Version:** 1.0.0
- **Author / Publisher:** Dr. Kuntal Roy
- **Platform:** Windows 10/11 (64-bit)

> ⚕️ AI-assisted results are intended to support, not replace, a qualified
> pathologist. All outputs must be reviewed and verified by a professional.

---

## Features

- **Home dashboard** — live counters and live Google Colab session status.
- **Train Model** — fine-tune YOLO detection or classification models on your own
  datasets, with live accuracy/error charts. Auto-detects Roboflow YOLO, plain
  YOLO, COCO, and Pascal VOC formats.
- **Train on Colab** — upload datasets to Google Drive, launch a ready-to-run
  Colab notebook, and have the finished model downloaded back automatically.
- **Training History** — every run, searchable, with charts and one-click re-run.
- **Datasets** — a library that auto-scans format, image counts, splits and classes.
- **Models** — 16 bundled starter models (YOLO26 + YOLOv8, detection + classification)
  plus your own trained/imported models; set a default for prediction.
- **Predict / Analyze** — single-image or whole-folder inference, with detection
  boxes, findings tables, and PDF / CSV / XLSX export.
- **Reports & Export** — clinical PDF reports with a live preview, patient details,
  and your own letterhead.
- **Settings** — report and training defaults, plus timezone.

## Tech stack

PyQt5 (UI) · Ultralytics YOLO + PyTorch (AI) · OpenCV + Pillow (imaging) ·
reportlab + PyMuPDF (PDF) · pandas + openpyxl (CSV/Excel) · matplotlib (charts) ·
Google Drive API (cloud).

---

## Run from source (developers)

```bash
# 1. Create a virtual environment (Python 3.12 recommended)
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
python main.py
```

> On Windows, the app imports the AI engine before PyQt5 (see `brain/aiboot.py`)
> to avoid a known PyTorch/Qt DLL load-order issue.

## Project layout

```
main.py            App entry point + main window / navigation
brain/             AI logic (training, inference, datasets, Drive, reports, ...)
screens/           One module per page of the UI
requirements.txt   Python dependencies
check_setup.py     Self-check that all tools are installed
DrRoyDTR.iss        Inno Setup script (builds the Windows installer)
version_info.txt    Windows exe version metadata
```

## Building the Windows installer

The shipped app bundles a relocatable copy of Python 3.12 plus all libraries
(PyInstaller is **not** used — PyTorch 2.12 deadlocks it). The bundle is then
wrapped with [Inno Setup](https://jrsoftware.org/isinfo.php) via `DrRoyDTR.iss`
to produce a per-user installer and a portable ZIP.

## Google Drive / Colab (optional)

Cloud training needs a one-time, free Google OAuth client file
(`client_secret.json`). The app guides you through creating it the first time you
click **Connect Google Drive**. This file and any tokens are **git-ignored** and
must never be committed.

## License

Copyright © Dr. Kuntal Roy. All rights reserved.
