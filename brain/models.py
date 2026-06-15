"""
Model library logic.

Lists the 8 pre-bundled starter models (YOLOv8 nano/small/medium/large, for
both detection and classification) plus any model the user has trained or
imported into models/. Handles the default-model setting, importing, deleting,
and locating bundled weights.

A model is "downloaded" when its .pt file is physically in models/. Bundled
models that haven't been downloaded yet are still listed (with known specs) and
can be fetched on demand.
"""

import os
import json
import shutil

from brain.paths import APP_ROOT
MODELS_DIR = os.path.join(APP_ROOT, "models")
SETTINGS   = os.path.join(APP_ROOT, "model_settings.json")

# Pre-bundled starter models. YOLO26 (newest generation) listed first, then
# YOLOv8. Params built from each architecture config; sizes are the real
# download sizes. Both families ship with the installed ultralytics 8.4.66.
BUNDLED = [
    # --- YOLO26 detection (COCO) ---
    {"file": "yolo26n.pt",     "task": "detection",      "size_name": "Nano",   "params": "2.6M",  "size_mb": 5.3,  "classes": "80 (COCO)"},
    {"file": "yolo26s.pt",     "task": "detection",      "size_name": "Small",  "params": "10.0M", "size_mb": 19.5, "classes": "80 (COCO)"},
    {"file": "yolo26m.pt",     "task": "detection",      "size_name": "Medium", "params": "21.9M", "size_mb": 42.2, "classes": "80 (COCO)"},
    {"file": "yolo26l.pt",     "task": "detection",      "size_name": "Large",  "params": "26.3M", "size_mb": 50.7, "classes": "80 (COCO)"},
    # --- YOLOv8 detection (COCO) ---
    {"file": "yolov8n.pt",     "task": "detection",      "size_name": "Nano",   "params": "3.2M",  "size_mb": 6.2,  "classes": "80 (COCO)"},
    {"file": "yolov8s.pt",     "task": "detection",      "size_name": "Small",  "params": "11.2M", "size_mb": 21.5, "classes": "80 (COCO)"},
    {"file": "yolov8m.pt",     "task": "detection",      "size_name": "Medium", "params": "25.9M", "size_mb": 49.7, "classes": "80 (COCO)"},
    {"file": "yolov8l.pt",     "task": "detection",      "size_name": "Large",  "params": "43.7M", "size_mb": 83.7, "classes": "80 (COCO)"},
    # --- YOLO26 classification (ImageNet) ---
    {"file": "yolo26n-cls.pt", "task": "classification", "size_name": "Nano",   "params": "2.8M",  "size_mb": 5.5,  "classes": "1000 (ImageNet)"},
    {"file": "yolo26s-cls.pt", "task": "classification", "size_name": "Small",  "params": "6.7M",  "size_mb": 13.0, "classes": "1000 (ImageNet)"},
    {"file": "yolo26m-cls.pt", "task": "classification", "size_name": "Medium", "params": "11.6M", "size_mb": 22.4, "classes": "1000 (ImageNet)"},
    {"file": "yolo26l-cls.pt", "task": "classification", "size_name": "Large",  "params": "14.1M", "size_mb": 27.2, "classes": "1000 (ImageNet)"},
    # --- YOLOv8 classification (ImageNet) ---
    {"file": "yolov8n-cls.pt", "task": "classification", "size_name": "Nano",   "params": "2.7M",  "size_mb": 5.3,  "classes": "1000 (ImageNet)"},
    {"file": "yolov8s-cls.pt", "task": "classification", "size_name": "Small",  "params": "6.4M",  "size_mb": 12.4, "classes": "1000 (ImageNet)"},
    {"file": "yolov8m-cls.pt", "task": "classification", "size_name": "Medium", "params": "17.0M", "size_mb": 32.9, "classes": "1000 (ImageNet)"},
    {"file": "yolov8l-cls.pt", "task": "classification", "size_name": "Large",  "params": "37.5M", "size_mb": 71.8, "classes": "1000 (ImageNet)"},
]
BUNDLED_FILES = {b["file"] for b in BUNDLED}

DEFAULT_DETECTION = "yolov8n.pt"
DEFAULT_CLASSIFICATION = "yolov8n-cls.pt"


def _fmt_size(mb):
    if mb >= 1:
        return f"{mb:.1f} MB"
    return f"{mb*1024:.0f} KB"


def get_defaults():
    data = {}
    if os.path.isfile(SETTINGS):
        try:
            with open(SETTINGS, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
        except Exception:
            data = {}
    return {
        "detection": data.get("default_detection", DEFAULT_DETECTION),
        "classification": data.get("default_classification", DEFAULT_CLASSIFICATION),
    }


def set_default(file, task):
    data = {}
    if os.path.isfile(SETTINGS):
        try:
            with open(SETTINGS, "r", encoding="utf-8") as fh:
                data = json.load(fh) or {}
        except Exception:
            data = {}
    key = "default_detection" if task == "detection" else "default_classification"
    data[key] = file
    with open(SETTINGS, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _sidecar_path(model_file):
    return os.path.join(MODELS_DIR, os.path.splitext(model_file)[0] + ".json")


def _read_sidecar(model_file):
    p = _sidecar_path(model_file)
    if os.path.isfile(p):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}
    return {}


def _user_models():
    """Scan models/ for .pt files that are not bundled starters."""
    out = []
    if not os.path.isdir(MODELS_DIR):
        return out
    for f in sorted(os.listdir(MODELS_DIR)):
        if not f.lower().endswith(".pt") or f in BUNDLED_FILES:
            continue
        side = _read_sidecar(f)
        # infer task: sidecar -> filename hint -> default detection
        task = side.get("task")
        if not task:
            task = "classification" if ("-cls" in f or "_cls" in f) else "detection"
        origin = side.get("origin") or ("Fine-tuned" if f.startswith("trained_") else "Imported")
        path = os.path.join(MODELS_DIR, f)
        out.append({
            "file": f, "task": task, "origin": origin,
            "params": side.get("params", "—"),
            "classes": side.get("classes", "—"),
            "size_mb": os.path.getsize(path) / (1024 * 1024),
            "downloaded": True,
            "bundled": False,
        })
    return out


def list_models():
    """Return (detection_list, classification_list) of model dicts."""
    defaults = get_defaults()
    items = []

    for b in BUNDLED:
        path = os.path.join(MODELS_DIR, b["file"])
        downloaded = os.path.isfile(path)
        size_mb = os.path.getsize(path) / (1024 * 1024) if downloaded else b["size_mb"]
        items.append({
            "file": b["file"], "task": b["task"], "origin": "Bundled",
            "size_name": b["size_name"], "params": b["params"], "classes": b["classes"],
            "size_mb": size_mb, "downloaded": downloaded, "bundled": True,
        })

    items.extend(_user_models())

    for m in items:
        m["is_default"] = (defaults[m["task"]] == m["file"])
        m["path"] = os.path.join(MODELS_DIR, m["file"])
        m["size_str"] = _fmt_size(m["size_mb"])

    det = [m for m in items if m["task"] == "detection"]
    cls = [m for m in items if m["task"] == "classification"]
    return det, cls


def count_models():
    det, cls = list_models()
    return len(det) + len(cls)


def import_pt(src):
    """Copy an external .pt into models/. Returns the destination filename."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    name = os.path.basename(src)
    dest = os.path.join(MODELS_DIR, name)
    # avoid clobbering
    if os.path.exists(dest) and os.path.abspath(src) != os.path.abspath(dest):
        stem, ext = os.path.splitext(name)
        i = 1
        while os.path.exists(os.path.join(MODELS_DIR, f"{stem}_{i}{ext}")):
            i += 1
        name = f"{stem}_{i}{ext}"
        dest = os.path.join(MODELS_DIR, name)
    if os.path.abspath(src) != os.path.abspath(dest):
        shutil.copy(src, dest)
    task = "classification" if ("-cls" in name or "_cls" in name) else "detection"
    with open(_sidecar_path(name), "w", encoding="utf-8") as fh:
        json.dump({"origin": "Imported", "task": task}, fh, indent=2)
    return name


def delete_model(file):
    """Remove a model's local .pt (and sidecar). Bundled stays available to re-download."""
    path = os.path.join(MODELS_DIR, file)
    if os.path.isfile(path):
        try:
            os.remove(path)
        except Exception:
            return False
    side = _sidecar_path(file)
    if os.path.isfile(side):
        try:
            os.remove(side)
        except Exception:
            pass
    return True
