"""
Inference engine — runs a trained/bundled model on a single image or a whole
folder, on a background QThread so the window stays responsive.

Each prediction saves an annotated PNG and a JSON sidecar into
output/predictions/ (the Reports page reads those sidecars later).

Public:
  run_single(model_file, image_path, conf, iou, task)  -> result dict
  run_batch(model_file, folder, conf, iou, task, on_progress, should_stop) -> dict
  InferenceWorker / BatchInferenceWorker  -> Qt thread wrappers
"""

import os
import json
import time

# Load AI engine before PyQt5 (Windows DLL order)
import brain.aiboot  # noqa: F401

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from brain.paths import APP_ROOT
MODELS_DIR  = os.path.join(APP_ROOT, "models")
PRED_DIR    = os.path.join(APP_ROOT, "output", "predictions")

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _resolve_weight(model_file, on_log=None):
    """Return a usable weight path; download a bundled weight if it's missing."""
    path = os.path.join(MODELS_DIR, model_file)
    if os.path.isfile(path):
        return path
    if on_log:
        on_log(f"Downloading {model_file}…")
    from brain.trainer import ensure_base_weight
    return ensure_base_weight(model_file, progress=on_log)


def _findings_from_result(res, task):
    """Build [{class, count, avg_conf}] from one ultralytics Result."""
    if task == "classification" and getattr(res, "probs", None) is not None:
        top1 = int(res.probs.top1)
        conf1 = float(res.probs.top1conf)
        return [{"class": res.names[top1], "count": 1, "avg_conf": round(conf1, 4)}]
    counts, confs = {}, {}
    boxes = getattr(res, "boxes", None)
    if boxes is not None and boxes.cls is not None:
        for c, cf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            name = res.names[int(c)]
            counts[name] = counts.get(name, 0) + 1
            confs.setdefault(name, []).append(cf)
    return [{"class": n, "count": counts[n], "avg_conf": round(sum(confs[n]) / len(confs[n]), 4)}
            for n in sorted(counts)]


def _predict_one(model, image_path, conf, iou, task, save_annotated=True):
    os.makedirs(PRED_DIR, exist_ok=True)
    t0 = time.time()
    res = model.predict(source=image_path, conf=conf, iou=iou, verbose=False, save=False)[0]
    elapsed = time.time() - t0

    ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time()%1)*1000):03d}"
    out_png = ""
    if save_annotated:
        annotated = res.plot()  # BGR uint8
        out_png = os.path.join(PRED_DIR, f"predict_{ts}.png")
        cv2.imwrite(out_png, annotated)

    findings = _findings_from_result(res, task)
    record = {
        "timestamp": ts,
        "image_name": os.path.basename(image_path),
        "image_path": os.path.abspath(image_path),
        "model": os.path.basename(getattr(model, "ckpt_path", "") or ""),
        "task": task,
        "elapsed": round(elapsed, 3),
        "findings": findings,
        "result_image": out_png,
    }
    # JSON sidecar for the Reports page
    if save_annotated:
        try:
            with open(os.path.join(PRED_DIR, f"result_{ts}.json"), "w", encoding="utf-8") as fh:
                json.dump(record, fh, indent=2)
        except Exception:
            pass
    return record


def run_single(model_file, image_path, conf, iou, task, on_log=None):
    from ultralytics import YOLO
    weight = _resolve_weight(model_file, on_log)
    model = YOLO(weight)
    model.ckpt_path = model_file
    return _predict_one(model, image_path, conf, iou, task)


def list_images(folder):
    out = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(IMG_EXTS):
            out.append(os.path.join(folder, f))
    return out


def run_batch(model_file, folder, conf, iou, task, on_progress=None, should_stop=None, on_log=None):
    from ultralytics import YOLO
    weight = _resolve_weight(model_file, on_log)
    model = YOLO(weight)
    model.ckpt_path = model_file

    images = list_images(folder)
    rows, aggregate = [], {}
    t0 = time.time()
    for i, img in enumerate(images, start=1):
        if should_stop and should_stop():
            break
        rec = _predict_one(model, img, conf, iou, task)
        total = sum(f["count"] for f in rec["findings"])
        breakdown = ", ".join(f"{f['class']}: {f['count']}" for f in rec["findings"]) or "—"
        for f in rec["findings"]:
            aggregate[f["class"]] = aggregate.get(f["class"], 0) + f["count"]
        rows.append({
            "image": rec["image_name"],
            "detections": total,
            "classes": breakdown,
            "elapsed": rec["elapsed"],
            "result_image": rec["result_image"],
        })
        if on_progress:
            on_progress({"i": i, "n": len(images), "image": rec["image_name"], "row": rows[-1]})
    return {
        "rows": rows, "aggregate": aggregate,
        "n": len(rows), "total_images": len(images),
        "elapsed_total": round(time.time() - t0, 2),
        "stopped": bool(should_stop and should_stop()),
    }


# ----------------------------------------------------------------------
class InferenceWorker(QThread):
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, model_file, image_path, conf, iou, task):
        super().__init__()
        self.args = (model_file, image_path, conf, iou, task)

    def run(self):
        try:
            self.done.emit(run_single(*self.args, on_log=self.log.emit))
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()}")


class BatchInferenceWorker(QThread):
    progress = pyqtSignal(dict)
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)
    log = pyqtSignal(str)

    def __init__(self, model_file, folder, conf, iou, task):
        super().__init__()
        self.args = (model_file, folder, conf, iou, task)
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        try:
            result = run_batch(*self.args, on_progress=self.progress.emit,
                               should_stop=lambda: self._stop, on_log=self.log.emit)
            self.done.emit(result)
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()}")
