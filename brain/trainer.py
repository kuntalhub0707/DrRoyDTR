"""
AI base-model download + model training, both on background QThreads so the
window never freezes.

  BaseModelDownloader  — on startup, make sure the base weight is present in
                         models/ (downloads it the first time).
  TrainingWorker       — runs a real ultralytics YOLO training run, emitting one
                         `progress` signal per round (epoch) with live accuracy
                         and loss, then saves the best model + a history entry.

The actual training loop lives in `run_training()` (a plain function), so it can
be unit-tested without Qt.
"""

import os
import time
import json
import shutil

# Load the AI engine (torch + ultralytics) BEFORE PyQt5 so the Windows DLL
# load-order clash never happens, no matter who imports this module first.
import brain.aiboot  # noqa: F401

from PyQt5.QtCore import QThread, pyqtSignal

from brain.paths import APP_ROOT
MODELS_DIR = os.path.join(APP_ROOT, "models")
RUNS_DIR   = os.path.join(APP_ROOT, "output", "training_runs")
HISTORY    = os.path.join(APP_ROOT, "training_history.json")

# Map the friendly size names to ultralytics weight stems
SIZE_STEM = {"Nano": "n", "Small": "s", "Medium": "m", "Large": "l"}


def base_weight_name(model_size, task):
    """e.g. ('Nano','detection') -> 'yolov8n.pt' ; classification -> 'yolov8n-cls.pt'."""
    stem = SIZE_STEM.get(model_size, "n")
    suffix = "-cls" if task == "classification" else ""
    return f"yolov8{stem}{suffix}.pt"


def ensure_base_weight(weight_name, progress=None):
    """
    Make sure `weight_name` exists in models/. Returns its absolute path.
    Downloads via ultralytics the first time (needs internet once).
    `progress` is an optional callable(str) for status messages.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    dest = os.path.join(MODELS_DIR, weight_name)
    if os.path.isfile(dest):
        return dest

    if progress:
        progress("Downloading AI base model...")

    from ultralytics import YOLO
    # Instantiating with just the name makes ultralytics download it (to CWD).
    YOLO(weight_name)
    # Move/копy the downloaded file into models/
    if os.path.isfile(weight_name) and not os.path.isfile(dest):
        try:
            shutil.move(weight_name, dest)
        except Exception:
            shutil.copy(weight_name, dest)
    if os.path.isfile(dest):
        return dest
    # Some ultralytics versions cache elsewhere; fall back to the name itself.
    return weight_name


def _extract_metric(metrics, task):
    """Pull a single 'accuracy-like' number (0..1) from the epoch metrics dict."""
    if not metrics:
        return 0.0
    if task == "classification":
        for k in ("metrics/accuracy_top1", "metrics/accuracy_top5"):
            if k in metrics:
                return float(metrics[k])
        return 0.0
    for k in ("metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/mAP50"):
        if k in metrics:
            return float(metrics[k])
    return 0.0


def _extract_loss(metrics):
    """Sum the validation losses (fall back to training losses) for an error curve."""
    if not metrics:
        return 0.0
    val = [v for k, v in metrics.items() if k.startswith("val/") and "loss" in k]
    if val:
        return float(sum(val))
    tr = [v for k, v in metrics.items() if k.startswith("train/") and "loss" in k]
    return float(sum(tr)) if tr else 0.0


def _append_history(entry):
    data = []
    if os.path.isfile(HISTORY):
        try:
            with open(HISTORY, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    data.append(entry)
    with open(HISTORY, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def run_training(cfg, on_epoch=None, should_stop=None, on_log=None):
    """
    Run one training job. Plain function (no Qt) so it is testable.

    cfg keys: task, model_size, training_rounds, image_size, data_arg,
              auto_save_best, dataset_name, dataset_path
    on_epoch(dict)  -> called once per epoch with progress info
    should_stop()   -> return True to request an early stop
    on_log(str)     -> optional log line

    Returns a result dict (also the saved history entry, plus model_path).
    """
    from ultralytics import YOLO

    task        = cfg.get("task", "detection")
    model_size  = cfg.get("model_size", "Nano")
    epochs      = int(cfg.get("training_rounds", 50))
    imgsz       = int(cfg.get("image_size", 640))
    data_arg    = cfg["data_arg"]
    auto_save   = cfg.get("auto_save_best", True)

    weight = ensure_base_weight(base_weight_name(model_size, task), progress=on_log)
    if on_log:
        on_log(f"Loading base model: {os.path.basename(weight)}")

    model = YOLO(weight)

    start = time.time()
    state = {"best": 0.0, "last_epoch": 0}

    def _cb(trainer):
        # request stop?
        if should_stop and should_stop():
            trainer.stop = True
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        # ultralytics fires this callback once more during final validation
        # (epoch == epochs+1); ignore that extra call so counts stay correct.
        if epoch > epochs:
            return
        metrics = dict(getattr(trainer, "metrics", {}) or {})
        metric = _extract_metric(metrics, task)
        loss = _extract_loss(metrics)
        state["best"] = max(state["best"], metric)
        state["last_epoch"] = epoch
        elapsed = time.time() - start
        per = elapsed / max(epoch, 1)
        remaining = max(epochs - epoch, 0) * per
        if on_epoch:
            on_epoch({
                "epoch": epoch, "total": epochs,
                "metric": metric, "best": state["best"], "loss": loss,
                "elapsed": elapsed, "eta": remaining,
            })

    model.add_callback("on_fit_epoch_end", _cb)

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"train_{task}_{ts}"
    os.makedirs(RUNS_DIR, exist_ok=True)

    model.train(
        data=data_arg, epochs=epochs, imgsz=imgsz,
        project=RUNS_DIR, name=run_name,
        exist_ok=True, verbose=False, plots=False,
        workers=0,   # no dataloader subprocesses — safe inside a GUI thread on Windows
    )

    # Locate the best checkpoint ultralytics saved
    best_src = None
    try:
        best_src = str(model.trainer.best)
    except Exception:
        pass
    if not best_src or not os.path.isfile(best_src):
        cand = os.path.join(RUNS_DIR, run_name, "weights", "best.pt")
        best_src = cand if os.path.isfile(cand) else None

    model_path = ""
    saved_name = ""
    if auto_save and best_src:
        os.makedirs(MODELS_DIR, exist_ok=True)
        saved_name = f"trained_{ 'det' if task=='detection' else 'cls' }_{ts}.pt"
        model_path = os.path.join(MODELS_DIR, saved_name)
        shutil.copy(best_src, model_path)
        # write a metadata sidecar so the Model Library can show classes/params
        try:
            names = cfg.get("names") or list(getattr(model, "names", {}).values())
            nparams = sum(p.numel() for p in model.model.parameters())
            side = os.path.join(MODELS_DIR, os.path.splitext(saved_name)[0] + ".json")
            with open(side, "w", encoding="utf-8") as fh:
                json.dump({
                    "origin": "Fine-tuned",
                    "task": task,
                    "classes": f"{len(names)} (custom)" if names else "—",
                    "names": names,
                    "params": f"{nparams/1e6:.1f}M",
                    "model_size": model_size,
                }, fh, indent=2)
        except Exception:
            pass

    duration = int(time.time() - start)
    stopped = bool(should_stop and should_stop())
    entry = {
        "timestamp": ts,
        "task": task,
        "model": saved_name,
        "model_size": model_size,
        "epochs": state["last_epoch"],
        "requested_epochs": epochs,
        "img_size": imgsz,
        "metric": round(state["best"], 4),
        "dataset": cfg.get("dataset_name", ""),
        "dataset_path": cfg.get("dataset_path", ""),
        "status": "Stopped" if stopped else "Completed",
        "duration_sec": duration,
    }
    _append_history(entry)
    entry = dict(entry)
    entry["model_path"] = model_path
    return entry


# ----------------------------------------------------------------------
# Qt thread wrappers
# ----------------------------------------------------------------------
class BaseModelDownloader(QThread):
    status = pyqtSignal(str)
    ready = pyqtSignal(str)     # path
    failed = pyqtSignal(str)

    def __init__(self, weight_name):
        super().__init__()
        self.weight_name = weight_name

    def run(self):
        try:
            path = ensure_base_weight(self.weight_name, progress=self.status.emit)
            self.ready.emit(path)
        except Exception as e:
            self.failed.emit(str(e))


class TrainingWorker(QThread):
    progress = pyqtSignal(dict)
    log = pyqtSignal(str)
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        try:
            cfg = dict(self.cfg)
            # If the dataset still needs preparing (detect/convert), do it here on
            # the worker thread so the window never freezes.
            if not cfg.get("data_arg"):
                import brain.dataset as dataset
                folder = cfg.get("dataset_folder", "")
                self.log.emit("Checking dataset structure…")
                res = dataset.prepare(folder, cfg.get("task", "detection"))
                if not res.ok:
                    self.failed.emit(res.message)
                    return
                self.log.emit(res.message)
                cfg["data_arg"] = res.data_arg
                cfg["names"] = res.names
                cfg.setdefault("dataset_name", os.path.basename(os.path.normpath(folder)))
                cfg.setdefault("dataset_path", folder)
            result = run_training(
                cfg,
                on_epoch=self.progress.emit,
                should_stop=lambda: self._stop,
                on_log=self.log.emit,
            )
            self.done.emit(result)
        except Exception as e:
            import traceback
            self.failed.emit(f"{e}\n{traceback.format_exc()}")
