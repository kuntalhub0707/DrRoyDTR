"""
Dataset auto-detection, validation and preparation.

Supports four annotation formats for Object Detection:
  - Roboflow YOLO  (data.yaml + train/valid/test folders of images+labels)
  - plain YOLO     (images/ + labels/ folders, or train/val splits)
  - COCO           (a *.json with images/annotations/categories)
  - Pascal VOC     (per-image .xml annotation files)

COCO and VOC are converted on the fly into a YOLO dataset under
output/_converted_<timestamp>/ so ultralytics can train on them directly.

For Image Classification it accepts an ImageFolder layout:
  - split  : root/train/<class>/*  (+ val|valid|test)
  - flat   : root/<class>/*        (auto-split 80/20 into a working dir)

Public entry point: prepare(root, task) -> Result(...)
"""

import os
import json
import time
import glob
import shutil
import random
import xml.etree.ElementTree as ET

import yaml

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

from brain.paths import APP_ROOT
WORK_DIR = os.path.join(APP_ROOT, "output", "_prepared")


class Result:
    """Outcome of preparing a dataset for training."""

    def __init__(self, ok, fmt=None, data_arg=None, names=None, message="", num_images=0):
        self.ok = ok
        self.format = fmt              # human label of detected format
        self.data_arg = data_arg       # what to pass to model.train(data=...)
        self.names = names or []       # class names
        self.message = message         # plain-English summary / error
        self.num_images = num_images


# ----------------------------------------------------------------------
# Low-level scanners
# ----------------------------------------------------------------------
def _walk(root, exts=None, limit=50000):
    out = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if exts is None or f.lower().endswith(exts):
                out.append(os.path.join(dirpath, f))
                if len(out) >= limit:
                    return out
    return out


def _subdirs(root):
    if not os.path.isdir(root):
        return []
    return [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]


def _find_coco_json(root):
    for p in _walk(root, (".json",)):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                d = json.load(fh)
            if isinstance(d, dict) and "images" in d and "annotations" in d and "categories" in d:
                return p
        except Exception:
            continue
    return None


def _has_voc(root):
    for p in _walk(root, (".xml",))[:60]:
        try:
            r = ET.parse(p).getroot()
            if r.tag == "annotation" and (r.find("object") is not None or r.find("size") is not None):
                return True
        except Exception:
            continue
    return False


def _find_data_yaml(root):
    for c in (os.path.join(root, "data.yaml"), os.path.join(root, "data.yml")):
        if os.path.isfile(c):
            return c
    for p in _walk(root, (".yaml", ".yml")):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                d = yaml.safe_load(fh)
            if isinstance(d, dict) and "names" in d:
                return p
        except Exception:
            continue
    return None


def _label_dirs(root):
    dirs = []
    for dirpath, _dirs, files in os.walk(root):
        if os.path.basename(dirpath).lower() == "labels" and any(f.lower().endswith(".txt") for f in files):
            dirs.append(dirpath)
    return dirs


# ----------------------------------------------------------------------
# Format detection
# ----------------------------------------------------------------------
def detect_format(root):
    """Return one of: roboflow_yolo, yolo, coco, voc, or None."""
    if not os.path.isdir(root):
        return None
    if _has_voc(root):
        return "voc"
    if _find_coco_json(root):
        return "coco"
    yaml_path = _find_data_yaml(root)
    subs = set(d.lower() for d in _subdirs(root))
    if yaml_path:
        if subs & {"train", "valid", "val", "test"}:
            return "roboflow_yolo"
        return "yolo"
    if _label_dirs(root):
        return "yolo"
    return None


FORMAT_LABELS = {
    "roboflow_yolo": "Roboflow YOLO",
    "yolo": "plain YOLO",
    "coco": "COCO",
    "voc": "Pascal VOC",
}


def detect_classification(root):
    """Return 'split', 'flat', or None."""
    subs = _subdirs(root)
    low = set(s.lower() for s in subs)
    if "train" in low:
        return "split"
    if subs:
        good = 0
        for s in subs:
            d = os.path.join(root, s)
            if any(f.lower().endswith(IMG_EXTS) for f in os.listdir(d)):
                good += 1
        if good >= 2:        # need at least two classes
            return "flat"
    return None


# ----------------------------------------------------------------------
# Helpers for building / converting to YOLO
# ----------------------------------------------------------------------
def _fresh_workdir(tag):
    os.makedirs(WORK_DIR, exist_ok=True)
    d = os.path.join(WORK_DIR, f"{tag}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(d, exist_ok=True)
    return d


def _max_class_id(label_dirs):
    mx = -1
    for d in label_dirs:
        for f in glob.glob(os.path.join(d, "*.txt")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.split()
                        if parts:
                            mx = max(mx, int(float(parts[0])))
            except Exception:
                continue
    return mx


def _write_yaml(path, ddict):
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(ddict, fh, sort_keys=False, allow_unicode=True)


def _build_yolo_yaml(root):
    """For a plain YOLO tree that lacks a usable data.yaml, build one."""
    # Find image dirs that have a sibling labels dir
    img_dirs = []
    for dirpath, _dirs, files in os.walk(root):
        base = os.path.basename(dirpath).lower()
        if base == "images" and any(f.lower().endswith(IMG_EXTS) for f in files):
            img_dirs.append(dirpath)
    if not img_dirs:
        # fall back: any dir with images
        for dirpath, _dirs, files in os.walk(root):
            if any(f.lower().endswith(IMG_EXTS) for f in files) and _label_dirs(os.path.dirname(dirpath)):
                img_dirs.append(dirpath)
    if not img_dirs:
        return None, []

    # Prefer a train/ split for train and a val/valid for val
    def pick(keys):
        for d in img_dirs:
            if any(k in d.lower() for k in keys):
                return d
        return None

    train_dir = pick(["train"]) or img_dirs[0]
    val_dir = pick(["valid", "val"]) or train_dir

    n = _max_class_id(_label_dirs(root))
    names = [f"class{i}" for i in range(n + 1)] if n >= 0 else ["class0"]

    out = _fresh_workdir("yolo_yaml")
    yaml_path = os.path.join(out, "data.yaml")
    _write_yaml(yaml_path, {
        "path": os.path.abspath(root),
        "train": os.path.abspath(train_dir),
        "val": os.path.abspath(val_dir),
        "names": {i: n_ for i, n_ in enumerate(names)},
    })
    return yaml_path, names


def _convert_coco(root, coco_json):
    """Convert a COCO json + its images into a YOLO dataset. Returns (yaml, names, n)."""
    with open(coco_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    cats = sorted(data["categories"], key=lambda c: c["id"])
    cat_index = {c["id"]: i for i, c in enumerate(cats)}
    names = [c["name"] for c in cats]
    images = {im["id"]: im for im in data["images"]}

    # group annotations per image
    per_img = {}
    for a in data["annotations"]:
        per_img.setdefault(a["image_id"], []).append(a)

    img_root = os.path.dirname(coco_json)
    out = _fresh_workdir("coco2yolo")
    img_out = os.path.join(out, "images"); os.makedirs(img_out, exist_ok=True)
    lbl_out = os.path.join(out, "labels"); os.makedirs(lbl_out, exist_ok=True)

    n_imgs = 0
    for iid, im in images.items():
        fn = im["file_name"]
        src = os.path.join(img_root, fn)
        if not os.path.isfile(src):
            hit = glob.glob(os.path.join(img_root, "**", os.path.basename(fn)), recursive=True)
            if not hit:
                continue
            src = hit[0]
        w, h = im.get("width", 0), im.get("height", 0)
        base = os.path.splitext(os.path.basename(fn))[0]
        shutil.copy(src, os.path.join(img_out, os.path.basename(fn)))
        lines = []
        for a in per_img.get(iid, []):
            if w <= 0 or h <= 0:
                continue
            x, y, bw, bh = a["bbox"]
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"{cat_index[a['category_id']]} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        with open(os.path.join(lbl_out, base + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        n_imgs += 1

    yaml_path = os.path.join(out, "data.yaml")
    _write_yaml(yaml_path, {
        "path": os.path.abspath(out),
        "train": "images",
        "val": "images",
        "names": {i: n_ for i, n_ in enumerate(names)},
    })
    return yaml_path, names, n_imgs


def _convert_voc(root):
    """Convert Pascal VOC xml annotations into a YOLO dataset. Returns (yaml, names, n)."""
    xmls = _walk(root, (".xml",))
    # first pass: collect class names
    names = []
    parsed = []
    for p in xmls:
        try:
            r = ET.parse(p).getroot()
        except Exception:
            continue
        if r.tag != "annotation":
            continue
        size = r.find("size")
        if size is None:
            continue
        w = float(size.findtext("width", "0"))
        h = float(size.findtext("height", "0"))
        if w <= 0 or h <= 0:
            continue
        fname = r.findtext("filename", "")
        objs = []
        for o in r.findall("object"):
            cls = o.findtext("name", "").strip()
            if not cls:
                continue
            if cls not in names:
                names.append(cls)
            b = o.find("bndbox")
            if b is None:
                continue
            xmin = float(b.findtext("xmin", "0")); ymin = float(b.findtext("ymin", "0"))
            xmax = float(b.findtext("xmax", "0")); ymax = float(b.findtext("ymax", "0"))
            objs.append((cls, xmin, ymin, xmax, ymax))
        parsed.append((p, fname, w, h, objs))

    names.sort()
    idx = {c: i for i, c in enumerate(names)}

    out = _fresh_workdir("voc2yolo")
    img_out = os.path.join(out, "images"); os.makedirs(img_out, exist_ok=True)
    lbl_out = os.path.join(out, "labels"); os.makedirs(lbl_out, exist_ok=True)

    n_imgs = 0
    for xml_path, fname, w, h, objs in parsed:
        # locate the image
        src = None
        if fname:
            cand = glob.glob(os.path.join(root, "**", os.path.basename(fname)), recursive=True)
            if cand:
                src = cand[0]
        if src is None:
            base = os.path.splitext(os.path.basename(xml_path))[0]
            for ext in IMG_EXTS:
                cand = glob.glob(os.path.join(root, "**", base + ext), recursive=True)
                if cand:
                    src = cand[0]; break
        if src is None:
            continue
        base = os.path.splitext(os.path.basename(src))[0]
        shutil.copy(src, os.path.join(img_out, os.path.basename(src)))
        lines = []
        for cls, xmin, ymin, xmax, ymax in objs:
            cx = ((xmin + xmax) / 2) / w
            cy = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h
            lines.append(f"{idx[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(lbl_out, base + ".txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        n_imgs += 1

    yaml_path = os.path.join(out, "data.yaml")
    _write_yaml(yaml_path, {
        "path": os.path.abspath(out),
        "train": "images",
        "val": "images",
        "names": {i: n_ for i, n_ in enumerate(names)} if names else {0: "object"},
    })
    return yaml_path, names, n_imgs


def _split_classification(root, classes):
    """Auto-split a flat ImageFolder into train/ val/ (80/20) in a working dir."""
    out = _fresh_workdir("cls_split")
    n_imgs = 0
    for cls in classes:
        srcd = os.path.join(root, cls)
        imgs = [f for f in os.listdir(srcd) if f.lower().endswith(IMG_EXTS)]
        random.shuffle(imgs)
        cut = max(1, int(len(imgs) * 0.8)) if len(imgs) > 1 else 1
        for split, group in (("train", imgs[:cut]), ("val", imgs[cut:] or imgs[:1])):
            dstd = os.path.join(out, split, cls)
            os.makedirs(dstd, exist_ok=True)
            for f in group:
                shutil.copy(os.path.join(srcd, f), os.path.join(dstd, f))
                n_imgs += 1
    return out, n_imgs


# ----------------------------------------------------------------------
# Lightweight scan (for the Dataset Library — no conversion)
# ----------------------------------------------------------------------
def _split_counts(root):
    counts = {"train": 0, "valid": 0, "test": 0}
    for dirpath, _d, files in os.walk(root):
        n = sum(1 for f in files if f.lower().endswith(IMG_EXTS))
        if not n:
            continue
        parts = [p.lower() for p in dirpath.split(os.sep)]
        if "train" in parts:
            counts["train"] += n
        elif "valid" in parts or "val" in parts:
            counts["valid"] += n
        elif "test" in parts:
            counts["test"] += n
    return counts


def _classes_for(root, fmt, task):
    try:
        if task == "classification":
            base = os.path.join(root, "train")
            src = base if os.path.isdir(base) else root
            return sorted(_subdirs(src))
        if fmt in ("roboflow_yolo", "yolo"):
            yp = _find_data_yaml(root)
            if yp:
                with open(yp, "r", encoding="utf-8") as fh:
                    d = yaml.safe_load(fh) or {}
                names = d.get("names")
                return list(names.values()) if isinstance(names, dict) else (names or [])
        if fmt == "coco":
            cj = _find_coco_json(root)
            if cj:
                with open(cj, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                return [c["name"] for c in sorted(d["categories"], key=lambda c: c["id"])]
        if fmt == "voc":
            names = []
            for p in _walk(root, (".xml",)):
                try:
                    r = ET.parse(p).getroot()
                except Exception:
                    continue
                for o in r.findall("object"):
                    nm = o.findtext("name", "").strip()
                    if nm and nm not in names:
                        names.append(nm)
                if len(names) >= 30:
                    break
            return sorted(names)
    except Exception:
        pass
    return []


_BADGE = {"roboflow_yolo": "YOLO", "yolo": "YOLO", "coco": "COCO",
          "voc": "VOC", "imagefolder": "FOLDER"}


def scan_dataset(root):
    """
    Inspect a folder and return metadata for the Dataset Library, or None if the
    folder is not a recognised dataset. Does NOT convert anything.
    """
    if not root or not os.path.isdir(root):
        return None
    fmt = detect_format(root)
    if fmt:
        task = "detection"
    elif detect_classification(root):
        task, fmt = "classification", "imagefolder"
    else:
        return None
    return {
        "format": FORMAT_LABELS.get(fmt, "ImageFolder"),
        "format_badge": _BADGE.get(fmt, "?"),
        "task": task,
        "total_images": len(_walk(root, IMG_EXTS)),
        "splits": _split_counts(root),
        "classes": _classes_for(root, fmt, task),
    }


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def prepare(root, task):
    """
    Validate + prepare a dataset folder for the given task.
    task: 'detection' or 'classification'.
    """
    if not root or not os.path.isdir(root):
        return Result(False, message="That folder could not be found. Please choose your dataset folder again.")

    if task == "classification":
        layout = detect_classification(root)
        if layout is None:
            return Result(False, message=(
                "This doesn't look like an image-classification dataset. Expected a folder with one "
                "sub-folder per category (each holding that category's images), optionally split into "
                "train/ and val/."))
        if layout == "split":
            classes = _subdirs(os.path.join(root, "train"))
            n = len(_walk(root, IMG_EXTS))
            return Result(True, fmt="ImageFolder (train/val split)", data_arg=os.path.abspath(root),
                          names=sorted(classes), num_images=n,
                          message=f"Image-classification dataset with {len(classes)} categories, {n} images.")
        # flat -> auto-split
        classes = sorted(_subdirs(root))
        workdir, n = _split_classification(root, classes)
        return Result(True, fmt="ImageFolder (auto-split 80/20)", data_arg=os.path.abspath(workdir),
                      names=classes, num_images=n,
                      message=f"Image-classification dataset with {len(classes)} categories; auto-split into train/val ({n} images).")

    # ---- detection ----
    fmt = detect_format(root)
    if fmt is None:
        return Result(False, message=(
            "Could not recognise the dataset format. Expected one of: Roboflow YOLO, plain YOLO, "
            "COCO (a .json with images/annotations/categories), or Pascal VOC (.xml files)."))

    label = FORMAT_LABELS.get(fmt, fmt)
    n_imgs = len(_walk(root, IMG_EXTS))

    try:
        if fmt == "roboflow_yolo":
            ypath = _find_data_yaml(root)
            with open(ypath, "r", encoding="utf-8") as fh:
                d = yaml.safe_load(fh) or {}
            names = d.get("names")
            names = list(names.values()) if isinstance(names, dict) else (names or [])
            return Result(True, fmt=label, data_arg=os.path.abspath(ypath), names=names, num_images=n_imgs,
                          message=f"Detected {label}: {len(names)} classes, {n_imgs} images.")

        if fmt == "yolo":
            ypath = _find_data_yaml(root)
            if ypath:
                with open(ypath, "r", encoding="utf-8") as fh:
                    d = yaml.safe_load(fh) or {}
                names = d.get("names")
                names = list(names.values()) if isinstance(names, dict) else (names or [])
                return Result(True, fmt=label, data_arg=os.path.abspath(ypath), names=names, num_images=n_imgs,
                              message=f"Detected {label}: {len(names)} classes, {n_imgs} images.")
            ypath, names = _build_yolo_yaml(root)
            if not ypath:
                return Result(False, message="Found YOLO labels but no images. Please check the folder.")
            return Result(True, fmt=label, data_arg=os.path.abspath(ypath), names=names, num_images=n_imgs,
                          message=f"Detected {label}: built data.yaml with {len(names)} classes, {n_imgs} images.")

        if fmt == "coco":
            ypath, names, n = _convert_coco(root, _find_coco_json(root))
            return Result(True, fmt=label, data_arg=os.path.abspath(ypath), names=names, num_images=n,
                          message=f"Detected {label}; converted to YOLO ({len(names)} classes, {n} images).")

        if fmt == "voc":
            ypath, names, n = _convert_voc(root)
            return Result(True, fmt=label, data_arg=os.path.abspath(ypath), names=names, num_images=n,
                          message=f"Detected {label}; converted to YOLO ({len(names)} classes, {n} images).")
    except Exception as e:
        return Result(False, fmt=label, message=f"Detected {label} but could not prepare it: {e}")

    return Result(False, message="Unsupported dataset layout.")
