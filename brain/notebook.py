"""
Google Colab notebook generator.

build_notebook(cfg) returns a valid .ipynb (nbformat 4) as a JSON string, with
all training settings pre-filled, so the user only has to press Runtime > Run All.

cfg keys: run_name, date, task ('detection'|'classification'), model_size,
          epochs, img_size, dataset_folder_name, dataset_folder_id, auto_download
"""

import json

SIZE_STEM = {"Nano": "n", "Small": "s", "Medium": "m", "Large": "l"}


def _weight_name(model_size, task):
    stem = SIZE_STEM.get(model_size, "n")
    suffix = "-cls" if task == "classification" else ""
    return f"yolov8{stem}{suffix}.pt"


def _md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def build_notebook(cfg):
    task = cfg.get("task", "detection")
    weight = _weight_name(cfg.get("model_size", "Nano"), task)
    run_name = cfg["run_name"]
    date = cfg["date"]

    settings = (
        "# === Training settings (filled automatically by Dr. Roy DT&R) ===\n"
        f"RUN_NAME            = {run_name!r}\n"
        f"DATE                = {date!r}\n"
        f"TASK                = {task!r}        # 'detection' or 'classification'\n"
        f"MODEL_WEIGHT        = {weight!r}\n"
        f"EPOCHS              = {int(cfg.get('epochs', 50))}\n"
        f"IMG_SIZE            = {int(cfg.get('img_size', 640))}\n"
        f"DATASET_FOLDER_NAME = {cfg.get('dataset_folder_name', '')!r}\n"
        f"DATASET_FOLDER_ID   = {cfg.get('dataset_folder_id', '')!r}\n"
        f"AUTO_DOWNLOAD       = {bool(cfg.get('auto_download', True))}\n"
        "print('Settings loaded for run:', RUN_NAME)\n"
    )

    sec1 = (
        "# Section 1 — Setup: install the AI tools\n"
        "!pip -q install ultralytics\n"
        "print('✓ Setup complete — ultralytics installed.')\n"
    )

    sec2 = (
        "# Section 2 — Connect to your Google Drive\n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "print('✓ Google Drive connected.')\n"
    )

    sec3 = (
        "# Section 3 — Download the AI base model you selected\n"
        "from ultralytics import YOLO\n"
        "print('Fetching base model:', MODEL_WEIGHT)\n"
        "YOLO(MODEL_WEIGHT)   # downloads + caches the chosen size\n"
        "print('✓ Base model ready.')\n"
    )

    sec4 = (
        "# Section 4 — Prepare the dataset: locate it, auto-detect format, build data.yaml\n"
        "import os, glob, yaml\n"
        "\n"
        "# locate the uploaded folder on Drive (by name)\n"
        "cands = glob.glob(f'/content/drive/MyDrive/**/{DATASET_FOLDER_NAME}', recursive=True)\n"
        "DATASET_DIR = cands[0] if cands else f'/content/drive/MyDrive/{DATASET_FOLDER_NAME}'\n"
        "assert os.path.isdir(DATASET_DIR), f'Dataset folder not found: {DATASET_DIR}'\n"
        "print('Dataset folder:', DATASET_DIR)\n"
        "\n"
        "def find_data_yaml(root):\n"
        "    for c in (os.path.join(root,'data.yaml'), os.path.join(root,'data.yml')):\n"
        "        if os.path.isfile(c): return c\n"
        "    for p in glob.glob(os.path.join(root,'**','*.yaml'), recursive=True):\n"
        "        try:\n"
        "            d = yaml.safe_load(open(p))\n"
        "            if isinstance(d, dict) and 'names' in d: return p\n"
        "        except Exception: pass\n"
        "    return None\n"
        "\n"
        "if TASK == 'classification':\n"
        "    # ultralytics classification takes the dataset directory directly\n"
        "    DATA_ARG = DATASET_DIR\n"
        "    print('Detected: image-classification folder')\n"
        "else:\n"
        "    yp = find_data_yaml(DATASET_DIR)\n"
        "    if yp:\n"
        "        DATA_ARG = yp\n"
        "        print('Detected: YOLO dataset with', os.path.basename(yp))\n"
        "    else:\n"
        "        print('Detected: YOLO images/labels — building data.yaml')\n"
        "        img_dirs = [d for d,_,fs in os.walk(DATASET_DIR)\n"
        "                    if os.path.basename(d).lower()=='images'\n"
        "                    and any(f.lower().endswith(('.jpg','.jpeg','.png')) for f in fs)]\n"
        "        def pick(keys):\n"
        "            for d in img_dirs:\n"
        "                if any(k in d.lower() for k in keys): return d\n"
        "            return img_dirs[0] if img_dirs else DATASET_DIR\n"
        "        train_dir = pick(['train']); val_dir = pick(['valid','val']) or train_dir\n"
        "        mx = -1\n"
        "        for lp in glob.glob(os.path.join(DATASET_DIR,'**','labels','*.txt'), recursive=True):\n"
        "            for line in open(lp):\n"
        "                p = line.split()\n"
        "                if p: mx = max(mx, int(float(p[0])))\n"
        "        names = {i: f'class{i}' for i in range(mx+1)} if mx>=0 else {0:'object'}\n"
        "        DATA_ARG = '/content/data.yaml'\n"
        "        yaml.safe_dump({'path': DATASET_DIR, 'train': train_dir, 'val': val_dir, 'names': names},\n"
        "                       open(DATA_ARG,'w'))\n"
        "    print('Data config:', DATA_ARG)\n"
    )

    sec5 = (
        "# Section 5 — Run training (a progress table updates each round)\n"
        "from ultralytics import YOLO\n"
        "model = YOLO(MODEL_WEIGHT)\n"
        "results = model.train(data=DATA_ARG, epochs=EPOCHS, imgsz=IMG_SIZE,\n"
        "                      project='/content/runs', name=RUN_NAME, exist_ok=True)\n"
        "# capture the best accuracy for the app\n"
        "try:\n"
        "    if TASK == 'classification':\n"
        "        METRIC = float(getattr(results, 'top1', 0.0) or 0.0)\n"
        "    else:\n"
        "        METRIC = float(getattr(getattr(results, 'box', None), 'map', 0.0) or 0.0)\n"
        "except Exception:\n"
        "    METRIC = 0.0\n"
        "print(f'✓ Training finished. Best score: {METRIC:.4f}')\n"
    )

    sec6 = (
        "# Section 6 — Save the best model to Drive\n"
        "import os, shutil, json\n"
        "DEST = f'/content/drive/MyDrive/DrRoyApp_Models/{RUN_NAME}_{DATE}'\n"
        "os.makedirs(DEST, exist_ok=True)\n"
        "best = f'/content/runs/{RUN_NAME}/weights/best.pt'\n"
        "out  = os.path.join(DEST, 'best.pt')\n"
        "shutil.copy(best, out)\n"
        "# write a small result file the app reads for best accuracy\n"
        "with open(os.path.join(DEST, 'DRROY_RESULT.json'), 'w') as fh:\n"
        "    json.dump({'run_name': RUN_NAME, 'date': DATE, 'task': TASK, 'model': 'best.pt',\n"
        "               'metric': round(METRIC, 4), 'epochs': EPOCHS, 'img_size': IMG_SIZE}, fh)\n"
        "print('='*60)\n"
        "print('MODEL SAVED TO:')\n"
        "print(out)\n"
        "print('='*60)\n"
    )

    sec7 = (
        "# Section 7 — Auto-download trigger (the app watches for this)\n"
        "if AUTO_DOWNLOAD:\n"
        "    with open(os.path.join(DEST, 'DRROY_DONE.txt'), 'w') as fh:\n"
        "        fh.write(out)\n"
        "    print('DOWNLOAD_READY:', out)\n"
        "    print('Dr. Roy DT&R will detect this model and download it into your app automatically.')\n"
        "else:\n"
        "    print('Auto-download is off. Your model is on Drive at:')\n"
        "    print(out)\n"
    )

    cells = [
        _md(f"# 🚀 Dr. Roy DT&R — Cloud Training\n"
            f"**Run:** `{run_name}`  ·  **Task:** {task}  ·  **Model:** {weight}  ·  "
            f"**Rounds:** {cfg.get('epochs', 50)}  ·  **Image size:** {cfg.get('img_size', 640)}\n\n"
            f"Just press **Runtime ▸ Run all** — every step below runs automatically."),
        _code(settings),
        _md("## Section 1 — Setup"), _code(sec1),
        _md("## Section 2 — Connect to Drive"), _code(sec2),
        _md("## Section 3 — Download AI base model"), _code(sec3),
        _md("## Section 4 — Prepare dataset"), _code(sec4),
        _md("## Section 5 — Run Training"), _code(sec5),
        _md("## Section 6 — Save the result"), _code(sec6),
        _md("## Section 7 — Auto-download trigger"), _code(sec7),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"name": f"DrRoyApp_Notebook_{run_name}.ipynb", "provenance": []},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1)
