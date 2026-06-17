"""
Explainable AI (XAI) — heatmaps that show WHERE a model looked.

Uses EigenCAM: the first principal component of a deep convolutional feature
map, projected back over the image. It needs no gradients and no target class,
so it works the same for detection AND classification, and for every bundled or
trained model. Self-contained (torch + OpenCV only — no extra dependency).

explain_image(model_file, image_path) -> path to a saved heatmap-overlay PNG
"""

import os
import time

# Load the AI engine before anything Qt-related might (Windows DLL order)
import brain.aiboot  # noqa: F401

import numpy as np
import cv2
import torch

from brain.paths import APP_ROOT

PRED_DIR = os.path.join(APP_ROOT, "output", "predictions")


def _letterbox(im, new=640, color=114):
    """Resize keeping aspect ratio, pad to a square (like YOLO preprocessing)."""
    h, w = im.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new, new, 3), color, dtype=np.uint8)
    top, left = (new - nh) // 2, (new - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def _eigen_cam(act):
    """act: torch tensor (C, H, W) -> normalised (H, W) saliency in [0,1]."""
    c, h, w = act.shape
    a = act.reshape(c, h * w).T              # (HW, C)
    a = a - a.mean(0, keepdim=True)
    try:
        _u, _s, v = torch.linalg.svd(a, full_matrices=False)
        cam = (a @ v[0]).reshape(h, w)       # projection onto 1st principal component
    except Exception:
        cam = a.norm(dim=1).reshape(h, w)    # fallback: activation magnitude
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy()


def explain_image(model_file, image_path, out_path=None, imgsz=640, alpha=0.45):
    from ultralytics import YOLO
    from brain.inference import _resolve_weight

    weight = _resolve_weight(model_file)
    net = YOLO(weight).model.float().eval()

    # capture every 4D conv feature map during one forward pass
    acts = []
    handles = []

    def _hook(_m, _i, out):
        if isinstance(out, torch.Tensor) and out.ndim == 4:
            acts.append(out.detach())

    for mod in net.modules():
        if isinstance(mod, torch.nn.Conv2d):
            handles.append(mod.register_forward_hook(_hook))

    img = cv2.imread(image_path)             # BGR
    if img is None:
        for h in handles:
            h.remove()
        raise RuntimeError("Could not read that image.")
    canvas = _letterbox(img, imgsz)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        net(tensor)
    for h in handles:
        h.remove()

    if not acts:
        raise RuntimeError("Could not read the model's feature maps for explanation.")

    # choose a deep, semantically-rich feature map: spatial size near imgsz/32,
    # and among those, the one with the most channels (the backbone output).
    target = imgsz / 32.0

    def _score(a):
        _, ch, hh, _ = a.shape
        return abs(hh - target) - ch * 0.001

    best = min(acts, key=_score)
    cam = _eigen_cam(best[0])
    cam = cv2.resize(cam, (imgsz, imgsz), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(canvas, 1 - alpha, heat, alpha, 0)

    os.makedirs(PRED_DIR, exist_ok=True)
    if out_path is None:
        ts = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"
        out_path = os.path.join(PRED_DIR, f"explain_{ts}.png")
    cv2.imwrite(out_path, overlay)
    return out_path
