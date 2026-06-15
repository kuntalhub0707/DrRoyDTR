"""
Predict / Analyze page.

Single Image or Folder Batch inference using any model in the library (the
default model is pre-selected). Detection shows boxes + a Class/Count/Avg-conf
table; results can be saved as an image or exported to PDF (single) or CSV/XLSX
(folder). All inference runs on a background thread.
"""

import os
import time

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QComboBox,
    QSlider, QScrollArea, QFileDialog, QMessageBox, QSizePolicy, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QProgressBar,
)

import brain.models as models

# --- palette ---
C_BG       = "#0d1117"
C_PANEL    = "#161b27"
C_PANEL2   = "#1c2333"
C_BORDER   = "#2a3045"
C_TEXT     = "#e6edf3"
C_TEXT_DIM = "#8b949e"
C_TEAL     = "#238f7a"
C_TEALH    = "#2aad93"
C_BLUE     = "#3b82f6"
C_RED      = "#e06c75"

from brain.paths import APP_ROOT
REPORTS    = os.path.join(APP_ROOT, "reports")
ACCEPT_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")
MAX_MB = 50

_FIELD_QSS = f"""
    QComboBox {{ background: {C_BG}; color: {C_TEXT}; border: 1px solid {C_BORDER};
        border-radius: 7px; padding: 8px 10px; font-size: 13px; }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox QAbstractItemView {{ background: {C_PANEL2}; color: {C_TEXT};
        selection-background-color: {C_TEAL}; border: 1px solid {C_BORDER}; outline: none; }}
"""
_SLIDER_QSS = f"""
    QSlider::groove:horizontal {{ height: 6px; border-radius: 3px; background: {C_BORDER}; }}
    QSlider::sub-page:horizontal {{ height: 6px; border-radius: 3px; background: {C_TEAL}; }}
    QSlider::handle:horizontal {{ width: 16px; height: 16px; margin: -6px 0; border-radius: 8px;
        background: {C_TEALH}; border: 2px solid {C_BG}; }}
"""


class DropZone(QFrame):
    """Dashed drop target that also acts as a click-to-browse button."""

    activated = pyqtSignal()      # clicked
    dropped = pyqtSignal(str)     # a path was dropped

    def __init__(self, mode_is_folder_getter):
        super().__init__()
        self._is_folder = mode_is_folder_getter
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(260)
        self._restyle(False)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        self.icon = QLabel("🖼")
        self.icon.setAlignment(Qt.AlignCenter)
        self.icon.setStyleSheet("font-size: 44px; background: transparent;")
        self.title = QLabel("Drop a slide patch, or click to browse")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet(f"color: {C_TEXT}; font-size: 16px; font-weight: 700; background: transparent;")
        self.hint = QLabel("Accepted: JPEG, PNG, TIFF, BMP, WebP · up to 50 MB")
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; background: transparent;")
        lay.addWidget(self.icon); lay.addWidget(self.title); lay.addWidget(self.hint)

    def set_folder_mode(self, folder):
        if folder:
            self.icon.setText("🗂")
            self.title.setText("Drop a folder, or click to choose a folder")
            self.hint.setText("Every image inside will be analysed")
        else:
            self.icon.setText("🖼")
            self.title.setText("Drop a slide patch, or click to browse")
            self.hint.setText("Accepted: JPEG, PNG, TIFF, BMP, WebP · up to 50 MB")

    def _restyle(self, hot):
        col = C_TEAL if hot else C_BORDER
        self.setStyleSheet(
            f"DropZone {{ background: {C_PANEL2}; border: 2px dashed {col}; border-radius: 14px; }}")

    def mousePressEvent(self, e):
        self.activated.emit()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction(); self._restyle(True)

    def dragLeaveEvent(self, e):
        self._restyle(False)

    def dropEvent(self, e):
        self._restyle(False)
        urls = e.mimeData().urls()
        if urls:
            self.dropped.emit(urls[0].toLocalFile())


class PredictPage(QWidget):
    def __init__(self, status_callback=None):
        super().__init__()
        self.status_callback = status_callback
        self.folder_mode = False
        self.input_path = ""
        self.worker = None
        self.last_result = None
        self._model_items = []
        self.setStyleSheet(f"background: {C_BG};")

        root = QVBoxLayout(self)
        root.setContentsMargins(36, 24, 36, 24)
        root.setSpacing(14)

        # ---- title + toggle ----
        head = QHBoxLayout()
        title = QLabel("Predict / Analyze")
        title.setStyleSheet(f"color: {C_TEXT}; font-size: 26px; font-weight: 800;")
        head.addWidget(title, 1)
        self.seg = self._build_toggle()
        head.addWidget(self.seg, 0, Qt.AlignVCenter)
        root.addLayout(head)

        # ---- body: left settings / right work area ----
        body = QHBoxLayout(); body.setSpacing(18)
        body.addWidget(self._build_left_panel(), 0)
        body.addWidget(self._build_right_panel(), 1)
        root.addLayout(body, 1)

        self._reload_models()

    # ------------------------------------------------------------------
    def _build_toggle(self):
        wrap = QFrame()
        wrap.setStyleSheet(f"QFrame {{ background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 9px; }}")
        lay = QHBoxLayout(wrap); lay.setContentsMargins(4, 4, 4, 4); lay.setSpacing(4)
        self.btn_single = QPushButton("Single Image")
        self.btn_batch = QPushButton("Folder Batch")
        for b in (self.btn_single, self.btn_batch):
            b.setCursor(Qt.PointingHandCursor); b.setCheckable(True)
            b.setMinimumWidth(130); b.setMinimumHeight(32)
        self.btn_single.setChecked(True)
        self.btn_single.clicked.connect(lambda: self._set_mode(False))
        self.btn_batch.clicked.connect(lambda: self._set_mode(True))
        lay.addWidget(self.btn_single); lay.addWidget(self.btn_batch)
        self._style_toggle()
        return wrap

    def _style_toggle(self):
        for b, on in ((self.btn_single, not self.folder_mode), (self.btn_batch, self.folder_mode)):
            if on:
                b.setStyleSheet(f"QPushButton {{ background: {C_TEAL}; color: #fff; border: none;"
                                f"border-radius: 7px; font-size: 13px; font-weight: 700; }}")
            else:
                b.setStyleSheet(f"QPushButton {{ background: transparent; color: {C_TEXT_DIM}; border: none;"
                                f"border-radius: 7px; font-size: 13px; font-weight: 600; }}"
                                f"QPushButton:hover {{ color: {C_TEXT}; }}")

    def _build_left_panel(self):
        panel = QFrame()
        panel.setFixedWidth(330)
        panel.setStyleSheet(f"QFrame {{ background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 12px; }}")
        lay = QVBoxLayout(panel); lay.setContentsMargins(18, 18, 18, 18); lay.setSpacing(10)

        hd = QLabel("Model")
        hd.setStyleSheet(f"color: {C_TEXT}; font-size: 18px; font-weight: 800; background: transparent;")
        lay.addWidget(hd)
        sub = QLabel("Detection: boxes filtered by confidence. IoU controls overlap removal.")
        sub.setWordWrap(True)
        sub.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; background: transparent;")
        lay.addWidget(sub)

        self.cmb_model = QComboBox(); self.cmb_model.setStyleSheet(_FIELD_QSS)
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        lay.addWidget(self.cmb_model)

        # confidence
        self.conf_val = QLabel("0.25")
        lay.addWidget(self._slider_block("Confidence", 10, 95, 25,
                                         "Boxes below this score are hidden", "conf"))
        # iou
        lay.addWidget(self._slider_block("IoU", 10, 90, 45,
                                         "Controls removal of overlapping boxes", "iou"))

        lay.addSpacing(6)
        run_row = QHBoxLayout(); run_row.setSpacing(8)
        self.btn_run = QPushButton("▶  Run Inference")
        self.btn_run.setMinimumHeight(44); self.btn_run.setCursor(Qt.PointingHandCursor)
        self.btn_run.clicked.connect(self._run)
        self.btn_cancel = QPushButton("✕")
        self.btn_cancel.setFixedSize(44, 44); self.btn_cancel.setCursor(Qt.PointingHandCursor)
        self.btn_cancel.setToolTip("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        self.btn_cancel.setStyleSheet(
            f"QPushButton {{ background: {C_BG}; color: {C_TEXT_DIM}; border: 1px solid {C_BORDER};"
            f"border-radius: 8px; font-size: 16px; }} QPushButton:hover:enabled {{ border: 1px solid {C_RED}; color: {C_RED}; }}"
            f"QPushButton:disabled {{ color: {C_BORDER}; }}")
        run_row.addWidget(self.btn_run, 1); run_row.addWidget(self.btn_cancel, 0)
        lay.addLayout(run_row)
        self._style_run(True)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; background: transparent;")
        lay.addWidget(self.lbl_status)

        lay.addStretch(1)
        return panel

    def _slider_block(self, name, lo, hi, val, hint, kind):
        box = QFrame(); box.setStyleSheet("background: transparent;")
        v = QVBoxLayout(box); v.setContentsMargins(0, 6, 0, 0); v.setSpacing(3)
        row = QHBoxLayout()
        nm = QLabel(name); nm.setStyleSheet(f"color: {C_TEXT}; font-size: 13px; font-weight: 700; background: transparent;")
        val_lbl = QLabel(f"{val/100:.2f}")
        val_lbl.setStyleSheet(f"color: {C_TEALH}; font-size: 13px; font-weight: 700; background: transparent;")
        row.addWidget(nm, 0); row.addStretch(1); row.addWidget(val_lbl, 0)
        v.addLayout(row)
        s = QSlider(Qt.Horizontal); s.setRange(lo, hi); s.setValue(val); s.setStyleSheet(_SLIDER_QSS)
        s.valueChanged.connect(lambda x, l=val_lbl: l.setText(f"{x/100:.2f}"))
        v.addWidget(s)
        h = QLabel(hint); h.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px; background: transparent;")
        v.addWidget(h)
        if kind == "conf":
            self.slider_conf = s
        else:
            self.slider_iou = s
        return box

    def _build_right_panel(self):
        panel = QFrame(); panel.setStyleSheet("background: transparent;")
        self.right = QVBoxLayout(panel); self.right.setContentsMargins(0, 0, 0, 0); self.right.setSpacing(12)

        self.drop = DropZone(lambda: self.folder_mode)
        self.drop.activated.connect(self._browse)
        self.drop.dropped.connect(self._on_dropped)
        self.right.addWidget(self.drop, 0)

        self.selected_lbl = QLabel("")
        self.selected_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")
        self.selected_lbl.setWordWrap(True)
        self.right.addWidget(self.selected_lbl, 0)

        # progress (batch)
        self.bar = QProgressBar(); self.bar.setVisible(False); self.bar.setFixedHeight(18)
        self.bar.setStyleSheet(
            f"QProgressBar {{ background: {C_BG}; border: 1px solid {C_BORDER}; border-radius: 6px;"
            f"text-align: center; color: {C_TEXT}; font-size: 11px; }}"
            f"QProgressBar::chunk {{ background: {C_TEAL}; border-radius: 5px; }}")
        self.right.addWidget(self.bar, 0)

        # results area (scroll)
        self.result_scroll = QScrollArea(); self.result_scroll.setWidgetResizable(True)
        self.result_scroll.setFrameShape(QFrame.NoFrame)
        self.result_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.result_host = QWidget(); self.result_host.setStyleSheet("background: transparent;")
        self.result_lay = QVBoxLayout(self.result_host); self.result_lay.setContentsMargins(0, 0, 0, 0)
        self.result_lay.setSpacing(12)
        self.result_scroll.setWidget(self.result_host)
        self.result_scroll.setVisible(False)
        self.right.addWidget(self.result_scroll, 1)
        return panel

    # ------------------------------------------------------------------
    def showEvent(self, e):
        self._reload_models()
        super().showEvent(e)

    def _reload_models(self):
        det, cls = models.list_models()
        defaults = models.get_defaults()
        cur = self.cmb_model.currentText() if self.cmb_model.count() else ""
        self._model_items = det + cls
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()
        default_index = 0
        for i, m in enumerate(self._model_items):
            tw = "Detection" if m["task"] == "detection" else "Classification"
            mark = "  ★ default" if m["is_default"] else ""
            self.cmb_model.addItem(f"{m['file']} — {tw}{mark}")
            if m["file"] == defaults["detection"] and m["task"] == "detection":
                default_index = i
        # restore previous selection if any
        if cur:
            idx = self.cmb_model.findText(cur)
            self.cmb_model.setCurrentIndex(idx if idx >= 0 else default_index)
        else:
            self.cmb_model.setCurrentIndex(default_index)
        self.cmb_model.blockSignals(False)

    def _current_model(self):
        i = self.cmb_model.currentIndex()
        if 0 <= i < len(self._model_items):
            return self._model_items[i]
        return None

    def _on_model_changed(self, _i):
        m = self._current_model()
        if not m:
            return
        is_cls = m["task"] == "classification"
        self.slider_iou.setEnabled(not is_cls)
        self.slider_conf.setEnabled(True)

    # ------------------------------------------------------------------
    def _set_mode(self, folder):
        self.folder_mode = folder
        self.btn_single.setChecked(not folder); self.btn_batch.setChecked(folder)
        self._style_toggle()
        self.drop.set_folder_mode(folder)
        self.input_path = ""
        self.selected_lbl.setText("")
        self.result_scroll.setVisible(False)
        self.bar.setVisible(False)

    def _browse(self):
        if self.folder_mode:
            path = QFileDialog.getExistingDirectory(self, "Choose a folder of images")
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Choose an image", "",
                "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp)")
        if path:
            self._accept_input(path)

    def _on_dropped(self, path):
        self._accept_input(path)

    def _accept_input(self, path):
        if self.folder_mode:
            if not os.path.isdir(path):
                QMessageBox.warning(self, "Pick a folder", "Folder Batch mode needs a folder of images.")
                return
            from brain.inference import list_images
            n = len(list_images(path))
            if n == 0:
                QMessageBox.warning(self, "No images", "That folder has no supported images.")
                return
            self.input_path = path
            self.selected_lbl.setText(f"📂 {path}  ·  {n} image(s)")
        else:
            if not os.path.isfile(path):
                return
            if os.path.splitext(path)[1].lower() not in ACCEPT_EXT:
                QMessageBox.warning(self, "Unsupported file", "Please choose a JPEG, PNG, TIFF, BMP or WebP image.")
                return
            if os.path.getsize(path) > MAX_MB * 1024 * 1024:
                QMessageBox.warning(self, "Too large", f"Please choose an image under {MAX_MB} MB.")
                return
            self.input_path = path
            self.selected_lbl.setText(f"🖼 {os.path.basename(path)}")

    # ------------------------------------------------------------------
    def _style_run(self, enabled):
        if enabled:
            self.btn_run.setStyleSheet(
                f"QPushButton {{ background: {C_TEAL}; color: #fff; border: none; border-radius: 8px;"
                f"font-size: 15px; font-weight: 700; }} QPushButton:hover {{ background: {C_TEALH}; }}")
        else:
            self.btn_run.setStyleSheet(
                f"QPushButton {{ background: {C_BORDER}; color: {C_TEXT_DIM}; border: none; border-radius: 8px;"
                f"font-size: 15px; font-weight: 700; }}")

    def _run(self):
        if self.worker is not None:
            return
        m = self._current_model()
        if not m:
            return
        if not self.input_path:
            self.lbl_status.setText("Choose an image (or folder) on the right first.")
            return
        conf = self.slider_conf.value() / 100.0
        iou = self.slider_iou.value() / 100.0
        self._clear_results()
        self.btn_run.setEnabled(False); self._style_run(False)
        self.btn_cancel.setEnabled(True)
        self.lbl_status.setText("Loading model…")

        if self.folder_mode:
            self.bar.setVisible(True); self.bar.setRange(0, 100); self.bar.setValue(0)
            from brain.inference import BatchInferenceWorker
            self.worker = BatchInferenceWorker(m["file"], self.input_path, conf, iou, m["task"])
            self.worker.progress.connect(self._on_batch_progress)
            self.worker.done.connect(self._on_batch_done)
            self.worker.failed.connect(self._on_failed)
            self.worker.log.connect(self.lbl_status.setText)
        else:
            from brain.inference import InferenceWorker
            self.worker = InferenceWorker(m["file"], self.input_path, conf, iou, m["task"])
            self.worker.done.connect(self._on_single_done)
            self.worker.failed.connect(self._on_failed)
            self.worker.log.connect(self.lbl_status.setText)
        if self.status_callback:
            self.status_callback("Running inference…")
        self.worker.start()

    def _cancel(self):
        if self.worker is not None and hasattr(self.worker, "request_stop"):
            self.worker.request_stop()
            self.lbl_status.setText("Stopping…")

    def _finish(self):
        self.worker = None
        self.btn_run.setEnabled(True); self._style_run(True)
        self.btn_cancel.setEnabled(False)

    # ---- single ----
    def _on_single_done(self, rec):
        self.last_result = rec
        self._finish()
        self.bar.setVisible(False)
        self.lbl_status.setText(f"Done in {rec['elapsed']}s")
        self._show_single(rec)
        if self.status_callback:
            self.status_callback(f"Inference done · {sum(f['count'] for f in rec['findings'])} findings")

    def _show_single(self, rec):
        self._clear_results()
        self.result_scroll.setVisible(True)

        if rec.get("result_image") and os.path.isfile(rec["result_image"]):
            img = QLabel(); img.setAlignment(Qt.AlignCenter)
            pix = QPixmap(rec["result_image"])
            if pix.width() > 620:
                pix = pix.scaledToWidth(620, Qt.SmoothTransformation)
            img.setPixmap(pix)
            img.setStyleSheet(f"border: 1px solid {C_BORDER}; border-radius: 10px; background: {C_PANEL2}; padding: 6px;")
            self.result_lay.addWidget(img, 0, Qt.AlignHCenter)

        table = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(["Class", "Count", "Avg Confidence"])
        self._style_table(table)
        for f in rec["findings"]:
            r = table.rowCount(); table.insertRow(r)
            table.setItem(r, 0, QTableWidgetItem(str(f["class"])))
            table.setItem(r, 1, QTableWidgetItem(str(f["count"])))
            table.setItem(r, 2, QTableWidgetItem(f"{f['avg_conf']*100:.1f}%"))
        if table.rowCount() == 0:
            table.insertRow(0)
            table.setItem(0, 0, QTableWidgetItem("No objects above threshold"))
        table.setMaximumHeight(min(46 * (table.rowCount() + 1) + 8, 260))
        self.result_lay.addWidget(table)

        meta = QLabel(f"⏱ {rec['elapsed']}s  ·  model {rec.get('model') or '—'}")
        meta.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")
        self.result_lay.addWidget(meta)

        actions = QHBoxLayout(); actions.setSpacing(8)
        b_img = QPushButton("Save Image"); b_pdf = QPushButton("Export PDF Report")
        for b in (b_img, b_pdf):
            b.setCursor(Qt.PointingHandCursor); b.setMinimumHeight(38)
        b_img.setStyleSheet(self._action_qss()); b_pdf.setStyleSheet(self._action_qss(primary=True))
        b_img.clicked.connect(self._save_image)
        b_pdf.clicked.connect(self._export_pdf)
        actions.addWidget(b_img); actions.addWidget(b_pdf); actions.addStretch(1)
        wrap = QWidget(); wrap.setLayout(actions); wrap.setStyleSheet("background: transparent;")
        self.result_lay.addWidget(wrap)
        self.result_lay.addStretch(1)

    # ---- batch ----
    def _on_batch_progress(self, d):
        pct = int(d["i"] / max(d["n"], 1) * 100)
        self.bar.setValue(pct)
        self.lbl_status.setText(f"Analysing {d['i']}/{d['n']}: {d['image']}")

    def _on_batch_done(self, result):
        self._batch_result = result
        self._finish()
        self.lbl_status.setText(
            f"{'Stopped. ' if result['stopped'] else ''}Analysed {result['n']}/{result['total_images']} "
            f"images in {result['elapsed_total']}s")
        self._show_batch(result)
        if self.status_callback:
            self.status_callback(f"Batch done · {result['n']} images")

    def _show_batch(self, result):
        self._clear_results()
        self.result_scroll.setVisible(True)

        table = QTableWidget(0, 4)
        table.setHorizontalHeaderLabels(["Image", "Detections", "Classes", "Time (s)"])
        self._style_table(table)
        for row in result["rows"]:
            r = table.rowCount(); table.insertRow(r)
            table.setItem(r, 0, QTableWidgetItem(row["image"]))
            table.setItem(r, 1, QTableWidgetItem(str(row["detections"])))
            table.setItem(r, 2, QTableWidgetItem(row["classes"]))
            table.setItem(r, 3, QTableWidgetItem(f"{row['elapsed']}"))
        self.result_lay.addWidget(table, 1)

        agg = result.get("aggregate", {})
        if agg:
            summary = "  ·  ".join(f"{k}: {v}" for k, v in sorted(agg.items()))
            s = QLabel("Totals — " + summary); s.setWordWrap(True)
            s.setStyleSheet(f"color: {C_TEXT}; font-size: 13px; font-weight: 600;")
            self.result_lay.addWidget(s)

        actions = QHBoxLayout(); actions.setSpacing(8)
        b_csv = QPushButton("Export CSV"); b_xlsx = QPushButton("Export XLSX")
        for b in (b_csv, b_xlsx):
            b.setCursor(Qt.PointingHandCursor); b.setMinimumHeight(38)
        b_csv.setStyleSheet(self._action_qss()); b_xlsx.setStyleSheet(self._action_qss(primary=True))
        b_csv.clicked.connect(lambda: self._export_table("csv"))
        b_xlsx.clicked.connect(lambda: self._export_table("xlsx"))
        actions.addWidget(b_csv); actions.addWidget(b_xlsx); actions.addStretch(1)
        wrap = QWidget(); wrap.setLayout(actions); wrap.setStyleSheet("background: transparent;")
        self.result_lay.addWidget(wrap)

    def _on_failed(self, message):
        self._finish()
        self.bar.setVisible(False)
        first = message.strip().splitlines()[0] if message.strip() else "Unknown error."
        self.lbl_status.setText("⚠ " + first)
        if self.status_callback:
            self.status_callback("Inference failed")

    # ------------------------------------------------------------------
    def _clear_results(self):
        while self.result_lay.count():
            it = self.result_lay.takeAt(0)
            w = it.widget()
            if w:
                w.setParent(None); w.deleteLater()

    def _style_table(self, t):
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setShowGrid(False)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.setStyleSheet(f"""
            QTableWidget {{ background: {C_PANEL2}; color: {C_TEXT}; border: 1px solid {C_BORDER};
                border-radius: 10px; font-size: 13px; }}
            QHeaderView::section {{ background: {C_PANEL}; color: {C_TEXT_DIM}; padding: 9px 8px;
                border: none; border-bottom: 1px solid {C_BORDER}; font-weight: 700; font-size: 12px; }}
            QTableWidget::item {{ padding: 8px; border-bottom: 1px solid {C_BORDER}; }}
        """)

    def _action_qss(self, primary=False):
        if primary:
            return (f"QPushButton {{ background: {C_TEAL}; color: #fff; border: none; border-radius: 8px;"
                    f"padding: 8px 16px; font-size: 13px; font-weight: 700; }}"
                    f"QPushButton:hover {{ background: {C_TEALH}; }}")
        return (f"QPushButton {{ background: {C_BG}; color: {C_TEXT}; border: 1px solid {C_BORDER};"
                f"border-radius: 8px; padding: 8px 16px; font-size: 13px; }}"
                f"QPushButton:hover {{ border: 1px solid {C_TEAL}; }}")

    # ---- exports ----
    def _save_image(self):
        if not self.last_result or not self.last_result.get("result_image"):
            return
        dest, _ = QFileDialog.getSaveFileName(self, "Save result image",
                                              "result.png", "PNG image (*.png)")
        if dest:
            import shutil
            try:
                shutil.copy(self.last_result["result_image"], dest)
                QMessageBox.information(self, "Saved", f"Saved to:\n{dest}")
            except Exception as e:
                QMessageBox.warning(self, "Could not save", str(e))

    def _export_pdf(self):
        if not self.last_result:
            return
        os.makedirs(REPORTS, exist_ok=True)
        default = os.path.join(REPORTS, f"report_{self.last_result['timestamp']}.pdf")
        dest, _ = QFileDialog.getSaveFileName(self, "Export PDF report", default, "PDF (*.pdf)")
        if not dest:
            return
        try:
            self._make_pdf(self.last_result, dest)
            QMessageBox.information(self, "Report saved", f"PDF written to:\n{dest}")
        except Exception as e:
            QMessageBox.warning(self, "Could not export", str(e))

    def _make_pdf(self, rec, dest):
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
                                        Table, TableStyle)
        from reportlab.lib.styles import getSampleStyleSheet
        styles = getSampleStyleSheet()
        doc = SimpleDocTemplate(dest, pagesize=A4, title="Dr. Roy DT&R Report")
        flow = [Paragraph("Dr. Roy — Analysis Report", styles["Title"]),
                Paragraph(f"Image: {rec['image_name']}", styles["Normal"]),
                Paragraph(f"Model: {rec.get('model') or '—'} · Task: {rec['task']} · "
                          f"Time: {rec['elapsed']}s", styles["Normal"]),
                Spacer(1, 8)]
        if rec.get("result_image") and os.path.isfile(rec["result_image"]):
            flow.append(RLImage(rec["result_image"], width=150 * mm, height=150 * mm, kind="proportional"))
            flow.append(Spacer(1, 10))
        data = [["Class", "Count", "Avg Confidence"]]
        for f in rec["findings"]:
            data.append([f["class"], str(f["count"]), f"{f['avg_conf']*100:.1f}%"])
        if len(data) == 1:
            data.append(["No objects above threshold", "", ""])
        tbl = Table(data, hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#238f7a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        flow.append(tbl)
        doc.build(flow)

    def _export_table(self, kind):
        result = getattr(self, "_batch_result", None)
        if not result or not result.get("rows"):
            return
        import pandas as pd
        df = pd.DataFrame([{k: r[k] for k in ("image", "detections", "classes", "elapsed")}
                           for r in result["rows"]])
        df = df.rename(columns={"image": "Image", "detections": "Detections",
                                "classes": "Classes", "elapsed": "Time (s)"})
        ext = "csv" if kind == "csv" else "xlsx"
        default = os.path.join(APP_ROOT, "output", f"batch_results.{ext}")
        dest, _ = QFileDialog.getSaveFileName(self, f"Export {ext.upper()}", default,
                                              f"{ext.upper()} (*.{ext})")
        if not dest:
            return
        try:
            if kind == "csv":
                df.to_csv(dest, index=False)
            else:
                df.to_excel(dest, index=False)
            QMessageBox.information(self, "Exported", f"Saved to:\n{dest}")
        except Exception as e:
            QMessageBox.warning(self, "Could not export", str(e))
