"""
Train Model page — LAYOUT + FORM STATE ONLY.

No AI is connected here. The 'Start Training' button only becomes clickable
once a dataset folder is chosen; the actual training worker is wired up later.

Call .get_config() to read the whole form back as a dict (used by later steps).
"""

import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QLineEdit, QComboBox, QSpinBox, QCheckBox, QSlider, QScrollArea,
    QFileDialog, QSizePolicy, QProgressBar,
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- palette (matches main.py) ---
C_BG        = "#0d1117"
C_PANEL     = "#161b27"
C_PANEL2    = "#1c2333"
C_BORDER    = "#2a3045"
C_TEXT      = "#e6edf3"
C_TEXT_DIM  = "#8b949e"
C_TEAL      = "#238f7a"
C_TEALH     = "#2aad93"
C_INACTIVE  = "#3a4255"
C_BLUE      = "#3b82f6"
C_BLUE_BG   = "#16233f"
C_GREEN     = "#3fb950"

_FIELD_QSS = f"""
    QLineEdit, QComboBox, QSpinBox {{
        background: {C_BG}; color: {C_TEXT};
        border: 1px solid {C_BORDER}; border-radius: 7px;
        padding: 8px 10px; font-size: 14px;
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{ border: 1px solid {C_TEAL}; }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox QAbstractItemView {{
        background: {C_PANEL2}; color: {C_TEXT};
        selection-background-color: {C_TEAL};
        border: 1px solid {C_BORDER}; outline: none;
    }}
    QSpinBox::up-button, QSpinBox::down-button {{ width: 18px; }}
"""

_SLIDER_QSS = f"""
    QSlider::groove:horizontal {{
        height: 6px; border-radius: 3px; background: {C_BORDER};
    }}
    QSlider::sub-page:horizontal {{
        height: 6px; border-radius: 3px; background: {C_TEAL};
    }}
    QSlider::handle:horizontal {{
        width: 18px; height: 18px; margin: -7px 0; border-radius: 9px;
        background: {C_TEALH}; border: 2px solid {C_BG};
    }}
"""

# Pill toggle made from a checkbox indicator
_TOGGLE_QSS = f"""
    QCheckBox {{ color: {C_TEXT}; font-size: 14px; spacing: 12px; }}
    QCheckBox::indicator {{ width: 44px; height: 22px; border-radius: 11px;
        background: {C_INACTIVE}; }}
    QCheckBox::indicator:checked {{ background: {C_TEAL}; }}
"""


def _fmt_time(secs):
    secs = int(max(0, secs))
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


class LiveChart(FigureCanvas):
    """A small dark live line chart that you feed one (x, y) point at a time."""

    def __init__(self, title, ylabel, color, ymax=None):
        fig = Figure(figsize=(4.2, 2.4), facecolor=C_PANEL2)
        super().__init__(fig)
        self._color = color
        self._ymax = ymax
        self.ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.16, right=0.97, top=0.82, bottom=0.22)
        self.ax.set_facecolor(C_BG)
        self.ax.set_title(title, color=C_TEXT, fontsize=10, fontweight="bold")
        self.ax.set_xlabel("Round", color=C_TEXT_DIM, fontsize=8)
        self.ax.set_ylabel(ylabel, color=C_TEXT_DIM, fontsize=8)
        for spine in self.ax.spines.values():
            spine.set_color(C_BORDER)
        self.ax.tick_params(colors=C_TEXT_DIM, labelsize=7)
        self.ax.grid(True, color=C_BORDER, linewidth=0.4, alpha=0.5)
        self.xs, self.ys = [], []
        (self.line,) = self.ax.plot([], [], color=color, linewidth=2, marker="o", markersize=3)
        if ymax is not None:
            self.ax.set_ylim(0, ymax)
        self.draw_idle()

    def add(self, x, y):
        self.xs.append(x)
        self.ys.append(y)
        self.line.set_data(self.xs, self.ys)
        self.ax.set_xlim(0.5, max(self.xs) + 0.5)
        if self._ymax is None:
            top = max(self.ys) * 1.15 or 1.0
            self.ax.set_ylim(0, top)
        self.draw_idle()

    def reset(self):
        self.xs, self.ys = [], []
        self.line.set_data([], [])
        self.draw_idle()


class TaskCard(QFrame):
    """Click-to-select task card with a teal 'Continue' link at the bottom."""

    selected = pyqtSignal()
    continue_clicked = pyqtSignal()

    def __init__(self, icon, title, desc, link_text):
        super().__init__()
        self.setCursor(Qt.PointingHandCursor)
        self._selected = False
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(8)

        head = QLabel(f"{icon}  {title}")
        head.setStyleSheet(f"color: {C_TEXT}; font-size: 17px; font-weight: 700; background: transparent;")
        lay.addWidget(head)

        d = QLabel(desc)
        d.setWordWrap(True)
        d.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px; background: transparent;")
        lay.addWidget(d)
        lay.addStretch(1)

        self._link = QLabel(link_text)
        self._link.setCursor(Qt.PointingHandCursor)
        self._link.setStyleSheet(f"color: {C_TEALH}; font-size: 14px; font-weight: 600; background: transparent;")
        # clicking the link selects this task AND advances
        self._link.mousePressEvent = self._on_link_click
        lay.addWidget(self._link)

        self._restyle()

    def _on_link_click(self, event):
        self.selected.emit()
        self.continue_clicked.emit()

    def mousePressEvent(self, event):
        self.selected.emit()
        super().mousePressEvent(event)

    def set_selected(self, value):
        self._selected = bool(value)
        self._restyle()

    def _restyle(self):
        border, bg = (C_BLUE, C_BLUE_BG) if self._selected else (C_BORDER, C_PANEL2)
        self.setStyleSheet(
            f"TaskCard {{ background: {bg}; border: 2px solid {border}; border-radius: 12px; }}")


class TrainModelPage(QWidget):
    def __init__(self, status_callback=None):
        super().__init__()
        self.status_callback = status_callback
        self.selected_task = None
        self.dataset_folder = ""

        self.setStyleSheet(f"background: {C_BG};")

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        page = QWidget()
        page.setStyleSheet("background: transparent;")
        root = QVBoxLayout(page)
        root.setContentsMargins(40, 32, 40, 40)
        root.setSpacing(10)

        # ---- Title ----
        title = QLabel("Fine-Tune AI on Your Own Dataset")
        title.setStyleSheet(f"color: {C_TEXT}; font-size: 26px; font-weight: 800;")
        root.addWidget(title)

        subtitle = QLabel("Pick a task, point the app at your image folder, and configure the run.")
        subtitle.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 14px;")
        root.addWidget(subtitle)
        root.addSpacing(16)

        # ---- Two task cards ----
        self.card_detection = TaskCard(
            "🎯", "Object Detection",
            "Count and locate objects in slide patches — WBC differential, mitosis, malaria screen.",
            "Continue — drop dataset folder")
        self.card_classification = TaskCard(
            "🏷️", "Image Classification",
            "Assign one label per image — tumour subtype, Gleason grade, smear pattern.",
            "Continue — drop ImageFolder")
        self.card_detection.selected.connect(lambda: self._select_task("detection"))
        self.card_classification.selected.connect(lambda: self._select_task("classification"))
        self.card_detection.continue_clicked.connect(self._on_browse)
        self.card_classification.continue_clicked.connect(self._on_browse)

        cards_row = QHBoxLayout()
        cards_row.setSpacing(16)
        cards_row.addWidget(self.card_detection)
        cards_row.addWidget(self.card_classification)
        root.addLayout(cards_row)
        root.addSpacing(22)

        # ---- Configuration panel ----
        cfg_hdr = QLabel("Configuration")
        cfg_hdr.setStyleSheet(f"color: {C_TEXT}; font-size: 18px; font-weight: 700;")
        root.addWidget(cfg_hdr)
        root.addSpacing(4)

        panel = QFrame()
        panel.setStyleSheet(
            f"QFrame {{ background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 12px; }}")
        grid = QGridLayout(panel)
        grid.setContentsMargins(22, 20, 22, 20)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(16)
        grid.setColumnStretch(1, 1)

        r = 0
        # Model Size
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["Nano", "Small", "Medium", "Large"])
        self.cmb_model.setStyleSheet(_FIELD_QSS)
        grid.addWidget(self._label("Model Size"), r, 0)
        grid.addWidget(self.cmb_model, r, 1); r += 1

        # Training Rounds
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 2000)
        self.spin_epochs.setValue(50)
        self.spin_epochs.setStyleSheet(_FIELD_QSS)
        grid.addWidget(self._label("Training Rounds"), r, 0)
        grid.addWidget(self.spin_epochs, r, 1); r += 1

        # Image Size
        self.cmb_imgsize = QComboBox()
        self.cmb_imgsize.addItems(["320", "416", "640", "1024"])
        self.cmb_imgsize.setCurrentText("640")
        self.cmb_imgsize.setStyleSheet(_FIELD_QSS)
        grid.addWidget(self._label("Image Size"), r, 0)
        grid.addWidget(self.cmb_imgsize, r, 1); r += 1

        # Training Speed slider
        speed_box = QVBoxLayout()
        speed_box.setSpacing(4)
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(1, 5)
        self.slider_speed.setValue(3)
        self.slider_speed.setStyleSheet(_SLIDER_QSS)
        ends = QHBoxLayout()
        l_lo = QLabel("Accurate"); l_lo.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px; background: transparent;")
        l_hi = QLabel("Fast"); l_hi.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px; background: transparent;")
        ends.addWidget(l_lo, 0, Qt.AlignLeft)
        ends.addStretch(1)
        ends.addWidget(l_hi, 0, Qt.AlignRight)
        speed_box.addWidget(self.slider_speed)
        speed_box.addLayout(ends)
        grid.addWidget(self._label("Training Speed"), r, 0, Qt.AlignTop)
        grid.addLayout(speed_box, r, 1); r += 1

        # Auto-save best result toggle (ON)
        self.toggle_autosave = QCheckBox("Auto-save best result")
        self.toggle_autosave.setChecked(True)
        self.toggle_autosave.setCursor(Qt.PointingHandCursor)
        self.toggle_autosave.setStyleSheet(_TOGGLE_QSS)
        grid.addWidget(self._label("Auto-save"), r, 0)
        grid.addWidget(self.toggle_autosave, r, 1); r += 1

        # Dataset folder + Browse
        folder_row = QHBoxLayout()
        folder_row.setSpacing(10)
        self.in_folder = QLineEdit()
        self.in_folder.setPlaceholderText("Select your dataset folder…")
        self.in_folder.setStyleSheet(_FIELD_QSS)
        self.in_folder.textChanged.connect(self._on_folder_text)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.setCursor(Qt.PointingHandCursor)
        self.btn_browse.setStyleSheet(self._secondary_btn_qss())
        self.btn_browse.clicked.connect(self._on_browse)
        folder_row.addWidget(self.in_folder, 1)
        folder_row.addWidget(self.btn_browse, 0)
        grid.addWidget(self._label("Dataset folder"), r, 0)
        grid.addLayout(folder_row, r, 1); r += 1

        root.addWidget(panel)
        root.addSpacing(22)

        # ---- Start Training button ----
        self.btn_start = QPushButton("Start Training")
        self.btn_start.setMinimumHeight(52)
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self._on_start)
        root.addWidget(self.btn_start)

        self.lbl_hint = QLabel("Select a dataset folder to enable training.")
        self.lbl_hint.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")
        root.addWidget(self.lbl_hint)

        # ---- Progress panel (hidden until training starts) ----
        self.progress_panel = self._build_progress_panel()
        self.progress_panel.setVisible(False)
        root.addSpacing(10)
        root.addWidget(self.progress_panel)

        root.addStretch(1)

        self.worker = None
        scroll.setWidget(page)
        wrap = QVBoxLayout(self)
        wrap.setContentsMargins(0, 0, 0, 0)
        wrap.addWidget(scroll)

        self._validate()

    # ------------------------------------------------------------------
    def _label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px; font-weight: 600; background: transparent;")
        return lbl

    def _secondary_btn_qss(self):
        return f"""
            QPushButton {{ background: {C_BG}; color: {C_TEXT};
                border: 1px solid {C_BORDER}; border-radius: 7px;
                padding: 8px 16px; font-size: 13px; font-weight: 600; }}
            QPushButton:hover {{ border: 1px solid {C_TEAL}; }}
        """

    # ------------------------------------------------------------------
    def _select_task(self, task):
        self.selected_task = task
        self.card_detection.set_selected(task == "detection")
        self.card_classification.set_selected(task == "classification")
        self._validate()

    def _on_browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select dataset folder")
        if folder:
            self.in_folder.setText(folder)   # triggers _on_folder_text -> validate

    def _on_folder_text(self, text):
        self.dataset_folder = text.strip()
        self._validate()

    def _validate(self):
        ready = bool(self.dataset_folder)
        self.btn_start.setEnabled(ready)
        self.btn_start.setStyleSheet(self._start_qss(ready))
        self.lbl_hint.setVisible(not ready)
        return ready

    def _start_qss(self, enabled):
        if enabled:
            return f"""
                QPushButton {{ background: {C_TEAL}; color: #ffffff; border: none;
                    border-radius: 10px; font-size: 16px; font-weight: 700; }}
                QPushButton:hover {{ background: {C_TEALH}; }}
            """
        return f"""
            QPushButton {{ background: {C_INACTIVE}; color: {C_TEXT_DIM}; border: none;
                border-radius: 10px; font-size: 16px; font-weight: 700; }}
        """

    # ------------------------------------------------------------------
    # Progress panel
    # ------------------------------------------------------------------
    def _build_progress_panel(self):
        panel = QFrame()
        panel.setStyleSheet(
            f"QFrame#prog {{ background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 12px; }}")
        panel.setObjectName("prog")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(22, 20, 22, 20)
        lay.setSpacing(12)

        # header: round + status
        top = QHBoxLayout()
        self.lbl_phase = QLabel("Preparing…")
        self.lbl_phase.setStyleSheet(f"color: {C_TEXT}; font-size: 17px; font-weight: 700; background: transparent;")
        self.lbl_round = QLabel("Round 0/0")
        self.lbl_round.setStyleSheet(f"color: {C_TEALH}; font-size: 15px; font-weight: 700; background: transparent;")
        top.addWidget(self.lbl_phase, 0)
        top.addStretch(1)
        top.addWidget(self.lbl_round, 0)
        lay.addLayout(top)

        # progress bar
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(True)
        self.bar.setFixedHeight(20)
        self.bar.setStyleSheet(f"""
            QProgressBar {{ background: {C_BG}; border: 1px solid {C_BORDER};
                border-radius: 6px; text-align: center; color: {C_TEXT}; font-size: 11px; }}
            QProgressBar::chunk {{ background: {C_TEAL}; border-radius: 5px; }}
        """)
        lay.addWidget(self.bar)

        # charts
        charts = QHBoxLayout()
        charts.setSpacing(14)
        self.chart_acc = LiveChart("Live Accuracy", "score", C_TEALH, ymax=1.0)
        self.chart_loss = LiveChart("Live Error Rate", "loss", "#e06c75")
        for c in (self.chart_acc, self.chart_loss):
            c.setMinimumHeight(190)
            charts.addWidget(c)
        lay.addLayout(charts)

        # elapsed / eta
        times = QHBoxLayout()
        self.lbl_elapsed = QLabel("Elapsed: 0s")
        self.lbl_eta = QLabel("Est. left: —")
        for l in (self.lbl_elapsed, self.lbl_eta):
            l.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px; background: transparent;")
        times.addWidget(self.lbl_elapsed, 0)
        times.addStretch(1)
        times.addWidget(self.lbl_eta, 0)
        lay.addLayout(times)

        # completion banner (hidden until done)
        self.lbl_complete = QLabel("")
        self.lbl_complete.setWordWrap(True)
        self.lbl_complete.setVisible(False)
        self.lbl_complete.setStyleSheet(
            f"color: {C_GREEN}; font-size: 16px; font-weight: 700; background: transparent;")
        lay.addWidget(self.lbl_complete)

        # action row: Stop / Use This Model
        actions = QHBoxLayout()
        self.btn_stop = QPushButton("Stop Training")
        self.btn_stop.setMinimumHeight(42)
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.setStyleSheet(f"""
            QPushButton {{ background: #5a2330; color: #ffb3bf; border: 1px solid #7a2e3e;
                border-radius: 8px; font-size: 14px; font-weight: 700; padding: 0 18px; }}
            QPushButton:hover {{ background: #6e2a3a; }}
        """)
        self.btn_stop.clicked.connect(self._on_stop)

        self.btn_use = QPushButton("Use This Model")
        self.btn_use.setMinimumHeight(42)
        self.btn_use.setCursor(Qt.PointingHandCursor)
        self.btn_use.setVisible(False)
        self.btn_use.setStyleSheet(f"""
            QPushButton {{ background: {C_TEAL}; color: #ffffff; border: none;
                border-radius: 8px; font-size: 14px; font-weight: 700; padding: 0 22px; }}
            QPushButton:hover {{ background: {C_TEALH}; }}
        """)
        self.btn_use.clicked.connect(self._on_use_model)

        actions.addWidget(self.btn_stop, 0)
        actions.addStretch(1)
        actions.addWidget(self.btn_use, 0)
        lay.addLayout(actions)

        self._last_model_path = ""
        self._last_model_name = ""
        return panel

    # ------------------------------------------------------------------
    # Training flow
    # ------------------------------------------------------------------
    def _infer_task(self):
        """Use the chosen task, or auto-detect it from the folder structure."""
        import brain.dataset as dataset
        if self.selected_task:
            return self.selected_task, ""
        if dataset.detect_format(self.dataset_folder):
            return "detection", "Auto-detected an object-detection dataset."
        if dataset.detect_classification(self.dataset_folder):
            return "classification", "Auto-detected an image-classification dataset."
        return None, ("Could not recognise this dataset. Choose a task above, or check the folder "
                      "contains a supported format (Roboflow/plain YOLO, COCO, VOC, or an ImageFolder).")

    def _on_start(self):
        if not self._validate():
            return
        if self.worker is not None:
            return  # already running

        task, note = self._infer_task()
        if task is None:
            self.lbl_hint.setVisible(True)
            self.lbl_hint.setText("⚠ " + note)
            self.lbl_hint.setStyleSheet("color: #e06c75; font-size: 12px;")
            return
        self.selected_task = task
        self.card_detection.set_selected(task == "detection")
        self.card_classification.set_selected(task == "classification")

        # reset + reveal progress panel
        self.chart_acc.reset()
        self.chart_loss.reset()
        self.bar.setValue(0)
        self.lbl_phase.setText("Preparing dataset…")
        self.lbl_round.setText("Round 0/0")
        self.lbl_elapsed.setText("Elapsed: 0s")
        self.lbl_eta.setText("Est. left: —")
        self.lbl_complete.setVisible(False)
        self.btn_use.setVisible(False)
        self.btn_stop.setVisible(True)
        self.btn_stop.setEnabled(True)
        self.progress_panel.setVisible(True)

        # lock the form while running
        self.btn_start.setEnabled(False)
        self.btn_start.setStyleSheet(self._start_qss(False))
        self.btn_start.setText("Training…")
        self.lbl_hint.setVisible(False)

        cfg = self.get_config()
        cfg["task"] = task
        cfg["dataset_name"] = os.path.basename(os.path.normpath(self.dataset_folder))
        cfg["dataset_path"] = self.dataset_folder

        from brain.trainer import TrainingWorker
        self.worker = TrainingWorker(cfg)
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._on_log)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

        if self.status_callback:
            self.status_callback("Training started…")

    def _on_progress(self, d):
        self.lbl_phase.setText("Training in progress")
        self.lbl_round.setText(f"Round {d['epoch']}/{d['total']}")
        pct = int(d["epoch"] / max(d["total"], 1) * 100)
        self.bar.setValue(pct)
        self.chart_acc.add(d["epoch"], d["metric"])
        self.chart_loss.add(d["epoch"], d["loss"])
        self.lbl_elapsed.setText("Elapsed: " + _fmt_time(d["elapsed"]))
        self.lbl_eta.setText("Est. left: " + _fmt_time(d["eta"]))

    def _on_log(self, msg):
        self.lbl_phase.setText(msg)
        if self.status_callback:
            self.status_callback(msg)

    def _on_done(self, result):
        self.worker = None
        stopped = result.get("status") == "Stopped"
        best = result.get("metric", 0.0)
        name = result.get("model", "")
        self._last_model_path = result.get("model_path", "")
        self._last_model_name = name

        self.bar.setValue(100)
        self.lbl_round.setText(f"Round {result.get('epochs', 0)}/{result.get('requested_epochs', 0)}")
        self.lbl_phase.setText("Training Stopped" if stopped else "Training Complete!")
        metric_word = "Top-1 accuracy" if result.get("task") == "classification" else "mAP@50-95"
        msg = f"{'Stopped early. ' if stopped else ''}Best {metric_word}: {best*100:.1f}%."
        if name:
            msg += f"  Saved as {name}."
        self.lbl_complete.setVisible(True)
        self.lbl_complete.setStyleSheet(
            f"color: {'#e5c07b' if stopped else C_GREEN}; font-size: 16px; font-weight: 700; background: transparent;")
        self.lbl_complete.setText(("⚠ " if stopped else "✓ ") + msg)

        self.btn_stop.setVisible(False)
        self.btn_use.setVisible(bool(self._last_model_path))
        self.btn_use.setEnabled(bool(self._last_model_path))

        self._unlock_form()
        if self.status_callback:
            self.status_callback(f"AI Model: {name or 'training run'} ready")

    def _on_failed(self, message):
        self.worker = None
        self.lbl_phase.setText("Training could not start")
        self.lbl_complete.setVisible(True)
        self.lbl_complete.setStyleSheet("color: #e06c75; font-size: 14px; font-weight: 600; background: transparent;")
        first_line = message.strip().splitlines()[0] if message.strip() else "Unknown error."
        self.lbl_complete.setText("⚠ " + first_line)
        self.btn_stop.setVisible(False)
        self._unlock_form()
        if self.status_callback:
            self.status_callback("Training failed — see the Train Model page.")

    def _on_stop(self):
        if self.worker is not None:
            self.btn_stop.setEnabled(False)
            self.btn_stop.setText("Stopping after this round…")
            self.worker.request_stop()

    def _on_use_model(self):
        self.btn_use.setText("✓ Model in use")
        self.btn_use.setEnabled(False)
        if self.status_callback and self._last_model_name:
            self.status_callback(f"AI Model: {self._last_model_name} (Loaded)")

    def _unlock_form(self):
        self.btn_start.setEnabled(True)
        self.btn_start.setText("Start Training")
        self._validate()

    def load_dataset_path(self, folder, task=None):
        """Pre-load a dataset folder (from the Dataset Library) without auto-starting."""
        if task:
            self._select_task(task)
        if folder:
            self.in_folder.setText(folder)

    def prefill_and_run(self, entry):
        """Prefill the form from a history entry and auto-start if the folder still exists."""
        if self.worker is not None:
            return False  # a run is already in progress
        task = entry.get("task")
        if task:
            self._select_task(task)
        if entry.get("model_size"):
            self.cmb_model.setCurrentText(entry["model_size"])
        ep = entry.get("requested_epochs") or entry.get("epochs")
        if ep:
            self.spin_epochs.setValue(int(ep))
        if entry.get("img_size"):
            self.cmb_imgsize.setCurrentText(str(entry["img_size"]))
        folder = entry.get("dataset_path", "")
        if folder:
            self.in_folder.setText(folder)
        if folder and os.path.isdir(folder):
            self._on_start()
            return True
        # folder gone — leave the form prefilled and tell the user
        self.lbl_hint.setVisible(True)
        self.lbl_hint.setText("⚠ Original dataset folder not found — choose it again, then press Start Training.")
        self.lbl_hint.setStyleSheet("color: #e5c07b; font-size: 12px;")
        return False

    # ------------------------------------------------------------------
    def get_config(self):
        return {
            "task": self.selected_task,
            "model_size": self.cmb_model.currentText(),
            "training_rounds": self.spin_epochs.value(),
            "image_size": int(self.cmb_imgsize.currentText()),
            "training_speed": self.slider_speed.value(),
            "auto_save_best": self.toggle_autosave.isChecked(),
            "dataset_folder": self.dataset_folder,
        }
