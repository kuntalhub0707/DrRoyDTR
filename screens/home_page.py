"""
Home / Overview page.

Dashboard landing screen: phase label, headline, four live summary counters,
three feature shortcut cards, a Colab status section, and a system-status line.

The summary counters read real numbers from the app's data folders/JSON when
they exist (so they update as the app is used) and default to 0 otherwise.

The Colab Status section becomes fully live in Step 16 (once monitoring data
exists). For now it shows the empty-state message and links to the Train on
Colab page already built in Step 3.

Navigation: the page is given a `navigate(index)` callback (MainWindow._go).
Stack indices — Train Model=1, Train on Colab=2, Models=5, Predict/Analyze=6.
"""

import os
import json

from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QScrollArea, QSizePolicy,
)

# --- palette (matches main.py) ---
C_BG        = "#0d1117"
C_PANEL     = "#161b27"
C_PANEL2    = "#1c2333"
C_BORDER    = "#2a3045"
C_TEXT      = "#e6edf3"
C_TEXT_DIM  = "#8b949e"
C_TEAL      = "#238f7a"
C_TEALH     = "#2aad93"
C_GREEN     = "#3fb950"   # status dot

from brain.paths import APP_ROOT

# Stack indices this page can jump to
IDX_TRAIN_MODEL = 1
IDX_TRAIN_COLAB = 2
IDX_HISTORY     = 3
IDX_MODELS      = 5
IDX_PREDICT     = 6

C_BLUE  = "#3b82f6"
C_AMBER = "#e5c07b"


def _count_files(folder, ext):
    path = os.path.join(APP_ROOT, folder)
    if not os.path.isdir(path):
        return 0
    return sum(1 for f in os.listdir(path) if f.lower().endswith(ext))


def _parse_ts(ts):
    import time
    try:
        return time.mktime(time.strptime(ts, "%Y%m%d_%H%M%S"))
    except Exception:
        return None


def _count_json_list(filename):
    path = os.path.join(APP_ROOT, filename)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            return len(data)
    except Exception:
        pass
    return 0


class FeatureCard(QFrame):
    """A clickable shortcut card: title, blurb, and a teal link line."""

    clicked = pyqtSignal()

    def __init__(self, icon, title, blurb, link_text):
        super().__init__()
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._restyle(False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 18, 20, 18)
        lay.setSpacing(8)

        head = QLabel(f"{icon}  {title}")
        head.setStyleSheet(f"color: {C_TEXT}; font-size: 17px; font-weight: 700; background: transparent;")
        lay.addWidget(head)

        b = QLabel(blurb)
        b.setWordWrap(True)
        b.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px; background: transparent;")
        lay.addWidget(b)

        lay.addStretch(1)

        link = QLabel(link_text)
        link.setStyleSheet(f"color: {C_TEALH}; font-size: 14px; font-weight: 600; background: transparent;")
        lay.addWidget(link)

    def enterEvent(self, event):
        self._restyle(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._restyle(False)
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def _restyle(self, hover):
        border = C_TEAL if hover else C_BORDER
        self.setStyleSheet(
            f"FeatureCard {{ background: {C_PANEL2}; border: 1px solid {border}; border-radius: 12px; }}")


class HomePage(QWidget):
    def __init__(self, navigate=None):
        super().__init__()
        self.navigate = navigate or (lambda _i: None)
        self.setStyleSheet(f"background: {C_BG};")

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        page = QWidget()
        page.setStyleSheet("background: transparent;")
        root = QVBoxLayout(page)
        root.setContentsMargins(40, 32, 40, 36)
        root.setSpacing(10)

        # ---- Phase label + headline + description ----
        phase = QLabel("PHASE 0 · OVERVIEW")
        phase.setStyleSheet(
            f"color: {C_TEAL}; font-size: 12px; font-weight: 700; letter-spacing: 2px;")
        root.addWidget(phase)

        heading = QLabel("Bringing AI into the histopathology and hematology reporting workflow.")
        heading.setWordWrap(True)
        heading.setStyleSheet(f"color: {C_TEXT}; font-size: 28px; font-weight: 800; line-height: 130%;")
        root.addWidget(heading)

        desc = QLabel(
            "Dr. Roy DT&R lets you train your own AI models on annotated slide images, run "
            "predictions on new cases, and turn the results into clinical PDF reports — entirely "
            "from your desktop. When you need more power, training can run on a free Google Colab "
            "cloud GPU and the finished model comes straight back into the app.")
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 14px;")
        root.addWidget(desc)
        root.addSpacing(18)

        # ---- Four summary cards ----
        counts = self._read_counts()
        summary_row = QHBoxLayout()
        summary_row.setSpacing(16)
        self._summary_widgets = {}
        for key, label, icon in [
            ("models",   "Total Models",       "🧠"),
            ("datasets", "Datasets Uploaded",  "🗂️"),
            ("runs",     "Training Runs",       "🤖"),
            ("reports",  "Reports Generated",  "📄"),
            ("colab",    "Colab Runs",         "☁️"),
        ]:
            card, num_lbl = self._summary_card(icon, str(counts[key]), label)
            self._summary_widgets[key] = num_lbl
            summary_row.addWidget(card)
        root.addLayout(summary_row)
        root.addSpacing(26)

        # ---- Three feature cards ----
        feat_hdr = QLabel("Quick Actions")
        feat_hdr.setStyleSheet(f"color: {C_TEXT}; font-size: 16px; font-weight: 700;")
        root.addWidget(feat_hdr)
        root.addSpacing(4)

        feat_row = QHBoxLayout()
        feat_row.setSpacing(16)

        c_train = FeatureCard("🤖", "Train Model",
                              "Train a YOLO model on your annotated slide patches.",
                              "Open Train Model →")
        c_train.clicked.connect(lambda: self.navigate(IDX_TRAIN_MODEL))

        c_predict = FeatureCard("🔬", "Predict / Analyze",
                                "Run a trained model on a slide image and see findings.",
                                "Open Predict →")
        c_predict.clicked.connect(lambda: self.navigate(IDX_PREDICT))

        c_models = FeatureCard("🧠", "Models Library",
                               "Browse, load and manage your trained models.",
                               "Browse Models →")
        c_models.clicked.connect(lambda: self.navigate(IDX_MODELS))

        for c in (c_train, c_predict, c_models):
            feat_row.addWidget(c)
        root.addLayout(feat_row)
        root.addSpacing(26)

        # ---- Colab status section (rebuilt live) ----
        colab_card = QFrame()
        colab_card.setStyleSheet(
            f"QFrame {{ background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 12px; }}")
        cl = QVBoxLayout(colab_card)
        cl.setContentsMargins(22, 18, 22, 18)
        cl.setSpacing(10)

        colab_hdr = QLabel("☁️ Active Colab Sessions")
        colab_hdr.setStyleSheet(f"color: {C_TEXT}; font-size: 16px; font-weight: 700; background: transparent;")
        cl.addWidget(colab_hdr)

        self.colab_body = QVBoxLayout()
        self.colab_body.setSpacing(10)
        cl.addLayout(self.colab_body)

        root.addWidget(colab_card)
        root.addSpacing(20)

        self._running_dots = []          # animated dots for in-progress sessions
        self._anim_on = True
        self._rebuild_colab_section()

        # animate the "in progress" badges
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._animate_badges)
        self._anim_timer.start(600)
        # auto-refresh the dashboard every 2 minutes
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._auto_refresh)
        self._refresh_timer.start(120000)

        # ---- System status line ----
        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {C_GREEN}; font-size: 14px; background: transparent;")
        st = QLabel("System Status: Ready")
        st.setStyleSheet(f"color: {C_TEXT}; font-size: 13px; font-weight: 600; background: transparent;")
        status_row.addWidget(dot, 0)
        status_row.addWidget(st, 0)
        status_row.addStretch(1)
        root.addLayout(status_row)

        root.addStretch(1)

        scroll.setWidget(page)
        wrap = QVBoxLayout(self)
        wrap.setContentsMargins(0, 0, 0, 0)
        wrap.addWidget(scroll)

    # ------------------------------------------------------------------
    def _summary_card(self, icon, value, label):
        card = QFrame()
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        card.setStyleSheet(
            f"QFrame {{ background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 12px; }}")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(2)

        top = QLabel(icon)
        top.setStyleSheet("font-size: 20px; background: transparent;")
        lay.addWidget(top)

        num = QLabel(value)
        num.setStyleSheet(f"color: {C_TEXT}; font-size: 32px; font-weight: 800; background: transparent;")
        lay.addWidget(num)

        cap = QLabel(label)
        cap.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px; background: transparent;")
        lay.addWidget(cap)
        return card, num

    def _read_counts(self):
        import brain.sessions as sessions
        return {
            "models":   _count_files("models", ".pt"),
            "datasets": _count_json_list("datasets.json"),
            "runs":     _count_json_list("training_history.json"),
            "reports":  _count_files("reports", ".pdf"),
            "colab":    sessions.completed_count(),
        }

    def refresh_counts(self):
        """Re-read the data folders and update the counters live."""
        counts = self._read_counts()
        for key, lbl in self._summary_widgets.items():
            lbl.setText(str(counts.get(key, 0)))

    # ------------------------------------------------------------------
    # Live Colab status
    # ------------------------------------------------------------------
    def showEvent(self, event):
        self.refresh_counts()
        self._rebuild_colab_section()
        super().showEvent(event)

    def _auto_refresh(self):
        self.refresh_counts()
        self._rebuild_colab_section()

    def _animate_badges(self):
        self._anim_on = not self._anim_on
        col = C_BLUE if self._anim_on else C_PANEL2
        for d in list(self._running_dots):
            try:
                d.setStyleSheet(f"color: {col}; font-size: 14px; background: transparent;")
            except RuntimeError:
                pass

    def _rebuild_colab_section(self):
        import brain.sessions as sessions
        self._running_dots = []
        while self.colab_body.count():
            it = self.colab_body.takeAt(0)
            w = it.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        items = sessions.visible_sessions()
        if not items:
            self.colab_body.addWidget(self._empty_colab())
            return
        for s in items:
            self.colab_body.addWidget(self._session_card(s))

    def _empty_colab(self):
        w = QWidget(); w.setStyleSheet("background: transparent;")
        row = QHBoxLayout(w); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(6)
        msg = QLabel("No active Colab sessions.")
        msg.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 14px; background: transparent;")
        link = QPushButton("Start one from Train on Colab →")
        link.setCursor(Qt.PointingHandCursor); link.setFlat(True)
        link.setStyleSheet(
            f"QPushButton {{ color: {C_TEALH}; font-size: 14px; font-weight: 600; "
            f"background: transparent; border: none; text-align: left; }}"
            f"QPushButton:hover {{ color: {C_TEAL}; text-decoration: underline; }}")
        link.clicked.connect(lambda: self.navigate(IDX_TRAIN_COLAB))
        row.addWidget(msg, 0); row.addWidget(link, 0); row.addStretch(1)
        return w

    def _fmt_elapsed(self, secs):
        secs = int(max(0, secs))
        m, s = divmod(secs, 60); h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m"
        if m:
            return f"{m}m {s}s"
        return f"{s}s"

    def _session_card(self, s):
        import time
        card = QFrame()
        card.setStyleSheet(f"QFrame {{ background: {C_BG}; border: 1px solid {C_BORDER}; border-radius: 10px; }}")
        h = QHBoxLayout(card); h.setContentsMargins(16, 12, 16, 12); h.setSpacing(12)

        left = QVBoxLayout(); left.setSpacing(4)
        name = QLabel(s.get("run_name", "run"))
        name.setStyleSheet(f"color: {C_TEXT}; font-size: 14px; font-weight: 700; background: transparent;")
        left.addWidget(name)
        task = QLabel("Object Detection" if s.get("task") == "detection" else "Image Classification")
        task.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; background: transparent;")
        left.addWidget(task)

        # status badge
        badge_row = QHBoxLayout(); badge_row.setSpacing(8)
        status = s.get("status", "running")
        dot = QLabel("●")
        if status == "complete":
            dot.setStyleSheet(f"color: {C_GREEN}; font-size: 14px; background: transparent;")
            txt = QLabel("Complete ✅"); col = C_GREEN
        elif status == "timed_out":
            dot.setStyleSheet(f"color: {C_AMBER}; font-size: 14px; background: transparent;")
            txt = QLabel("Timed out ⚠️"); col = C_AMBER
        else:
            dot.setStyleSheet(f"color: {C_BLUE}; font-size: 14px; background: transparent;")
            txt = QLabel("Training in progress…"); col = C_BLUE
            self._running_dots.append(dot)   # animate this one
        txt.setStyleSheet(f"color: {col}; font-size: 12px; font-weight: 700; background: transparent;")
        badge_row.addWidget(dot, 0); badge_row.addWidget(txt, 0); badge_row.addStretch(1)
        left.addLayout(badge_row)

        # elapsed since launch
        launched = _parse_ts(s.get("launched_at", ""))
        end = _parse_ts(s.get("completed_at", "")) if status != "running" else time.time()
        elapsed = (end - launched) if (launched and end) else 0
        el = QLabel("Elapsed: " + self._fmt_elapsed(elapsed))
        el.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; background: transparent;")
        left.addWidget(el)
        h.addLayout(left, 1)

        # buttons
        btns = QVBoxLayout(); btns.setSpacing(6)
        b_open = QPushButton("Open Notebook")
        b_open.setCursor(Qt.PointingHandCursor)
        b_open.setStyleSheet(
            f"QPushButton {{ background: {C_PANEL2}; color: {C_TEXT}; border: 1px solid {C_BORDER};"
            f"border-radius: 7px; padding: 6px 12px; font-size: 12px; }}"
            f"QPushButton:hover {{ border: 1px solid {C_TEAL}; }}")
        url = s.get("notebook_url", "")
        b_open.clicked.connect(lambda _c, u=url: self._open_notebook(u))
        b_open.setEnabled(bool(url))
        b_hist = QPushButton("View in History")
        b_hist.setCursor(Qt.PointingHandCursor)
        b_hist.setStyleSheet(
            f"QPushButton {{ background: {C_PANEL2}; color: {C_TEXT}; border: 1px solid {C_BORDER};"
            f"border-radius: 7px; padding: 6px 12px; font-size: 12px; }}"
            f"QPushButton:hover {{ border: 1px solid {C_TEAL}; }}")
        b_hist.clicked.connect(lambda: self.navigate(IDX_HISTORY))
        btns.addWidget(b_open); btns.addWidget(b_hist)
        h.addLayout(btns, 0)
        return card

    def _open_notebook(self, url):
        if url:
            import webbrowser
            webbrowser.open(url)
