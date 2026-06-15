"""
DR. ROY — DATA TRAINING & REPORTING
Main application window.

IMPORTANT: brain.aiboot is imported FIRST (before PyQt5) so the AI engine
(PyTorch + YOLO) initialises before Qt loads its DLLs. See brain/aiboot.py.
"""

# --- AI engine must load before any PyQt5 import (Windows DLL order) ---
try:
    import brain.aiboot  # noqa: F401  loads torch + ultralytics safely first
    _AI_READY = True
except Exception as _e:                      # app still opens even if AI missing
    _AI_READY = False
    _AI_ERROR = str(_e)

import sys
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QStackedWidget, QFrame, QSizePolicy,
)

# ----------------------------------------------------------------------
# Colour palette
# ----------------------------------------------------------------------
C_BG        = "#0d1117"   # very dark navy — main background
C_PANEL     = "#161b27"   # sidebar / top bar
C_PANEL2    = "#1c2333"   # cards / inner panels
C_BORDER    = "#2a3045"   # subtle borders
C_TEXT      = "#e6edf3"   # near-white text
C_TEXT_DIM  = "#8b949e"   # muted / secondary text
C_TEAL      = "#238f7a"   # primary accent / active buttons
C_TEALH     = "#2aad93"   # teal hover
C_INACTIVE  = "#3a4255"   # disabled / greyed out

APP_VERSION = "v1.0"

# ----------------------------------------------------------------------
# Navigation definition: (section header, [(emoji+label, page title), ...])
# ----------------------------------------------------------------------
NAV = [
    ("WORKSPACE", [
        ("🏠  Home",            "Home"),
        ("🤖  Train Model",     "Train Model"),
        ("☁️  Train on Colab",  "Train on Colab"),
        ("📜  Training History", "Training History"),
        ("🗂️  Datasets",         "Datasets"),
        ("🧠  Models",           "Models"),
    ]),
    ("ANALYSIS", [
        ("🔬  Predict / Analyze", "Predict / Analyze"),
        ("📄  Reports & Export",  "Reports & Export"),
    ]),
    ("PREFERENCES", [
        ("⚙️  Settings", "Settings"),
    ]),
    ("ABOUT", [
        ("📋  Changelog (v1.0)", "Changelog"),
    ]),
]

# Short descriptions shown on each placeholder page
PAGE_BLURB = {
    "Home":              "Welcome to your AI pathology workspace. Pick a tool from the left to begin.",
    "Train Model":       "Train a YOLO model on your annotated slide patches.",
    "Train on Colab":    "Offload heavy training to free Google Colab GPUs, then download the trained model.",
    "Training History":  "Review every training run — accuracy, epochs, dataset and duration.",
    "Datasets":          "Add, scan and manage your image datasets.",
    "Models":            "Your library of trained models — load, rename or delete.",
    "Predict / Analyze": "Run a model on a slide image and see detected findings.",
    "Reports & Export":  "Turn prediction results into clinical PDF reports.",
    "Settings":          "Lab name, pathologist name, defaults and preferences.",
    "Changelog":         "What's new in Dr. Roy DT&R.",
}


class NavButton(QPushButton):
    """A sidebar navigation button that can be active (teal) or idle."""

    def __init__(self, text):
        super().__init__(text)
        self.setCursor(Qt.PointingHandCursor)
        self.setCheckable(True)
        self.setMinimumHeight(40)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._apply_style(False)

    def set_active(self, active):
        self.setChecked(active)
        self._apply_style(active)

    def _apply_style(self, active):
        if active:
            bg, fg, weight = C_TEAL, "#ffffff", "600"
            hover = C_TEALH
        else:
            bg, fg, weight = "transparent", C_TEXT_DIM, "500"
            hover = C_PANEL2
        self.setStyleSheet(f"""
            QPushButton {{
                background: {bg};
                color: {fg};
                border: none;
                border-radius: 8px;
                padding: 8px 14px;
                text-align: left;
                font-size: 14px;
                font-weight: {weight};
            }}
            QPushButton:hover {{ background: {hover}; color: {C_TEXT}; }}
        """)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dr. Roy — Data Training & Reporting")
        self.resize(1280, 800)
        self.setMinimumSize(1000, 640)
        self.setStyleSheet(f"background: {C_BG};")

        self.nav_buttons = []     # list of (NavButton, stack_index)
        self.stack = QStackedWidget()

        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_top_bar())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        body.addWidget(self._build_sidebar())
        body.addWidget(self._build_content(), 1)
        body_wrap = QWidget()
        body_wrap.setLayout(body)
        outer.addWidget(body_wrap, 1)

        outer.addWidget(self._build_status_bar())

        # Live clock
        self._tick()
        self.clock = QTimer(self)
        self.clock.timeout.connect(self._tick)
        self.clock.start(1000)

        # Start on Home
        self._go(0)

        # Check / download the AI base model in the background
        self._model_thread = None
        self._start_model_check()

    # ---- Train with a dataset from the library -----------------------
    def _train_with_dataset(self, entry):
        page = self.pages.get("Train Model")
        if "Train Model" in self.page_idx:
            self._go(self.page_idx["Train Model"])
        if page is not None:
            page.load_dataset_path(entry.get("path", ""), task=entry.get("task"))

    # ---- Re-run a training run from history --------------------------
    def _rerun_training(self, entry):
        page = self.pages.get("Train Model")
        if "Train Model" in self.page_idx:
            self._go(self.page_idx["Train Model"])
        if page is not None:
            page.prefill_and_run(entry)

    # ---- AI base-model check on startup ------------------------------
    def _start_model_check(self):
        """Make sure the base AI model is present; download it the first time."""
        self.lbl_model.setText("AI Model: Checking…")
        if not _AI_READY:
            self.lbl_model.setText("AI Model: engine not loaded")
            return
        try:
            from brain.trainer import BaseModelDownloader
        except Exception:
            self.lbl_model.setText("AI Model: unavailable")
            return
        self._model_thread = BaseModelDownloader("yolov8n.pt")
        self._model_thread.status.connect(lambda m: self.lbl_model.setText("AI Model: " + m))
        self._model_thread.ready.connect(lambda _p: self.lbl_model.setText("AI Model: Ready"))
        self._model_thread.failed.connect(
            lambda _e: self.lbl_model.setText("AI Model: base download pending (will fetch on first training)"))
        self._model_thread.start()

    # ---- Top bar -----------------------------------------------------
    def _build_top_bar(self):
        bar = QFrame()
        bar.setFixedHeight(72)
        bar.setStyleSheet(f"background: {C_PANEL}; border-bottom: 1px solid {C_BORDER};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 10, 24, 10)

        left = QVBoxLayout()
        left.setSpacing(0)
        title = QLabel("🔬 Dr. Roy DT&R")
        title.setStyleSheet(f"color: {C_TEXT}; font-size: 20px; font-weight: 700;")
        subtitle = QLabel("AI-Powered Pathology Platform")
        subtitle.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")
        left.addWidget(title)
        left.addWidget(subtitle)
        lay.addLayout(left)

        lay.addStretch(1)

        self.lbl_datetime = QLabel("")
        self.lbl_datetime.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_datetime.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px;")
        lay.addWidget(self.lbl_datetime)
        return bar

    # ---- Sidebar -----------------------------------------------------
    def _build_sidebar(self):
        side = QFrame()
        side.setFixedWidth(240)
        side.setStyleSheet(f"background: {C_PANEL}; border-right: 1px solid {C_BORDER};")
        lay = QVBoxLayout(side)
        lay.setContentsMargins(12, 16, 12, 16)
        lay.setSpacing(4)

        idx = 0
        for section, items in NAV:
            hdr = QLabel(section)
            hdr.setStyleSheet(
                f"color: {C_TEXT_DIM}; font-size: 11px; font-weight: 700;"
                f"letter-spacing: 1px; padding: 12px 8px 4px 8px;")
            lay.addWidget(hdr)
            for label, _title in items:
                btn = NavButton(label)
                stack_index = idx
                btn.clicked.connect(lambda _checked, i=stack_index: self._go(i))
                lay.addWidget(btn)
                self.nav_buttons.append((btn, stack_index))
                idx += 1

        lay.addStretch(1)
        return side

    # ---- Content (stacked pages: real where built, placeholder elsewhere) ----
    def _build_content(self):
        from screens.colab_page import TrainOnColabPage
        from screens.home_page import HomePage
        from screens.train_page import TrainModelPage
        from screens.history_page import TrainingHistoryPage
        from screens.datasets_page import DatasetLibraryPage
        from screens.models_page import ModelLibraryPage
        from screens.predict_page import PredictPage
        from screens.reports_page import ReportsPage
        from screens.settings_page import SettingsPage
        from screens.changelog_page import ChangelogPage
        self.pages = {}
        self.page_idx = {}
        idx = 0
        for _section, items in NAV:
            for _label, title in items:
                if title == "Home":
                    page = HomePage(navigate=self._go)
                elif title == "Train Model":
                    page = TrainModelPage(status_callback=self._update_model_status)
                elif title == "Train on Colab":
                    page = TrainOnColabPage(status_callback=self._update_model_status)
                elif title == "Training History":
                    page = TrainingHistoryPage(navigate=self._go, rerun_callback=self._rerun_training)
                elif title == "Datasets":
                    page = DatasetLibraryPage(navigate=self._go, train_callback=self._train_with_dataset)
                elif title == "Models":
                    page = ModelLibraryPage(status_callback=self._update_model_status)
                elif title == "Predict / Analyze":
                    page = PredictPage(status_callback=self._update_model_status)
                elif title == "Reports & Export":
                    page = ReportsPage(status_callback=self._update_model_status)
                elif title == "Settings":
                    page = SettingsPage(status_callback=self._update_model_status)
                elif title == "Changelog":
                    page = ChangelogPage()
                else:
                    page = self._placeholder_page(title)
                self.stack.addWidget(page)
                self.pages[title] = page
                self.page_idx[title] = idx
                idx += 1
        wrap = QFrame()
        wrap.setStyleSheet(f"background: {C_BG};")
        wl = QVBoxLayout(wrap)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.addWidget(self.stack)
        return wrap

    def _placeholder_page(self, title):
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(40, 36, 40, 36)
        lay.setSpacing(14)

        h = QLabel(title)
        h.setStyleSheet(f"color: {C_TEXT}; font-size: 28px; font-weight: 700;")
        lay.addWidget(h)

        blurb = QLabel(PAGE_BLURB.get(title, ""))
        blurb.setWordWrap(True)
        blurb.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 15px;")
        lay.addWidget(blurb)

        card = QFrame()
        card.setStyleSheet(
            f"background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 12px;")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(28, 28, 28, 28)
        tag = QLabel(f"🚧  “{title}” page — coming soon")
        tag.setStyleSheet(f"color: {C_TEXT}; font-size: 16px; font-weight: 600;")
        note = QLabel("This screen is a placeholder. The full tool will be built here next.")
        note.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px;")
        cl.addWidget(tag)
        cl.addWidget(note)
        lay.addWidget(card)

        lay.addStretch(1)
        return page

    # ---- Status bar --------------------------------------------------
    def _build_status_bar(self):
        bar = QFrame()
        bar.setFixedHeight(34)
        bar.setStyleSheet(f"background: {C_PANEL}; border-top: 1px solid {C_BORDER};")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(18, 0, 18, 0)

        self.lbl_status = QLabel("Status: Ready")
        self.lbl_status.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")
        lay.addWidget(self.lbl_status)

        lay.addStretch(1)
        self.lbl_model = QLabel("AI Model: Not Loaded")
        self.lbl_model.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")
        lay.addWidget(self.lbl_model)
        lay.addStretch(1)

        right = QLabel(f"Dr. Roy DT&R {APP_VERSION}")
        right.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")
        lay.addWidget(right)
        return bar

    # ---- Behaviour ---------------------------------------------------
    def _go(self, index):
        self.stack.setCurrentIndex(index)
        for btn, i in self.nav_buttons:
            btn.set_active(i == index)

    def _tick(self):
        now = datetime.now()
        self.lbl_datetime.setText(now.strftime("%a, %d %b %Y   %I:%M:%S %p"))

    def _update_model_status(self, msg):
        self.lbl_model.setText(msg)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    if not _AI_READY:
        win.lbl_status.setText("Status: Ready (AI engine not loaded)")
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
