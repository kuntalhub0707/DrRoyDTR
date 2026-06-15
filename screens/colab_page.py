"""
Train on Google Colab page.

This step builds the page LAYOUT and FORM STATE only.
The 'Connect Google Drive' button, auto-generate-notebook logic, and the
monitoring/download flow are wired up in later Colab steps.

The big 'Launch Colab Training' button stays greyed out until the three
essentials are provided: a Task, a Dataset Source, and the Google account email.
Call .get_config() to read the whole form back as a dict (used by later steps).
"""

from PyQt5.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QLineEdit, QComboBox, QSpinBox, QCheckBox, QScrollArea,
    QFileDialog, QSizePolicy, QProgressBar, QMessageBox,
)


class _DriveConnectWorker(QThread):
    ok = pyqtSignal(str)          # account name
    need_setup = pyqtSignal()
    failed = pyqtSignal(str)

    def run(self):
        import brain.gdrive as gd
        try:
            if not gd.is_configured():
                self.need_setup.emit(); return
            self.ok.emit(gd.connect())
        except Exception as e:
            if str(e) == "NOT_CONFIGURED":
                self.need_setup.emit()
            else:
                self.failed.emit(str(e))


class _DriveUploadWorker(QThread):
    progress = pyqtSignal(int, int, str)   # done, total, filename
    done = pyqtSignal(str, str)            # folder_id, folder_name
    failed = pyqtSignal(str)

    def __init__(self, folder):
        super().__init__(); self.folder = folder; self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        import brain.gdrive as gd
        try:
            fid, name = gd.upload_folder(
                self.folder,
                on_progress=lambda d, t, f: self.progress.emit(d, t, f),
                should_stop=lambda: self._stop)
            self.done.emit(fid, name)
        except Exception as e:
            self.failed.emit(str(e))


class _DriveValidateWorker(QThread):
    ok = pyqtSignal(str, str, int)   # folder_id, name, image_count
    failed = pyqtSignal(str)

    def __init__(self, link):
        super().__init__(); self.link = link

    def run(self):
        import brain.gdrive as gd
        try:
            fid, name, cnt = gd.validate_folder(self.link)
            self.ok.emit(fid, name, cnt)
        except Exception as e:
            self.failed.emit(str(e))


class _LaunchWorker(QThread):
    """Build the Colab notebook and upload it to Drive. Emits the new file ID."""
    done = pyqtSignal(str)   # notebook file id
    failed = pyqtSignal(str)

    def __init__(self, cfg, nb_name):
        super().__init__(); self.cfg = cfg; self.nb_name = nb_name

    def run(self):
        import brain.notebook as nbmod
        import brain.gdrive as gd
        try:
            nb_json = nbmod.build_notebook(self.cfg)
            file_id = gd.upload_file_content(self.nb_name, nb_json.encode("utf-8"),
                                             mime="application/json")
            self.done.emit(file_id)
        except Exception as e:
            self.failed.emit(str(e))


class _ModelPollWorker(QThread):
    """Check Drive once for the finished model. Emits its metadata if present."""
    found = pyqtSignal(dict)    # {file_id, name, size, modified, metric, task}
    not_yet = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, run_name, date):
        super().__init__(); self.run_name = run_name; self.date = date

    def run(self):
        import brain.gdrive as gd
        try:
            meta = gd.find_model_result(self.run_name, self.date)
            if not meta:
                self.not_yet.emit()
            else:
                self.found.emit(meta)
        except Exception as e:
            self.failed.emit(str(e))


class _ModelDownloadWorker(QThread):
    """Download the trained model from Drive with progress."""
    progress = pyqtSignal(int)
    done = pyqtSignal(str)      # local path
    failed = pyqtSignal(str)

    def __init__(self, file_id, dest):
        super().__init__(); self.file_id = file_id; self.dest = dest

    def run(self):
        import brain.gdrive as gd
        try:
            gd.download_file(self.file_id, self.dest, on_progress=lambda p: self.progress.emit(p))
            self.done.emit(self.dest)
        except Exception as e:
            self.failed.emit(str(e))

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
C_BLUE      = "#3b82f6"   # selected-card highlight
C_BLUE_BG   = "#16233f"   # subtle blue tint behind a selected card
C_GREEN     = "#3fb950"   # success / connected

# Shared style for text inputs / combos / spinboxes
_FIELD_QSS = f"""
    QLineEdit, QComboBox, QSpinBox {{
        background: {C_BG};
        color: {C_TEXT};
        border: 1px solid {C_BORDER};
        border-radius: 7px;
        padding: 8px 10px;
        font-size: 14px;
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus {{
        border: 1px solid {C_TEAL};
    }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox QAbstractItemView {{
        background: {C_PANEL2};
        color: {C_TEXT};
        selection-background-color: {C_TEAL};
        border: 1px solid {C_BORDER};
        outline: none;
    }}
    QSpinBox::up-button, QSpinBox::down-button {{ width: 18px; }}
"""


class SelectableCard(QFrame):
    """A click-to-select card. Selected = blue border + blue tint."""

    clicked = pyqtSignal()

    def __init__(self, title, desc):
        super().__init__()
        self.setCursor(Qt.PointingHandCursor)
        self._selected = False
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(6)

        self._title = QLabel(title)
        self._title.setStyleSheet(f"color: {C_TEXT}; font-size: 16px; font-weight: 700; background: transparent;")
        d = QLabel(desc)
        d.setWordWrap(True)
        d.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px; background: transparent;")
        lay.addWidget(self._title)
        lay.addWidget(d)
        lay.addStretch(1)

        self._restyle()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def set_selected(self, value):
        self._selected = bool(value)
        self._restyle()

    def is_selected(self):
        return self._selected

    def _restyle(self):
        if self._selected:
            border, bg = C_BLUE, C_BLUE_BG
        else:
            border, bg = C_BORDER, C_PANEL2
        self.setStyleSheet(
            f"SelectableCard {{ background: {bg}; border: 2px solid {border}; border-radius: 12px; }}")


class RadioSourceCard(QFrame):
    """
    A larger radio-button style card used for the dataset source choice.
    Holds a radio dot + title + description, plus an inner 'body' area where
    page-specific controls (Browse button, text box) are placed.
    """

    clicked = pyqtSignal()

    def __init__(self, title, desc):
        super().__init__()
        self.setCursor(Qt.PointingHandCursor)
        self._selected = False
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        outer.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(12)
        self._dot = QLabel("○")
        self._dot.setFixedWidth(20)
        self._dot.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 18px; background: transparent;")
        head.addWidget(self._dot, 0, Qt.AlignTop)

        textcol = QVBoxLayout()
        textcol.setSpacing(4)
        self._title = QLabel(title)
        self._title.setStyleSheet(f"color: {C_TEXT}; font-size: 16px; font-weight: 700; background: transparent;")
        d = QLabel(desc)
        d.setWordWrap(True)
        d.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px; background: transparent;")
        textcol.addWidget(self._title)
        textcol.addWidget(d)
        head.addLayout(textcol, 1)
        outer.addLayout(head)

        # Body area for inputs (Browse / textbox)
        self.body = QWidget()
        self.body.setStyleSheet("background: transparent;")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(32, 0, 0, 0)
        self.body_layout.setSpacing(8)
        outer.addWidget(self.body)

        self._restyle()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def set_selected(self, value):
        self._selected = bool(value)
        self._restyle()

    def is_selected(self):
        return self._selected

    def _restyle(self):
        if self._selected:
            border, bg, dot = C_BLUE, C_BLUE_BG, C_BLUE
            self._dot.setText("●")
        else:
            border, bg, dot = C_BORDER, C_PANEL2, C_TEXT_DIM
            self._dot.setText("○")
        self._dot.setStyleSheet(f"color: {dot}; font-size: 18px; background: transparent;")
        self.setStyleSheet(
            f"RadioSourceCard {{ background: {bg}; border: 2px solid {border}; border-radius: 12px; }}")


class TrainOnColabPage(QWidget):
    def __init__(self, status_callback=None):
        super().__init__()
        self.status_callback = status_callback
        self.selected_folder = ""

        self.setStyleSheet(f"background: {C_BG};")

        # Scrollable so the long form fits any window height
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        page = QWidget()
        page.setStyleSheet("background: transparent;")
        root = QVBoxLayout(page)
        root.setContentsMargins(40, 32, 40, 40)
        root.setSpacing(8)

        # ---- Title + description ----
        title = QLabel("Train on Google Colab —  Use Cloud GPU Power")
        title.setStyleSheet(f"color: {C_TEXT}; font-size: 26px; font-weight: 800;")
        root.addWidget(title)

        desc = QLabel(
            "Your computer does not need a graphics card. Google Colab provides a free "
            "powerful GPU in the cloud. You upload your images, press one button, and the "
            "trained model downloads back into your app automatically.")
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 14px;")
        root.addWidget(desc)
        root.addSpacing(10)

        # ---- Section 1 — Choose Your Task ----
        root.addWidget(self._section_heading("Step 1 — What do you want to train?"))
        self.card_detection = SelectableCard(
            "Object Detection", "Count and locate cells, mitosis, parasites")
        self.card_classification = SelectableCard(
            "Image Classification", "Label whole images by category or grade")
        self.card_detection.clicked.connect(lambda: self._select_task("detection"))
        self.card_classification.clicked.connect(lambda: self._select_task("classification"))
        self.selected_task = None

        task_row = QHBoxLayout()
        task_row.setSpacing(16)
        task_row.addWidget(self.card_detection)
        task_row.addWidget(self.card_classification)
        root.addLayout(task_row)
        root.addSpacing(18)

        # ---- Section 2 — Choose Dataset Source ----
        root.addWidget(self._section_heading("Step 2 — Where are your images?"))
        self.selected_source = None
        self.drive_folder_id = ""        # set after upload or validation
        self.drive_folder_name = ""
        self.drive_connected = False
        self._conn_worker = None
        self._upload_worker = None
        self._validate_worker = None
        self._launch_worker = None

        # Connect Google Drive row
        conn_row = QHBoxLayout()
        conn_row.setSpacing(10)
        self.btn_connect = QPushButton("🔗  Connect Google Drive")
        self.btn_connect.setCursor(Qt.PointingHandCursor)
        self.btn_connect.setMinimumHeight(40)
        self.btn_connect.setStyleSheet(
            f"QPushButton {{ background: {C_BG}; color: {C_TEXT}; border: 1px solid {C_BORDER};"
            f"border-radius: 8px; padding: 8px 16px; font-size: 14px; font-weight: 600; }}"
            f"QPushButton:hover {{ border: 1px solid {C_TEAL}; }}")
        self.btn_connect.clicked.connect(self._connect_drive)
        self.lbl_conn = QLabel("Not connected")
        self.lbl_conn.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px; background: transparent;")
        conn_row.addWidget(self.btn_connect, 0)
        conn_row.addWidget(self.lbl_conn, 1)
        root.addLayout(conn_row)
        root.addSpacing(10)

        # Option A — upload from computer
        self.src_upload = RadioSourceCard(
            "Upload from my Computer",
            "Select a folder from your computer. The app will upload it to Google Drive automatically.")
        browse_row = QHBoxLayout()
        browse_row.setSpacing(10)
        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.setCursor(Qt.PointingHandCursor)
        self.btn_browse.setStyleSheet(self._secondary_btn_qss())
        self.btn_browse.clicked.connect(self._on_browse)
        self.lbl_folder = QLabel("No folder selected")
        self.lbl_folder.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px; background: transparent;")
        browse_row.addWidget(self.btn_browse, 0)
        browse_row.addWidget(self.lbl_folder, 1)
        self.src_upload.body_layout.addLayout(browse_row)

        # found-count + upload controls (hidden until a folder is chosen)
        self.lbl_found = QLabel("")
        self.lbl_found.setVisible(False)
        self.lbl_found.setStyleSheet(f"color: {C_TEXT}; font-size: 13px; font-weight: 600; background: transparent;")
        self.src_upload.body_layout.addWidget(self.lbl_found)

        self.btn_upload = QPushButton("⬆  Upload to Drive")
        self.btn_upload.setVisible(False)
        self.btn_upload.setCursor(Qt.PointingHandCursor)
        self.btn_upload.setStyleSheet(
            f"QPushButton {{ background: {C_TEAL}; color: #fff; border: none; border-radius: 7px;"
            f"padding: 8px 16px; font-size: 13px; font-weight: 700; }}"
            f"QPushButton:hover {{ background: {C_TEALH}; }}")
        self.btn_upload.clicked.connect(self._upload_to_drive)
        self.src_upload.body_layout.addWidget(self.btn_upload)

        self.upload_bar = QProgressBar()
        self.upload_bar.setVisible(False)
        self.upload_bar.setFixedHeight(18)
        self.upload_bar.setStyleSheet(
            f"QProgressBar {{ background: {C_BG}; border: 1px solid {C_BORDER}; border-radius: 6px;"
            f"text-align: center; color: {C_TEXT}; font-size: 11px; }}"
            f"QProgressBar::chunk {{ background: {C_TEAL}; border-radius: 5px; }}")
        self.src_upload.body_layout.addWidget(self.upload_bar)

        self.lbl_upload_status = QLabel("")
        self.lbl_upload_status.setVisible(False)
        self.lbl_upload_status.setWordWrap(True)
        self.lbl_upload_status.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; background: transparent;")
        self.src_upload.body_layout.addWidget(self.lbl_upload_status)

        self.src_upload.clicked.connect(lambda: self._select_source("upload"))

        # Option B — already on Google Drive
        self.src_drive = RadioSourceCard(
            "Already on Google Drive",
            "Paste the Google Drive folder link or ID.")
        self.in_drive = QLineEdit()
        self.in_drive.setPlaceholderText("Google Drive Folder Link or ID")
        self.in_drive.setStyleSheet(_FIELD_QSS)
        self.in_drive.textChanged.connect(self._on_drive_link_changed)
        self.src_drive.body_layout.addWidget(self.in_drive)
        self.lbl_drive_validate = QLabel("")
        self.lbl_drive_validate.setVisible(False)
        self.lbl_drive_validate.setWordWrap(True)
        self.lbl_drive_validate.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; background: transparent;")
        self.src_drive.body_layout.addWidget(self.lbl_drive_validate)
        self.src_drive.clicked.connect(lambda: self._select_source("drive"))

        # debounce timer for Drive-link validation
        self._validate_timer = QTimer(self)
        self._validate_timer.setSingleShot(True)
        self._validate_timer.timeout.connect(self._validate_drive_link)

        root.addWidget(self.src_upload)
        root.addSpacing(12)
        root.addWidget(self.src_drive)
        root.addSpacing(18)

        # ---- Section 3 — Colab Settings ----
        root.addWidget(self._section_heading("Step 3 — Configure your training"))
        settings_card = QFrame()
        settings_card.setStyleSheet(
            f"QFrame {{ background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 12px; }}")
        grid = QGridLayout(settings_card)
        grid.setContentsMargins(22, 20, 22, 20)
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(16)
        grid.setColumnStretch(1, 1)

        r = 0
        # Model Size
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["Nano", "Small", "Medium", "Large"])
        self.cmb_model.setStyleSheet(_FIELD_QSS)
        grid.addWidget(self._field_label("Model Size"), r, 0)
        grid.addWidget(self.cmb_model, r, 1); r += 1

        # Training Rounds
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 2000)
        self.spin_epochs.setValue(50)
        self.spin_epochs.setStyleSheet(_FIELD_QSS)
        grid.addWidget(self._field_label("Training Rounds"), r, 0)
        grid.addWidget(self.spin_epochs, r, 1); r += 1

        # Image Size
        self.cmb_imgsize = QComboBox()
        self.cmb_imgsize.addItems(["320", "416", "640", "1024"])
        self.cmb_imgsize.setCurrentText("640")
        self.cmb_imgsize.setStyleSheet(_FIELD_QSS)
        grid.addWidget(self._field_label("Image Size"), r, 0)
        grid.addWidget(self.cmb_imgsize, r, 1); r += 1

        # Google Account Email
        self.in_email = QLineEdit()
        self.in_email.setPlaceholderText("you@gmail.com")
        self.in_email.setStyleSheet(_FIELD_QSS)
        self.in_email.textChanged.connect(self._validate)
        grid.addWidget(self._field_label("Google Account Email"), r, 0)
        grid.addWidget(self.in_email, r, 1); r += 1

        # Colab Notebook
        self.cmb_notebook = QComboBox()
        self.cmb_notebook.addItems([
            "Auto-generate (recommended)",
            "Use my existing notebook (paste link)",
        ])
        self.cmb_notebook.setStyleSheet(_FIELD_QSS)
        self.cmb_notebook.currentIndexChanged.connect(self._on_notebook_changed)
        grid.addWidget(self._field_label("Colab Notebook"), r, 0)
        grid.addWidget(self.cmb_notebook, r, 1); r += 1

        # Existing-notebook URL (hidden unless option 2 chosen)
        self.in_notebook_url = QLineEdit()
        self.in_notebook_url.setPlaceholderText("https://colab.research.google.com/…")
        self.in_notebook_url.setStyleSheet(_FIELD_QSS)
        self.lbl_notebook_url = self._field_label("Notebook Link")
        self.lbl_notebook_url.setVisible(False)
        self.in_notebook_url.setVisible(False)
        grid.addWidget(self.lbl_notebook_url, r, 0)
        grid.addWidget(self.in_notebook_url, r, 1); r += 1

        root.addWidget(settings_card)
        root.addSpacing(14)

        # After Training options
        root.addWidget(self._section_heading("After Training", small=True))
        self.chk_autodownload = QCheckBox("Auto-download model into my app")
        self.chk_manuallink = QCheckBox("Show manual download link")
        for chk in (self.chk_autodownload, self.chk_manuallink):
            chk.setChecked(True)
            chk.setCursor(Qt.PointingHandCursor)
            chk.setStyleSheet(self._checkbox_qss())
            root.addWidget(chk)
        root.addSpacing(22)

        # ---- Launch button ----
        self.btn_launch = QPushButton("🚀  Launch Colab Training")
        self.btn_launch.setMinimumHeight(52)
        self.btn_launch.setCursor(Qt.PointingHandCursor)
        self.btn_launch.clicked.connect(self._on_launch)
        root.addWidget(self.btn_launch)

        self.lbl_hint = QLabel("Choose a task, a dataset source, and enter your Google email to enable launch.")
        self.lbl_hint.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")
        root.addWidget(self.lbl_hint)

        # ---- Monitoring panel (hidden until launch) ----
        self.launch_panel = self._build_monitor_panel()
        self.launch_panel.setVisible(False)
        root.addWidget(self.launch_panel)

        # timers + workers for monitoring
        self._blink_on = True
        self._blink_timer = QTimer(self); self._blink_timer.timeout.connect(self._blink)
        self._poll_timer = QTimer(self); self._poll_timer.timeout.connect(self._poll_for_model)
        self._elapsed_timer = QTimer(self); self._elapsed_timer.timeout.connect(self._tick_elapsed)
        self._poll_worker = None
        self._dl_worker = None
        self._monitor_start = None
        self._warned_timeout = False
        self._found_meta = None
        self._notebook_url = ""

        root.addStretch(1)

        scroll.setWidget(page)
        wrap = QVBoxLayout(self)
        wrap.setContentsMargins(0, 0, 0, 0)
        wrap.addWidget(scroll)

        self._validate()  # set initial (disabled) button state

    # ------------------------------------------------------------------
    # Small UI helpers
    # ------------------------------------------------------------------
    def _section_heading(self, text, small=False):
        lbl = QLabel(text)
        size = 15 if small else 18
        lbl.setStyleSheet(
            f"color: {C_TEXT}; font-size: {size}px; font-weight: 700; padding: 4px 0 8px 0;")
        return lbl

    def _field_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px; font-weight: 600; background: transparent;")
        return lbl

    def _secondary_btn_qss(self):
        return f"""
            QPushButton {{
                background: {C_BG}; color: {C_TEXT};
                border: 1px solid {C_BORDER}; border-radius: 7px;
                padding: 8px 16px; font-size: 13px; font-weight: 600;
            }}
            QPushButton:hover {{ border: 1px solid {C_TEAL}; }}
        """

    def _checkbox_qss(self):
        return f"""
            QCheckBox {{ color: {C_TEXT}; font-size: 14px; spacing: 10px; }}
            QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 4px;
                border: 1px solid {C_BORDER}; background: {C_BG}; }}
            QCheckBox::indicator:checked {{ background: {C_TEAL}; border: 1px solid {C_TEAL}; }}
        """

    # ------------------------------------------------------------------
    # Selection logic
    # ------------------------------------------------------------------
    def _select_task(self, task):
        self.selected_task = task
        self.card_detection.set_selected(task == "detection")
        self.card_classification.set_selected(task == "classification")
        self._validate()

    def _select_source(self, source):
        self.selected_source = source
        self.src_upload.set_selected(source == "upload")
        self.src_drive.set_selected(source == "drive")
        self._validate()

    def _on_browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select image folder")
        if folder:
            self.selected_folder = folder
            self.lbl_folder.setText(folder)
            self.lbl_folder.setStyleSheet(f"color: {C_TEXT}; font-size: 13px; background: transparent;")
            self._select_source("upload")   # picking a folder implies this source
            # count images + reveal upload controls
            import brain.gdrive as gd
            n_imgs, n_subs = gd.count_local_images(folder)
            self.lbl_found.setText(f"Found {n_imgs} images in {n_subs} sub-folders")
            self.lbl_found.setVisible(True)
            self.btn_upload.setVisible(n_imgs > 0)
            self.upload_bar.setVisible(False)
            self.lbl_upload_status.setVisible(False)
            self.drive_folder_id = ""   # new folder chosen — clear any previous upload id
        self._validate()

    # ---- Google Drive: connect ----
    def _connect_drive(self):
        import brain.gdrive as gd
        if gd.is_connected():
            name = gd.saved_account_name() or "your Google account"
            self._mark_connected(name)
            return
        if not gd.is_configured():
            # no credential file yet → go straight to the guided setup dialog
            self._on_drive_need_setup()
            return
        self.btn_connect.setEnabled(False)
        self.btn_connect.setText("Opening browser…")
        self.lbl_conn.setText("Waiting for Google sign-in in your browser…")
        self.lbl_conn.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px; background: transparent;")
        self._conn_worker = _DriveConnectWorker()
        self._conn_worker.ok.connect(self._on_drive_connected)
        self._conn_worker.need_setup.connect(self._on_drive_need_setup)
        self._conn_worker.failed.connect(self._on_drive_failed)
        self._conn_worker.start()

    def _mark_connected(self, name):
        self.drive_connected = True
        self.btn_connect.setEnabled(True)
        self.btn_connect.setText("🔗  Reconnect Google Drive")
        self.lbl_conn.setText(f"✓ Connected: {name}")
        self.lbl_conn.setStyleSheet(f"color: {C_GREEN}; font-size: 13px; font-weight: 700; background: transparent;")

    def _on_drive_connected(self, name):
        self._mark_connected(name)
        if self.status_callback:
            self.status_callback(f"Google Drive connected: {name}")

    def _on_drive_need_setup(self):
        self.btn_connect.setEnabled(True)
        self.btn_connect.setText("🔗  Connect Google Drive")
        self.lbl_conn.setText("One-time Google credential file needed — click Connect for setup")
        self.lbl_conn.setStyleSheet(f"color: #e5c07b; font-size: 13px; background: transparent;")
        self._show_setup_dialog()

    def _show_setup_dialog(self):
        from PyQt5.QtWidgets import QDialog
        import brain.gdrive as gd
        dlg = QDialog(self)
        dlg.setWindowTitle("Connect Google Drive — one-time setup")
        dlg.setMinimumWidth(560)
        dlg.setStyleSheet(f"QDialog {{ background: {C_BG}; }}")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(24, 22, 24, 22)
        lay.setSpacing(14)

        head = QLabel("One-time Google setup")
        head.setStyleSheet(f"color: {C_TEXT}; font-size: 18px; font-weight: 800;")
        lay.addWidget(head)

        steps = QLabel(
            "Google needs a small, free credential file before the app can sign you in. "
            "You only do this once.<br><br>"
            "1. Click <b>Open Google Cloud Console</b> below<br>"
            "2. Create a project → enable the <b>Google Drive API</b><br>"
            "3. <b>OAuth consent screen</b> → External → add your own Gmail as a test user<br>"
            "4. <b>Credentials → Create Credentials → OAuth client ID → Desktop app</b><br>"
            "5. <b>Download the JSON</b>, then come back and click "
            "<b>I downloaded the file — Select it</b> (the app renames &amp; places it for you)")
        steps.setTextFormat(Qt.RichText)
        steps.setWordWrap(True)
        steps.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px; line-height: 150%;")
        lay.addWidget(steps)

        status = QLabel("")
        status.setWordWrap(True)
        status.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")
        lay.addWidget(status)

        def open_console():
            import webbrowser
            webbrowser.open("https://console.cloud.google.com/apis/credentials")

        def pick_file():
            path, _ = QFileDialog.getOpenFileName(
                dlg, "Select the OAuth client JSON you downloaded", "",
                "Google credential JSON (*.json)")
            if not path:
                return
            try:
                gd.install_client_secret(path)
            except Exception as e:
                status.setText("⚠ " + str(e))
                status.setStyleSheet("color: #e06c75; font-size: 12px;")
                return
            status.setText("✓ Credential file added. Opening Google sign-in…")
            status.setStyleSheet(f"color: {C_GREEN}; font-size: 12px;")
            dlg.accept()
            self._connect_drive()   # now configured → runs the real OAuth

        btn_row = QHBoxLayout(); btn_row.setSpacing(10)
        b_open = QPushButton("Open Google Cloud Console")
        b_open.setCursor(Qt.PointingHandCursor)
        b_open.setStyleSheet(
            f"QPushButton {{ background: {C_BG}; color: {C_TEXT}; border: 1px solid {C_BORDER};"
            f"border-radius: 8px; padding: 9px 14px; font-size: 13px; }}"
            f"QPushButton:hover {{ border: 1px solid {C_TEAL}; }}")
        b_open.clicked.connect(open_console)
        b_pick = QPushButton("I downloaded the file — Select it")
        b_pick.setCursor(Qt.PointingHandCursor)
        b_pick.setStyleSheet(
            f"QPushButton {{ background: {C_TEAL}; color: #fff; border: none; border-radius: 8px;"
            f"padding: 9px 16px; font-size: 13px; font-weight: 700; }}"
            f"QPushButton:hover {{ background: {C_TEALH}; }}")
        b_pick.clicked.connect(pick_file)
        b_close = QPushButton("Close")
        b_close.setCursor(Qt.PointingHandCursor)
        b_close.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {C_TEXT_DIM}; border: none; padding: 9px 12px; }}"
            f"QPushButton:hover {{ color: {C_TEXT}; }}")
        b_close.clicked.connect(dlg.reject)
        btn_row.addWidget(b_open); btn_row.addWidget(b_pick); btn_row.addStretch(1); btn_row.addWidget(b_close)
        lay.addLayout(btn_row)
        dlg.exec_()

    def _on_drive_failed(self, msg):
        self.btn_connect.setEnabled(True)
        self.btn_connect.setText("🔗  Connect Google Drive")
        self.lbl_conn.setText("Sign-in failed — please try again")
        self.lbl_conn.setStyleSheet(f"color: #e06c75; font-size: 13px; background: transparent;")

    # ---- Google Drive: upload (Option A) ----
    def _upload_to_drive(self):
        if not self.drive_connected:
            QMessageBox.information(self, "Connect first",
                                    "Click 'Connect Google Drive' above before uploading.")
            return
        if not getattr(self, "selected_folder", ""):
            return
        if self._upload_worker is not None:
            return
        self.btn_upload.setEnabled(False)
        self.upload_bar.setVisible(True)
        self.upload_bar.setRange(0, 100); self.upload_bar.setValue(0)
        self.lbl_upload_status.setVisible(True)
        self.lbl_upload_status.setText("Preparing upload…")
        self.lbl_upload_status.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; background: transparent;")
        self._upload_worker = _DriveUploadWorker(self.selected_folder)
        self._upload_worker.progress.connect(self._on_upload_progress)
        self._upload_worker.done.connect(self._on_upload_done)
        self._upload_worker.failed.connect(self._on_upload_failed)
        self._upload_worker.start()

    def _on_upload_progress(self, done, total, fname):
        pct = int(done / max(total, 1) * 100)
        self.upload_bar.setValue(pct)
        self.lbl_upload_status.setText(f"Uploading {total} files… {pct}%")

    def _on_upload_done(self, folder_id, folder_name):
        self._upload_worker = None
        self.drive_folder_id = folder_id
        self.drive_folder_name = folder_name
        self.btn_upload.setEnabled(True)
        self.upload_bar.setValue(100)
        self.lbl_upload_status.setText(f"✓ Upload complete. Folder ready on Drive: {folder_name}")
        self.lbl_upload_status.setStyleSheet(f"color: {C_GREEN}; font-size: 12px; font-weight: 600; background: transparent;")
        if self.status_callback:
            self.status_callback(f"Dataset uploaded to Drive: {folder_name}")
        self._validate()

    def _on_upload_failed(self, msg):
        self._upload_worker = None
        self.btn_upload.setEnabled(True)
        self.lbl_upload_status.setText("⚠ Upload failed: " + (msg.splitlines()[0] if msg else "unknown error"))
        self.lbl_upload_status.setStyleSheet(f"color: #e06c75; font-size: 12px; background: transparent;")

    # ---- Google Drive: validate existing folder (Option B) ----
    def _on_drive_link_changed(self, _text):
        self._select_source("drive")
        self.drive_folder_id = ""
        self._validate_timer.start(600)

    def _validate_drive_link(self):
        link = self.in_drive.text().strip()
        if not link:
            self.lbl_drive_validate.setVisible(False)
            return
        if not self.drive_connected:
            self.lbl_drive_validate.setVisible(True)
            self.lbl_drive_validate.setText("Connect Google Drive above to check this folder.")
            self.lbl_drive_validate.setStyleSheet("color: #e5c07b; font-size: 12px; background: transparent;")
            return
        self.lbl_drive_validate.setVisible(True)
        self.lbl_drive_validate.setText("Checking folder…")
        self.lbl_drive_validate.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; background: transparent;")
        self._validate_worker = _DriveValidateWorker(link)
        self._validate_worker.ok.connect(self._on_validate_ok)
        self._validate_worker.failed.connect(self._on_validate_failed)
        self._validate_worker.start()

    def _on_validate_ok(self, fid, name, count):
        self.drive_folder_id = fid
        self.drive_folder_name = name
        self.lbl_drive_validate.setText(f"✓ Folder found: {name} — {count} images")
        self.lbl_drive_validate.setStyleSheet(f"color: {C_GREEN}; font-size: 12px; font-weight: 600; background: transparent;")
        self._validate()

    def _on_validate_failed(self, msg):
        self.drive_folder_id = ""
        self.lbl_drive_validate.setText("⚠ " + (msg.splitlines()[0] if msg else "Could not read that folder."))
        self.lbl_drive_validate.setStyleSheet("color: #e06c75; font-size: 12px; background: transparent;")

    def showEvent(self, event):
        # reflect an existing saved connection
        try:
            import brain.gdrive as gd
            if gd.is_connected() and not self.drive_connected:
                self._mark_connected(gd.saved_account_name() or "your Google account")
        except Exception:
            pass
        super().showEvent(event)

    def _on_notebook_changed(self, index):
        use_existing = (index == 1)
        self.lbl_notebook_url.setVisible(use_existing)
        self.in_notebook_url.setVisible(use_existing)

    # ------------------------------------------------------------------
    # Validation + launch
    # ------------------------------------------------------------------
    def _validate(self):
        ready = bool(self.selected_task) and bool(self.selected_source) \
            and bool(self.in_email.text().strip())
        self.btn_launch.setEnabled(ready)
        self.btn_launch.setStyleSheet(self._launch_qss(ready))
        self.lbl_hint.setVisible(not ready)
        return ready

    def _launch_qss(self, enabled):
        if enabled:
            return f"""
                QPushButton {{ background: {C_TEAL}; color: #ffffff;
                    border: none; border-radius: 10px;
                    font-size: 16px; font-weight: 700; }}
                QPushButton:hover {{ background: {C_TEALH}; }}
            """
        return f"""
            QPushButton {{ background: {C_INACTIVE}; color: {C_TEXT_DIM};
                border: none; border-radius: 10px;
                font-size: 16px; font-weight: 700; }}
        """

    def _build_monitor_panel(self):
        panel = QFrame()
        panel.setStyleSheet(f"QFrame#mon {{ background: {C_PANEL2}; border: 1px solid {C_BLUE}; border-radius: 12px; }}")
        panel.setObjectName("mon")
        lp = QVBoxLayout(panel)
        lp.setContentsMargins(20, 16, 20, 18)
        lp.setSpacing(10)

        # header: dot + waiting + elapsed
        head = QHBoxLayout(); head.setSpacing(10)
        self.launch_dot = QLabel("●")
        self.launch_dot.setStyleSheet(f"color: {C_BLUE}; font-size: 16px; background: transparent;")
        self.launch_wait = QLabel("Waiting for Colab…")
        self.launch_wait.setStyleSheet(f"color: {C_TEXT}; font-size: 15px; font-weight: 700; background: transparent;")
        self.launch_elapsed = QLabel("Elapsed: 0s")
        self.launch_elapsed.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px; background: transparent;")
        head.addWidget(self.launch_dot, 0); head.addWidget(self.launch_wait, 0)
        head.addStretch(1); head.addWidget(self.launch_elapsed, 0)
        lp.addLayout(head)

        sub_row = QHBoxLayout(); sub_row.setSpacing(12)
        self.launch_sub = QLabel("Checking Google Drive every 2 minutes")
        self.launch_sub.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px; background: transparent;")
        self.btn_cancel_monitor = QPushButton("Cancel Monitoring")
        self.btn_cancel_monitor.setCursor(Qt.PointingHandCursor)
        self.btn_cancel_monitor.setStyleSheet(
            f"QPushButton {{ background: {C_BG}; color: {C_TEXT_DIM}; border: 1px solid {C_BORDER};"
            f"border-radius: 7px; padding: 6px 14px; font-size: 12px; }}"
            f"QPushButton:hover {{ border: 1px solid #e06c75; color: #e06c75; }}")
        self.btn_cancel_monitor.clicked.connect(self._cancel_monitor)
        sub_row.addWidget(self.launch_sub, 0); sub_row.addStretch(1); sub_row.addWidget(self.btn_cancel_monitor, 0)
        lp.addLayout(sub_row)

        # 3-hour warning (hidden)
        self.warn_box = QFrame()
        self.warn_box.setVisible(False)
        self.warn_box.setStyleSheet("QFrame { background: #3a2f12; border: 1px solid #e5c07b; border-radius: 8px; }")
        wb = QVBoxLayout(self.warn_box); wb.setContentsMargins(14, 10, 14, 12); wb.setSpacing(8)
        warn_lbl = QLabel("⚠ Training is taking longer than expected. Check your Colab session — it may "
                          "have timed out.")
        warn_lbl.setWordWrap(True)
        warn_lbl.setStyleSheet("color: #e5c07b; font-size: 13px; background: transparent;")
        wb.addWidget(warn_lbl)
        self.btn_open_nb = QPushButton("Open Notebook in Browser")
        self.btn_open_nb.setCursor(Qt.PointingHandCursor)
        self.btn_open_nb.setStyleSheet(
            f"QPushButton {{ background: {C_BG}; color: #e5c07b; border: 1px solid #e5c07b; border-radius: 7px;"
            f"padding: 7px 14px; font-size: 12px; font-weight: 600; }} QPushButton:hover {{ background: #4a3c18; }}")
        self.btn_open_nb.clicked.connect(self._open_notebook)
        wb.addWidget(self.btn_open_nb, 0, Qt.AlignLeft)
        lp.addWidget(self.warn_box)

        # green banner (hidden)
        self.result_banner = QLabel("✅  Training Complete! Your model is ready.")
        self.result_banner.setVisible(False)
        self.result_banner.setStyleSheet(
            f"background: #143524; color: {C_GREEN}; border: 1px solid {C_GREEN}; border-radius: 8px;"
            f"padding: 10px 14px; font-size: 15px; font-weight: 800;")
        lp.addWidget(self.result_banner)

        # results card (hidden)
        self.result_card = QFrame()
        self.result_card.setVisible(False)
        self.result_card.setStyleSheet(f"QFrame {{ background: {C_BG}; border: 1px solid {C_BORDER}; border-radius: 8px; }}")
        rc = QGridLayout(self.result_card); rc.setContentsMargins(16, 12, 16, 12)
        rc.setHorizontalSpacing(14); rc.setVerticalSpacing(6)

        def rclabel(t):
            l = QLabel(t); l.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; background: transparent;")
            return l

        def rcval():
            l = QLabel("—"); l.setStyleSheet(f"color: {C_TEXT}; font-size: 13px; font-weight: 600; background: transparent;")
            return l

        self.res_name = rcval(); self.res_size = rcval(); self.res_time = rcval(); self.res_metric = rcval()
        rc.addWidget(rclabel("Model file"), 0, 0); rc.addWidget(self.res_name, 0, 1)
        rc.addWidget(rclabel("File size"), 1, 0); rc.addWidget(self.res_size, 1, 1)
        rc.addWidget(rclabel("Completed"), 2, 0); rc.addWidget(self.res_time, 2, 1)
        rc.addWidget(rclabel("Best accuracy"), 3, 0); rc.addWidget(self.res_metric, 3, 1)
        self.btn_view_colab = QPushButton("View Colab results →")
        self.btn_view_colab.setCursor(Qt.PointingHandCursor); self.btn_view_colab.setFlat(True)
        self.btn_view_colab.setStyleSheet(
            f"QPushButton {{ color: {C_TEALH}; font-size: 13px; font-weight: 600; background: transparent; border: none; text-align: left; }}"
            f"QPushButton:hover {{ text-decoration: underline; }}")
        self.btn_view_colab.clicked.connect(self._open_notebook)
        rc.addWidget(self.btn_view_colab, 4, 0, 1, 2)
        lp.addWidget(self.result_card)

        # download progress (hidden)
        self.download_bar = QProgressBar(); self.download_bar.setVisible(False); self.download_bar.setFixedHeight(18)
        self.download_bar.setStyleSheet(
            f"QProgressBar {{ background: {C_BG}; border: 1px solid {C_BORDER}; border-radius: 6px;"
            f"text-align: center; color: {C_TEXT}; font-size: 11px; }}"
            f"QProgressBar::chunk {{ background: {C_TEAL}; border-radius: 5px; }}")
        lp.addWidget(self.download_bar)
        self.download_status = QLabel(""); self.download_status.setVisible(False); self.download_status.setWordWrap(True)
        self.download_status.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; background: transparent;")
        lp.addWidget(self.download_status)

        # action buttons (hidden)
        act_row = QHBoxLayout(); act_row.setSpacing(8)
        self.btn_set_default = QPushButton("Set as Default Model")
        self.btn_set_default.setVisible(False); self.btn_set_default.setCursor(Qt.PointingHandCursor)
        self.btn_set_default.setStyleSheet(
            f"QPushButton {{ background: {C_TEAL}; color: #fff; border: none; border-radius: 7px;"
            f"padding: 8px 16px; font-size: 13px; font-weight: 700; }} QPushButton:hover {{ background: {C_TEALH}; }}")
        self.btn_set_default.clicked.connect(self._set_default_colab)
        self.btn_download_now = QPushButton("⬇  Download Model Now")
        self.btn_download_now.setVisible(False); self.btn_download_now.setCursor(Qt.PointingHandCursor)
        self.btn_download_now.setStyleSheet(
            f"QPushButton {{ background: {C_BLUE}; color: #fff; border: none; border-radius: 7px;"
            f"padding: 8px 16px; font-size: 13px; font-weight: 700; }} QPushButton:hover {{ background: #4f91f7; }}")
        self.btn_download_now.clicked.connect(self._on_download_now)
        act_row.addWidget(self.btn_set_default, 0); act_row.addWidget(self.btn_download_now, 0); act_row.addStretch(1)
        lp.addLayout(act_row)

        self.lbl_drive_link = QLabel(""); self.lbl_drive_link.setVisible(False); self.lbl_drive_link.setWordWrap(True)
        self.lbl_drive_link.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lbl_drive_link.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px; background: transparent;")
        lp.addWidget(self.lbl_drive_link)

        return panel

    def _sanitize(self, text):
        keep = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (text or ""))
        return keep.strip("_") or "run"

    def _on_launch(self):
        if not self._validate():
            return
        if self._launch_worker is not None:
            return
        if not self.drive_connected:
            QMessageBox.information(self, "Connect Google Drive",
                                    "Click 'Connect Google Drive' in Step 2 first.")
            return
        if not self.drive_folder_id:
            QMessageBox.information(self, "Dataset not on Drive yet",
                                    "Upload your folder to Drive (Option A) or paste & validate a Drive "
                                    "folder link (Option B) in Step 2 first.")
            return

        import time
        base = self.drive_folder_name or ""
        if base.startswith("DrRoyApp_Training_"):
            base = base[len("DrRoyApp_Training_"):]   # avoid a doubled prefix
        self._run_name = self._sanitize(f"{self.selected_task}_{base}" if base else self.selected_task)
        self._run_date = time.strftime("%Y%m%d_%H%M")
        self._auto_download = self.chk_autodownload.isChecked()
        self._manual_download = self.chk_manuallink.isChecked()
        cfg = {
            "run_name": self._run_name,
            "date": self._run_date,
            "task": self.selected_task,
            "model_size": self.cmb_model.currentText(),
            "epochs": self.spin_epochs.value(),
            "img_size": int(self.cmb_imgsize.currentText()),
            "dataset_folder_name": self.drive_folder_name,
            "dataset_folder_id": self.drive_folder_id,
            "auto_download": self._auto_download,
        }
        nb_name = f"DrRoyApp_Notebook_{self._run_name}.ipynb"

        self.btn_launch.setEnabled(False)
        self.lbl_hint.setVisible(True)
        self.lbl_hint.setText("Building notebook and uploading to Drive…")
        self.lbl_hint.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px;")

        self._launch_worker = _LaunchWorker(cfg, nb_name)
        self._launch_worker.done.connect(self._on_launch_done)
        self._launch_worker.failed.connect(self._on_launch_failed)
        self._launch_worker.start()
        if self.status_callback:
            self.status_callback("Building Colab notebook…")

    def _on_launch_done(self, file_id):
        self._launch_worker = None
        self.btn_launch.setEnabled(True)
        self.lbl_hint.setVisible(False)

        import brain.gdrive as gd, webbrowser, time
        self._notebook_url = gd.colab_url(file_id)
        webbrowser.open(self._notebook_url)

        # record the session so the Home dashboard can show it live
        import brain.sessions as sessions
        sessions.add_session(self._run_name, self._run_date, self.selected_task, self._notebook_url)

        # reset monitoring panel to the waiting state
        self._found_meta = None
        self._warned_timeout = False
        self._monitor_start = time.time()
        self._reset_monitor_ui()
        self.launch_panel.setVisible(True)

        self._blink_timer.start(600)
        self._elapsed_timer.start(1000)
        self._poll_timer.start(120000)   # every 2 minutes
        if self.status_callback:
            self.status_callback("Colab notebook launched — watching Drive for your model…")

    def _on_launch_failed(self, msg):
        self._launch_worker = None
        self.btn_launch.setEnabled(True)
        self.lbl_hint.setVisible(True)
        self.lbl_hint.setText("⚠ Could not launch: " + (msg.splitlines()[0] if msg else "unknown error"))
        self.lbl_hint.setStyleSheet("color: #e06c75; font-size: 12px;")

    def _reset_monitor_ui(self):
        self.launch_dot.setStyleSheet(f"color: {C_BLUE}; font-size: 16px; background: transparent;")
        self.launch_wait.setText("Waiting for Colab…")
        self.launch_wait.setStyleSheet(f"color: {C_TEXT}; font-size: 15px; font-weight: 700; background: transparent;")
        self.launch_sub.setVisible(True)
        self.launch_elapsed.setText("Elapsed: 0s")
        self.launch_elapsed.setVisible(True)
        self.btn_cancel_monitor.setVisible(True)
        self.warn_box.setVisible(False)
        self.result_banner.setVisible(False)
        self.result_card.setVisible(False)
        self.download_bar.setVisible(False)
        self.download_status.setVisible(False)
        self.btn_set_default.setVisible(False)
        self.btn_download_now.setVisible(False)
        self.lbl_drive_link.setVisible(False)

    def _blink(self):
        self._blink_on = not self._blink_on
        self.launch_dot.setStyleSheet(
            f"color: {C_BLUE if self._blink_on else C_PANEL2}; font-size: 16px; background: transparent;")

    def _tick_elapsed(self):
        if not self._monitor_start:
            return
        secs = int(__import__("time").time() - self._monitor_start)
        m, s = divmod(secs, 60); h, m = divmod(m, 60)
        txt = (f"{h}h " if h else "") + (f"{m}m " if (h or m) else "") + f"{s}s"
        self.launch_elapsed.setText("Elapsed: " + txt)
        if secs >= 3 * 3600 and not self._warned_timeout and not self._found_meta:
            self._warned_timeout = True
            self.warn_box.setVisible(True)
            import brain.sessions as sessions, time
            sessions.update_session(self._run_name, self._run_date, status="timed_out",
                                    completed_at=time.strftime("%Y%m%d_%H%M%S"))

    def _poll_for_model(self):
        if self._poll_worker is not None or self._found_meta or not self.drive_connected:
            return
        self._poll_worker = _ModelPollWorker(self._run_name, self._run_date)
        self._poll_worker.found.connect(self._on_model_found)
        self._poll_worker.not_yet.connect(self._on_model_not_yet)
        self._poll_worker.failed.connect(lambda _e: setattr(self, "_poll_worker", None))
        self._poll_worker.start()

    def _on_model_not_yet(self):
        self._poll_worker = None  # keep waiting; timer fires again in 2 min

    def _on_model_found(self, meta):
        self._poll_worker = None
        if self._found_meta:
            return
        self._found_meta = meta
        self._poll_timer.stop()
        self._blink_timer.stop()
        self._elapsed_timer.stop()

        # waiting → complete
        self.launch_dot.setStyleSheet(f"color: {C_GREEN}; font-size: 16px; background: transparent;")
        self.launch_wait.setText("Training complete!")
        self.launch_sub.setVisible(False)
        self.btn_cancel_monitor.setVisible(False)
        self.warn_box.setVisible(False)

        # green banner
        self.result_banner.setVisible(True)

        # results card
        import os, time
        self._local_model_name = f"colab_{self._run_name}_{self._run_date}.pt"
        self.res_name.setText(self._local_model_name)
        self.res_size.setText(self._fmt_bytes(meta.get("size", 0)))
        when = meta.get("modified", "")
        self.res_time.setText(when.replace("T", " ").replace("Z", " UTC") if when else time.strftime("%d %b %Y, %H:%M"))
        metric = meta.get("metric")
        self.res_metric.setText(f"{metric*100:.1f}%" if isinstance(metric, (int, float)) else "—")
        self.result_card.setVisible(True)

        # always: add to Training History + mark the session complete
        self._append_colab_history(meta)
        import brain.sessions as sessions, time
        sessions.update_session(self._run_name, self._run_date, status="complete",
                                metric=meta.get("metric"),
                                completed_at=time.strftime("%Y%m%d_%H%M%S"))

        # auto-download
        if self._auto_download:
            self._start_download(meta)
        # manual download
        if self._manual_download:
            self.btn_download_now.setVisible(True)
            self.lbl_drive_link.setVisible(True)
            self.lbl_drive_link.setText(f"Drive link: {self._notebook_url}")

        if self.status_callback:
            self.status_callback("Colab training complete — model ready")

    def _fmt_bytes(self, n):
        n = float(n or 0)
        for unit in ("B", "KB", "MB", "GB"):
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} TB"

    def _model_dest(self):
        import os
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "models", f"colab_{self._run_name}_{self._run_date}.pt")

    def _start_download(self, meta):
        if self._dl_worker is not None:
            return
        self.download_bar.setVisible(True); self.download_bar.setValue(0)
        self.download_status.setVisible(True)
        self.download_status.setText("Downloading model… 0%")
        self.download_status.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; background: transparent;")
        self._dl_worker = _ModelDownloadWorker(meta["file_id"], self._model_dest())
        self._dl_worker.progress.connect(self._on_download_progress)
        self._dl_worker.done.connect(self._on_download_done)
        self._dl_worker.failed.connect(self._on_download_failed)
        self._dl_worker.start()

    def _on_download_progress(self, pct):
        self.download_bar.setValue(pct)
        self.download_status.setText(f"Downloading model… {pct}%")

    def _on_download_done(self, path):
        self._dl_worker = None
        self.download_bar.setValue(100)
        self.download_status.setText("✓ Model saved to your model library")
        self.download_status.setStyleSheet(f"color: {C_GREEN}; font-size: 12px; font-weight: 600; background: transparent;")
        self._write_model_sidecar(path)
        self.btn_set_default.setVisible(True)
        if self.status_callback:
            import os
            self.status_callback(f"Model downloaded: {os.path.basename(path)}")

    def _on_download_failed(self, msg):
        self._dl_worker = None
        self.download_status.setText("⚠ Download failed: " + (msg.splitlines()[0] if msg else "unknown error"))
        self.download_status.setStyleSheet("color: #e06c75; font-size: 12px; background: transparent;")

    def _write_model_sidecar(self, model_path):
        import os, json
        meta = self._found_meta or {}
        task = self.selected_task or meta.get("task") or "detection"
        metric = meta.get("metric")
        side = os.path.splitext(model_path)[0] + ".json"
        try:
            with open(side, "w", encoding="utf-8") as fh:
                json.dump({"origin": "Cloud-trained", "task": task,
                           "classes": "—", "params": "—",
                           "metric": metric, "model_size": self.cmb_model.currentText()}, fh, indent=2)
        except Exception:
            pass

    def _on_download_now(self):
        if self._found_meta:
            self._start_download(self._found_meta)

    def _set_default_colab(self):
        import os
        import brain.models as models
        meta = self._found_meta or {}
        task = self.selected_task or meta.get("task") or "detection"
        models.set_default(os.path.basename(self._model_dest()), task)
        self.btn_set_default.setText("✓ Default model")
        self.btn_set_default.setEnabled(False)
        if self.status_callback:
            self.status_callback("Set as default model")

    def _open_notebook(self):
        if self._notebook_url:
            import webbrowser
            webbrowser.open(self._notebook_url)

    def _cancel_monitor(self):
        self._poll_timer.stop()
        self._blink_timer.stop()
        self._elapsed_timer.stop()
        self.launch_dot.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 16px; background: transparent;")
        self.launch_wait.setText("Monitoring cancelled")
        self.launch_sub.setVisible(False)
        self.btn_cancel_monitor.setVisible(False)
        if not self._found_meta and getattr(self, "_run_name", None):
            import brain.sessions as sessions, time
            sessions.update_session(self._run_name, self._run_date, status="cancelled",
                                    completed_at=time.strftime("%Y%m%d_%H%M%S"))
        if self.status_callback:
            self.status_callback("Stopped watching Drive")

    def _append_colab_history(self, meta):
        import os, json, time
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "training_history.json")
        data = []
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if not isinstance(data, list):
                    data = []
            except Exception:
                data = []
        elapsed = int(time.time() - self._monitor_start) if self._monitor_start else 0
        metric = meta.get("metric")
        data.append({
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "task": self.selected_task or meta.get("task") or "detection",
            "model": f"colab_{self._run_name}_{self._run_date}.pt",
            "model_size": self.cmb_model.currentText(),
            "epochs": self.spin_epochs.value(),
            "requested_epochs": self.spin_epochs.value(),
            "img_size": int(self.cmb_imgsize.currentText()),
            "metric": round(metric, 4) if isinstance(metric, (int, float)) else 0,
            "dataset": self.drive_folder_name,
            "dataset_path": "Google Drive",
            "status": "Completed on Colab",
            "duration_sec": elapsed,
            "colab_url": self._notebook_url,
        })
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public: read the whole form back (used by later Colab steps)
    # ------------------------------------------------------------------
    def get_config(self):
        return {
            "task": self.selected_task,
            "source": self.selected_source,
            "folder_path": self.selected_folder,
            "drive_link": self.in_drive.text().strip(),
            "drive_folder_id": self.drive_folder_id,
            "drive_connected": self.drive_connected,
            "model_size": self.cmb_model.currentText(),
            "training_rounds": self.spin_epochs.value(),
            "image_size": int(self.cmb_imgsize.currentText()),
            "email": self.in_email.text().strip(),
            "notebook_mode": ("existing" if self.cmb_notebook.currentIndex() == 1
                              else "auto"),
            "notebook_url": self.in_notebook_url.text().strip(),
            "auto_download": self.chk_autodownload.isChecked(),
            "show_manual_link": self.chk_manuallink.isChecked(),
        }
