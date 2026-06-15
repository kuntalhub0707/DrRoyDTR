"""
Dataset Library page.

Every dataset the user uploads/imports is recorded in datasets.json. This page
shows them as cards (name editable inline, format/task badges, image counts,
split, class names, date) with 'Train with this' and 'Delete' actions.

Constructed with:
  navigate(index)         -> jump to another page
  train_callback(entry)   -> open Train Model pre-loaded with this dataset

Deleting only removes the library entry — it never touches the user's files.
"""

import os
import json
import time

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QComboBox, QLineEdit, QScrollArea, QFileDialog, QMessageBox, QSizePolicy,
)

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
DATASETS_JSON = os.path.join(APP_ROOT, "datasets.json")

IDX_TRAIN_MODEL = 1

BADGE_COLOR = {"YOLO": "#3b82f6", "COCO": "#8b5cf6", "VOC": "#0ea5e9", "FOLDER": "#64748b"}
TASK_COLOR = {"detection": C_TEAL, "classification": "#b8860b"}


def _load_datasets():
    if not os.path.isfile(DATASETS_JSON):
        return []
    try:
        with open(DATASETS_JSON, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_datasets(items):
    with open(DATASETS_JSON, "w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=2)


def _fmt_date(ts):
    try:
        return time.strftime("%d %b %Y", time.strptime(ts, "%Y%m%d_%H%M%S"))
    except Exception:
        return ts or "—"


def _pill(text, bg, fg="#ffffff"):
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"background: {bg}; color: {fg}; border-radius: 9px; padding: 2px 10px; "
        f"font-size: 11px; font-weight: 700;")
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
    return lbl


class DatasetLibraryPage(QWidget):
    def __init__(self, navigate=None, train_callback=None):
        super().__init__()
        self.navigate = navigate or (lambda _i: None)
        self.train_callback = train_callback
        self._items = []
        self.setStyleSheet(f"background: {C_BG};")

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 28, 40, 28)
        root.setSpacing(10)

        # ---- header ----
        head = QHBoxLayout()
        left = QVBoxLayout(); left.setSpacing(2)
        title = QLabel("Dataset Library")
        title.setStyleSheet(f"color: {C_TEXT}; font-size: 26px; font-weight: 800;")
        left.addWidget(title)
        head.addLayout(left, 1)

        self.btn_upload = QPushButton("⬆  Upload New Dataset")
        self.btn_upload.setCursor(Qt.PointingHandCursor)
        self.btn_upload.setStyleSheet(
            f"QPushButton {{ background: #000000; color: #ffffff; border: 1px solid {C_BORDER};"
            f"border-radius: 8px; padding: 10px 18px; font-size: 13px; font-weight: 700; }}"
            f"QPushButton:hover {{ border: 1px solid {C_TEAL}; }}")
        self.btn_upload.clicked.connect(self._upload)
        head.addWidget(self.btn_upload, 0, Qt.AlignTop)
        root.addLayout(head)

        desc = QLabel("Every dataset you have uploaded or imported lives here. Click any row to rename, "
                      "see image counts, and replay runs. Re-use any dataset by going to Train Model "
                      "and choosing Pick from Library.")
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 14px;")
        root.addWidget(desc)

        # ---- sort bar ----
        self.sort_bar = QHBoxLayout()
        sort_lbl = QLabel("Sort")
        sort_lbl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; font-weight: 700;")
        self.cmb_sort = QComboBox()
        self.cmb_sort.addItems(["Most Recently Used", "Newest", "Name A–Z", "Most Images"])
        self.cmb_sort.setStyleSheet(f"""
            QComboBox {{ background: {C_BG}; color: {C_TEXT}; border: 1px solid {C_BORDER};
                border-radius: 7px; padding: 6px 10px; font-size: 13px; min-width: 180px; }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox QAbstractItemView {{ background: {C_PANEL2}; color: {C_TEXT};
                selection-background-color: {C_TEAL}; border: 1px solid {C_BORDER}; outline: none; }}
        """)
        self.cmb_sort.currentIndexChanged.connect(self._render)
        self.sort_bar.addStretch(1)
        self.sort_bar.addWidget(sort_lbl, 0)
        self.sort_bar.addWidget(self.cmb_sort, 0)
        sb = QWidget(); sb.setLayout(self.sort_bar)
        self.sort_bar_widget = sb
        root.addWidget(sb)

        # ---- empty state ----
        self.empty = self._build_empty_state()
        root.addWidget(self.empty, 1)

        # ---- cards grid (scrollable) ----
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.grid_host = QWidget(); self.grid_host.setStyleSheet("background: transparent;")
        self.grid = QGridLayout(self.grid_host)
        self.grid.setContentsMargins(0, 4, 0, 4)
        self.grid.setHorizontalSpacing(16)
        self.grid.setVerticalSpacing(16)
        self.scroll.setWidget(self.grid_host)
        root.addWidget(self.scroll, 1)

        self.reload_from_disk()

    # ------------------------------------------------------------------
    def _build_empty_state(self):
        wrap = QFrame(); wrap.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(wrap)
        lay.addStretch(1)
        icon = QLabel("🗄️"); icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet("font-size: 56px; background: transparent;")
        lay.addWidget(icon)
        msg = QLabel("No datasets yet"); msg.setAlignment(Qt.AlignCenter)
        msg.setStyleSheet(f"color: {C_TEXT}; font-size: 18px; font-weight: 700; background: transparent;")
        lay.addWidget(msg)
        link = QPushButton("Go to Upload →")
        link.setCursor(Qt.PointingHandCursor); link.setFlat(True)
        link.setStyleSheet(
            f"QPushButton {{ color: {C_BLUE}; font-size: 14px; font-weight: 600; background: transparent; border: none; }}"
            f"QPushButton:hover {{ text-decoration: underline; }}")
        link.clicked.connect(self._upload)
        row = QHBoxLayout(); row.addStretch(1); row.addWidget(link); row.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(2)
        return wrap

    # ------------------------------------------------------------------
    def showEvent(self, event):
        self.reload_from_disk()
        super().showEvent(event)

    def reload_from_disk(self):
        self._items = _load_datasets()
        self._render()

    def _sorted_items(self):
        items = list(self._items)
        mode = self.cmb_sort.currentText()
        if mode == "Most Recently Used":
            items.sort(key=lambda d: d.get("last_used", d.get("date_added", "")), reverse=True)
        elif mode == "Newest":
            items.sort(key=lambda d: d.get("date_added", ""), reverse=True)
        elif mode == "Name A–Z":
            items.sort(key=lambda d: d.get("name", "").lower())
        elif mode == "Most Images":
            items.sort(key=lambda d: d.get("total_images", 0), reverse=True)
        return items

    def _render(self):
        has = bool(self._items)
        self.empty.setVisible(not has)
        self.scroll.setVisible(has)
        self.sort_bar_widget.setVisible(has)

        # clear grid (detach immediately so old cards don't ghost behind new ones)
        while self.grid.count():
            it = self.grid.takeAt(0)
            w = it.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        if not has:
            return
        cols = 2
        for i, entry in enumerate(self._sorted_items()):
            self.grid.addWidget(self._card(entry), i // cols, i % cols)
        # keep cards from stretching to full height
        self.grid.setRowStretch((len(self._items) + 1) // cols, 1)

    # ------------------------------------------------------------------
    def _card(self, entry):
        card = QFrame()
        card.setStyleSheet(
            f"QFrame#card {{ background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 12px; }}")
        card.setObjectName("card")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(10)

        # name (inline editable) + date
        top = QHBoxLayout()
        name = QLineEdit(entry.get("name", "(unnamed)"))
        name.setStyleSheet(
            f"QLineEdit {{ background: transparent; border: none; color: {C_TEXT};"
            f"font-size: 18px; font-weight: 800; padding: 0; }}"
            f"QLineEdit:focus {{ border-bottom: 1px solid {C_TEAL}; }}")
        name.setToolTip("Click to rename")
        name.editingFinished.connect(lambda e=entry, w=name: self._rename(e, w.text()))
        top.addWidget(name, 1)
        date = QLabel("📅 " + _fmt_date(entry.get("date_added", "")))
        date.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; background: transparent;")
        top.addWidget(date, 0, Qt.AlignTop)
        lay.addLayout(top)

        # badges
        badges = QHBoxLayout(); badges.setSpacing(8)
        badge = entry.get("format_badge", "?")
        badges.addWidget(_pill(badge, BADGE_COLOR.get(badge, C_BORDER)))
        task = entry.get("task", "detection")
        badges.addWidget(_pill("Detection" if task == "detection" else "Classification",
                               TASK_COLOR.get(task, C_TEAL)))
        badges.addStretch(1)
        lay.addLayout(badges)

        # image count + split
        sp = entry.get("splits", {}) or {}
        total = QLabel(f"🖼  {entry.get('total_images', 0)} images")
        total.setStyleSheet(f"color: {C_TEXT}; font-size: 14px; font-weight: 600; background: transparent;")
        lay.addWidget(total)
        split = QLabel(f"Train {sp.get('train', 0)}  ·  Valid {sp.get('valid', 0)}  ·  Test {sp.get('test', 0)}")
        split.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; background: transparent;")
        lay.addWidget(split)

        # classes (first 5 + ...more)
        classes = entry.get("classes", []) or []
        if classes:
            shown = ", ".join(classes[:5])
            if len(classes) > 5:
                shown += f"  …+{len(classes) - 5} more"
            cls = QLabel("🏷  " + shown)
        else:
            cls = QLabel("🏷  (no class names found)")
        cls.setWordWrap(True)
        cls.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; background: transparent;")
        lay.addWidget(cls)

        # path hint (small)
        path = QLabel(entry.get("path", ""))
        path.setStyleSheet(f"color: {C_BORDER}; font-size: 10px; background: transparent;")
        path.setWordWrap(True)
        lay.addWidget(path)

        # actions
        actions = QHBoxLayout(); actions.setSpacing(8)
        b_train = QPushButton("Train with this")
        b_train.setCursor(Qt.PointingHandCursor)
        b_train.setStyleSheet(
            f"QPushButton {{ background: {C_TEAL}; color: #fff; border: none; border-radius: 7px;"
            f"padding: 8px 14px; font-size: 13px; font-weight: 700; }}"
            f"QPushButton:hover {{ background: {C_TEALH}; }}")
        b_train.clicked.connect(lambda _c, e=entry: self._train(e))
        b_del = QPushButton("Delete")
        b_del.setCursor(Qt.PointingHandCursor)
        b_del.setStyleSheet(
            f"QPushButton {{ background: {C_BG}; color: {C_TEXT_DIM}; border: 1px solid {C_BORDER};"
            f"border-radius: 7px; padding: 8px 14px; font-size: 13px; }}"
            f"QPushButton:hover {{ border: 1px solid {C_RED}; color: {C_RED}; }}")
        b_del.clicked.connect(lambda _c, e=entry: self._delete(e))
        actions.addWidget(b_train, 1)
        actions.addWidget(b_del, 0)
        lay.addLayout(actions)
        return card

    # ------------------------------------------------------------------
    def _upload(self):
        folder = QFileDialog.getExistingDirectory(self, "Select a dataset folder")
        if not folder:
            return
        # already in library?
        for e in self._items:
            if os.path.normpath(e.get("path", "")) == os.path.normpath(folder):
                QMessageBox.information(self, "Already added",
                                        "That folder is already in your Dataset Library.")
                return
        import brain.dataset as dataset
        meta = dataset.scan_dataset(folder)
        if not meta:
            QMessageBox.warning(self, "Not recognised",
                                "Could not recognise this as a dataset.\n\nSupported: Roboflow/plain YOLO, "
                                "COCO (.json), Pascal VOC (.xml), or an image-classification folder "
                                "(one sub-folder per category).")
            return
        now = time.strftime("%Y%m%d_%H%M%S")
        entry = {
            "id": now,
            "name": os.path.basename(os.path.normpath(folder)),
            "path": folder,
            "date_added": now,
            "last_used": now,
        }
        entry.update(meta)
        self._items.append(entry)
        _save_datasets(self._items)
        self.cmb_sort.setCurrentText("Newest")
        self._render()

    def _rename(self, entry, new_name):
        new_name = (new_name or "").strip()
        if not new_name or new_name == entry.get("name"):
            return
        entry["name"] = new_name
        _save_datasets(self._items)

    def _train(self, entry):
        entry["last_used"] = time.strftime("%Y%m%d_%H%M%S")
        _save_datasets(self._items)
        if self.train_callback:
            self.train_callback(entry)
        else:
            self.navigate(IDX_TRAIN_MODEL)

    def _delete(self, entry):
        resp = QMessageBox.question(
            self, "Remove dataset",
            f"Remove “{entry.get('name')}” from the library?\n\n"
            "This only removes it from this list — your image files on disk are NOT deleted.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if resp != QMessageBox.Yes:
            return
        self._items = [e for e in self._items if e is not entry]
        _save_datasets(self._items)
        self._render()
