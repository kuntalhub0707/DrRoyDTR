"""
Model Library page.

Shows the 8 pre-bundled starter models plus any trained/imported model, split
into Detection and Classification sections. Each card can be set as the default
for its task (Predict uses the default automatically), downloaded/exported,
revealed in the file manager, or deleted.
"""

import os
import sys
import subprocess

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout,
    QFrame, QScrollArea, QFileDialog, QMessageBox, QSizePolicy,
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

ORIGIN_COLOR = {"Bundled": "#64748b", "Fine-tuned": C_TEAL, "Imported": "#8b5cf6"}


def _pill(text, bg, fg="#ffffff"):
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"background: {bg}; color: {fg}; border-radius: 9px; padding: 2px 10px; "
        f"font-size: 11px; font-weight: 700;")
    lbl.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
    return lbl


class ModelLibraryPage(QWidget):
    def __init__(self, status_callback=None):
        super().__init__()
        self.status_callback = status_callback
        self._threads = []
        self.setStyleSheet(f"background: {C_BG};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        host = QWidget(); host.setStyleSheet("background: transparent;")
        self.root = QVBoxLayout(host)
        self.root.setContentsMargins(40, 28, 40, 32)
        self.root.setSpacing(12)

        # ---- header ----
        head = QHBoxLayout()
        left = QVBoxLayout(); left.setSpacing(2)
        title_row = QHBoxLayout(); title_row.setSpacing(12)
        title = QLabel("Model Library")
        title.setStyleSheet(f"color: {C_TEXT}; font-size: 26px; font-weight: 800;")
        title_row.addWidget(title, 0)
        self.count_badge = QLabel("")
        self.count_badge.setStyleSheet(
            f"background: {C_PANEL2}; color: {C_TEXT_DIM}; border: 1px solid {C_BORDER};"
            f"border-radius: 11px; padding: 2px 12px; font-size: 12px; font-weight: 700;")
        self.count_badge.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        title_row.addWidget(self.count_badge, 0, Qt.AlignVCenter)
        title_row.addStretch(1)
        left.addLayout(title_row)
        head.addLayout(left, 1)

        self.btn_import = QPushButton("⬇  Import .pt file")
        self.btn_import.setCursor(Qt.PointingHandCursor)
        self.btn_import.setStyleSheet(
            f"QPushButton {{ background: #000000; color: #fff; border: 1px solid {C_BORDER};"
            f"border-radius: 8px; padding: 10px 18px; font-size: 13px; font-weight: 700; }}"
            f"QPushButton:hover {{ border: 1px solid {C_TEAL}; }}")
        self.btn_import.clicked.connect(self._import)
        head.addWidget(self.btn_import, 0, Qt.AlignTop)
        self.root.addLayout(head)

        desc = QLabel("Pre-loaded starter models plus any model you have trained or imported. "
                      "Set one as default and Predict uses it automatically.")
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 14px;")
        self.root.addWidget(desc)

        # ---- sections container ----
        self.sections_host = QWidget(); self.sections_host.setStyleSheet("background: transparent;")
        self.sections = QVBoxLayout(self.sections_host)
        self.sections.setContentsMargins(0, 0, 0, 0)
        self.sections.setSpacing(22)
        self.root.addWidget(self.sections_host)
        self.root.addStretch(1)

        scroll.setWidget(host)
        outer.addWidget(scroll)

        self.refresh()

    # ------------------------------------------------------------------
    def showEvent(self, event):
        self.refresh()
        super().showEvent(event)

    def _clear_sections(self):
        while self.sections.count():
            it = self.sections.takeAt(0)
            w = it.widget()
            if w:
                w.setParent(None)
                w.deleteLater()

    def refresh(self):
        det, cls = models.list_models()
        self.count_badge.setText(f"{len(det) + len(cls)} models")
        self._clear_sections()
        self.sections.addWidget(self._section(
            "🎯", "Detection Models",
            "Per-object localisation. Cell counts, mitosis, malaria, WBC differential.", det))
        self.sections.addWidget(self._section(
            "🏷️", "Classification Models",
            "Per-image label. Tumour subtype, Gleason grade, smear pathology.", cls))

    # ------------------------------------------------------------------
    def _section(self, icon, heading, subtitle, items):
        wrap = QFrame(); wrap.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        hd = QLabel(f"{icon}  {heading}")
        hd.setStyleSheet(f"color: {C_TEXT}; font-size: 19px; font-weight: 800; background: transparent;")
        lay.addWidget(hd)
        sub = QLabel(subtitle)
        sub.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px; background: transparent;")
        lay.addWidget(sub)

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)
        grid.setContentsMargins(0, 6, 0, 0)
        cols = 2
        for i, m in enumerate(items):
            grid.addWidget(self._card(m), i // cols, i % cols)
        lay.addLayout(grid)
        return wrap

    def _card(self, m):
        card = QFrame(); card.setObjectName("mcard")
        border = C_BLUE if m["is_default"] else C_BORDER
        card.setStyleSheet(
            f"QFrame#mcard {{ background: {C_PANEL2}; border: 2px solid {border}; border-radius: 12px; }}")
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(8)

        # filename + default badge
        top = QHBoxLayout(); top.setSpacing(8)
        fn = QLabel(m["file"])
        fn.setStyleSheet(f"color: {C_TEXT}; font-size: 17px; font-weight: 800; background: transparent;")
        top.addWidget(fn, 0)
        if m["is_default"]:
            top.addWidget(_pill("Default", C_BLUE), 0)
        top.addStretch(1)
        if not m["downloaded"]:
            top.addWidget(_pill("not downloaded", C_BG, C_TEXT_DIM), 0)
        lay.addLayout(top)

        # origin · task label
        origin = m["origin"]
        task_word = "Detection" if m["task"] == "detection" else "Classification"
        label = QLabel(f"{origin} · {task_word}" + (f" · {m['size_name']}" if m.get("size_name") else ""))
        label.setStyleSheet(f"color: {ORIGIN_COLOR.get(origin, C_TEXT_DIM)}; font-size: 12px; font-weight: 700; background: transparent;")
        lay.addWidget(label)

        # specs
        specs = QLabel(f"Classes: {m['classes']}     Parameters: {m['params']}     Size: {m['size_str']}")
        specs.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; background: transparent;")
        lay.addWidget(specs)

        path = QLabel(m["path"] if m["downloaded"] else "(not downloaded yet — click Download)")
        path.setWordWrap(True)
        path.setStyleSheet(f"color: {C_BORDER}; font-size: 10px; background: transparent;")
        lay.addWidget(path)

        # buttons
        btns = QHBoxLayout(); btns.setSpacing(6)
        # Set as Default
        if m["is_default"]:
            b_def = QPushButton("✓ Default"); b_def.setEnabled(False)
            b_def.setStyleSheet(
                f"QPushButton {{ background: {C_BG}; color: {C_BLUE}; border: 1px solid {C_BLUE};"
                f"border-radius: 6px; padding: 6px 10px; font-size: 12px; font-weight: 700; }}")
        else:
            b_def = QPushButton("Set as Default")
            b_def.setCursor(Qt.PointingHandCursor)
            b_def.setStyleSheet(self._btn_qss(primary=True))
            b_def.clicked.connect(lambda _c, mm=m: self._set_default(mm))
        btns.addWidget(b_def)

        # Download / Export
        b_dl = QPushButton("Download" if not m["downloaded"] else "Export…")
        b_dl.setCursor(Qt.PointingHandCursor)
        b_dl.setStyleSheet(self._btn_qss())
        b_dl.clicked.connect(lambda _c, mm=m, btn=b_dl: self._download(mm, btn))
        btns.addWidget(b_dl)

        # Reveal in Folder
        b_rev = QPushButton("Reveal in Folder")
        b_rev.setCursor(Qt.PointingHandCursor)
        b_rev.setEnabled(m["downloaded"])
        b_rev.setStyleSheet(self._btn_qss())
        b_rev.clicked.connect(lambda _c, mm=m: self._reveal(mm))
        btns.addWidget(b_rev)

        # Delete
        b_del = QPushButton("Delete")
        b_del.setCursor(Qt.PointingHandCursor)
        b_del.setEnabled(m["downloaded"])
        b_del.setStyleSheet(self._btn_qss(danger=True))
        b_del.clicked.connect(lambda _c, mm=m: self._delete(mm))
        btns.addWidget(b_del)

        btns.addStretch(1)
        lay.addLayout(btns)
        return card

    def _btn_qss(self, primary=False, danger=False):
        if primary:
            return (f"QPushButton {{ background: {C_TEAL}; color: #fff; border: none; border-radius: 6px;"
                    f"padding: 6px 12px; font-size: 12px; font-weight: 700; }}"
                    f"QPushButton:hover {{ background: {C_TEALH}; }}")
        hover = C_RED if danger else C_TEAL
        color = C_TEXT_DIM
        return (f"QPushButton {{ background: {C_BG}; color: {color}; border: 1px solid {C_BORDER};"
                f"border-radius: 6px; padding: 6px 12px; font-size: 12px; }}"
                f"QPushButton:hover {{ border: 1px solid {hover}; color: {hover}; }}"
                f"QPushButton:disabled {{ color: {C_BORDER}; border: 1px solid {C_BORDER}; }}")

    # ------------------------------------------------------------------
    def _set_default(self, m):
        models.set_default(m["file"], m["task"])
        if self.status_callback:
            self.status_callback(f"AI Model: {m['file']} set as default ({m['task']})")
        self.refresh()

    def _download(self, m, btn):
        if m["downloaded"]:
            # export a copy elsewhere
            dest, _ = QFileDialog.getSaveFileName(self, "Save a copy of the model", m["file"], "PyTorch model (*.pt)")
            if dest:
                import shutil
                try:
                    shutil.copy(m["path"], dest)
                    QMessageBox.information(self, "Saved", f"Copied to:\n{dest}")
                except Exception as e:
                    QMessageBox.warning(self, "Could not save", str(e))
            return
        # fetch the bundled weight in the background
        btn.setEnabled(False)
        btn.setText("Downloading…")
        try:
            from brain.trainer import BaseModelDownloader
        except Exception as e:
            QMessageBox.warning(self, "Unavailable", str(e))
            btn.setEnabled(True); btn.setText("Download")
            return
        th = BaseModelDownloader(m["file"])
        th.status.connect(lambda s: btn.setText(s if "Download" in s else "Downloading…"))
        th.ready.connect(lambda _p, mm=m: self._on_downloaded(mm))
        th.failed.connect(lambda e, b=btn: (QMessageBox.warning(self, "Download failed", e),
                                            b.setEnabled(True), b.setText("Download")))
        self._threads.append(th)
        if self.status_callback:
            self.status_callback(f"Downloading {m['file']}…")
        th.start()

    def _on_downloaded(self, m):
        if self.status_callback:
            self.status_callback(f"{m['file']} downloaded")
        self.refresh()

    def _reveal(self, m):
        if not m["downloaded"]:
            return
        path = m["path"]
        try:
            if sys.platform.startswith("win"):
                subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-R", path])
            else:
                subprocess.Popen(["xdg-open", os.path.dirname(path)])
        except Exception as e:
            QMessageBox.warning(self, "Could not open folder", str(e))

    def _delete(self, m):
        if not m["downloaded"]:
            return
        if m["bundled"]:
            text = (f"Remove the downloaded file for “{m['file']}”?\n\n"
                    "It's a bundled starter model — you can download it again anytime.")
        else:
            text = (f"Permanently delete “{m['file']}”?\n\nThis removes the model file from disk and cannot be undone.")
        resp = QMessageBox.question(self, "Delete model", text,
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if resp != QMessageBox.Yes:
            return
        if models.delete_model(m["file"]):
            if self.status_callback:
                self.status_callback(f"{m['file']} deleted")
            self.refresh()
        else:
            QMessageBox.warning(self, "Could not delete", "The file may be open in another program.")

    def _import(self):
        src, _ = QFileDialog.getOpenFileName(self, "Import a .pt model file", "", "PyTorch model (*.pt)")
        if not src:
            return
        try:
            name = models.import_pt(src)
        except Exception as e:
            QMessageBox.warning(self, "Import failed", str(e))
            return
        if self.status_callback:
            self.status_callback(f"Imported {name}")
        self.refresh()
