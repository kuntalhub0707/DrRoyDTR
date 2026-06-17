"""
Reports & Export page.

Left: patient information form + report options. Right: a live preview of the
actual PDF (rendered with PyMuPDF). Buttons generate the PDF (auto-saved to
reports/ with a timestamped name), export the findings as CSV/XLSX, or print.
"""

import os
import sys
import subprocess

import fitz  # PyMuPDF — renders the PDF for the live preview

from PyQt5.QtCore import Qt, QTimer, QDate
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
    QLineEdit, QComboBox, QPlainTextEdit, QDateEdit, QCheckBox, QScrollArea,
    QFileDialog, QMessageBox, QSizePolicy,
)

import brain.report as report

# --- palette ---
C_BG       = "#0d1117"
C_PANEL    = "#161b27"
C_PANEL2   = "#1c2333"
C_BORDER   = "#2a3045"
C_TEXT     = "#e6edf3"
C_TEXT_DIM = "#8b949e"
C_TEAL     = "#238f7a"
C_TEALH    = "#2aad93"
C_INACTIVE = "#3a4255"

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SETTINGS = os.path.join(APP_ROOT, "app_settings.json")

SAMPLE_TYPES = ["Blood Smear", "Histopathology", "FNAC", "Bone Marrow",
                "Urine Cytology", "Sputum", "Other"]

_FIELD_QSS = f"""
    QLineEdit, QComboBox, QDateEdit, QPlainTextEdit {{
        background: {C_BG}; color: {C_TEXT}; border: 1px solid {C_BORDER};
        border-radius: 6px; padding: 6px 8px; font-size: 13px; }}
    QLineEdit:focus, QComboBox:focus, QDateEdit:focus, QPlainTextEdit:focus {{ border: 1px solid {C_TEAL}; }}
    QComboBox::drop-down, QDateEdit::drop-down {{ border: none; width: 20px; }}
    QComboBox QAbstractItemView {{ background: {C_PANEL2}; color: {C_TEXT};
        selection-background-color: {C_TEAL}; border: 1px solid {C_BORDER}; outline: none; }}
"""
_TOGGLE_QSS = f"""
    QCheckBox {{ color: {C_TEXT}; font-size: 13px; spacing: 10px; }}
    QCheckBox::indicator {{ width: 40px; height: 20px; border-radius: 10px; background: {C_INACTIVE}; }}
    QCheckBox::indicator:checked {{ background: {C_TEAL}; }}
"""


def _app_default(key, fallback):
    try:
        import json
        with open(SETTINGS, "r", encoding="utf-8") as fh:
            return json.load(fh).get(key, fallback)
    except Exception:
        return fallback


class ReportsPage(QWidget):
    def __init__(self, status_callback=None):
        super().__init__()
        self.status_callback = status_callback
        self._runs = []
        self.setStyleSheet(f"background: {C_BG};")

        root = QVBoxLayout(self)
        root.setContentsMargins(30, 22, 30, 22)
        root.setSpacing(12)

        # debounce timer for live preview — MUST exist before the form is built,
        # because building the form sets field text which fires _schedule().
        self._timer = QTimer(self); self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._render_preview)

        title = QLabel("Reports & Export")
        title.setStyleSheet(f"color: {C_TEXT}; font-size: 26px; font-weight: 800;")
        root.addWidget(title)

        body = QHBoxLayout(); body.setSpacing(18)
        body.addWidget(self._build_left(), 0)
        body.addWidget(self._build_right(), 1)
        root.addLayout(body, 1)

        self._reload_runs()
        self._render_preview()

    # ------------------------------------------------------------------
    def _label(self, text):
        l = QLabel(text)
        l.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; font-weight: 600; background: transparent;")
        return l

    def _line(self, placeholder=""):
        e = QLineEdit(); e.setPlaceholderText(placeholder); e.setStyleSheet(_FIELD_QSS)
        e.textChanged.connect(self._schedule)
        return e

    def _build_left(self):
        panel = QFrame(); panel.setFixedWidth(400)
        panel.setStyleSheet(f"QFrame {{ background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 12px; }}")
        outer = QVBoxLayout(panel); outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        host = QWidget(); host.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(host); lay.setContentsMargins(18, 18, 18, 18); lay.setSpacing(8)

        # Patient Information
        lay.addWidget(self._heading("Patient Information"))
        self.in_name = self._line("Full name")
        lay.addWidget(self._label("Patient Name")); lay.addWidget(self.in_name)
        self.in_id = self._line("Lab number / MRN")
        lay.addWidget(self._label("Patient ID / Lab Number")); lay.addWidget(self.in_id)

        ag = QHBoxLayout(); ag.setSpacing(10)
        agew = QVBoxLayout(); agew.setSpacing(4)
        self.in_age = self._line("Age")
        agew.addWidget(self._label("Age")); agew.addWidget(self.in_age)
        genw = QVBoxLayout(); genw.setSpacing(4)
        self.in_gender = QComboBox(); self.in_gender.addItems(["Male", "Female", "Other"])
        self.in_gender.setStyleSheet(_FIELD_QSS); self.in_gender.currentIndexChanged.connect(self._schedule)
        genw.addWidget(self._label("Gender")); genw.addWidget(self.in_gender)
        ag.addLayout(agew, 1); ag.addLayout(genw, 1)
        lay.addLayout(ag)

        self.in_sample = QComboBox(); self.in_sample.addItems(SAMPLE_TYPES)
        self.in_sample.setStyleSheet(_FIELD_QSS); self.in_sample.currentIndexChanged.connect(self._schedule)
        lay.addWidget(self._label("Sample Type")); lay.addWidget(self.in_sample)

        self.in_date = QDateEdit(); self.in_date.setCalendarPopup(True)
        self.in_date.setDate(QDate.currentDate()); self.in_date.setDisplayFormat("dd MMM yyyy")
        self.in_date.setStyleSheet(_FIELD_QSS); self.in_date.dateChanged.connect(self._schedule)
        lay.addWidget(self._label("Date of Collection")); lay.addWidget(self.in_date)

        self.in_doctor = self._line("Referring doctor")
        lay.addWidget(self._label("Referring Doctor")); lay.addWidget(self.in_doctor)

        self.in_query = QPlainTextEdit(); self.in_query.setFixedHeight(64)
        self.in_query.setStyleSheet(_FIELD_QSS); self.in_query.textChanged.connect(self._schedule)
        lay.addWidget(self._label("Clinical Diagnosis / Query")); lay.addWidget(self.in_query)

        self.in_result = QComboBox(); self.in_result.setStyleSheet(_FIELD_QSS)
        self.in_result.currentIndexChanged.connect(self._schedule)
        lay.addWidget(self._label("Select Result")); lay.addWidget(self.in_result)

        # Report Options
        lay.addWidget(self._heading("Report Options"))
        self.opt_image = self._toggle("Include result image", True)
        self.opt_table = self._toggle("Include detection table", True)
        self.opt_explain = self._toggle("Include AI explanation (heatmap)", True)
        self.opt_model = self._toggle("Include AI model info", True)
        for t in (self.opt_image, self.opt_table, self.opt_explain, self.opt_model):
            lay.addWidget(t)

        self.in_letter = self._line(); self.in_letter.setText(_app_default("letterhead", "Tata 1mg Diagnostic Centre"))
        lay.addWidget(self._label("Letterhead")); lay.addWidget(self.in_letter)
        self.in_path = self._line(); self.in_path.setText(_app_default("pathologist", "Dr. Kuntal Roy"))
        lay.addWidget(self._label("Pathologist")); lay.addWidget(self.in_path)
        self.opt_sign = self._toggle("Signature line", True)
        lay.addWidget(self.opt_sign)

        lay.addSpacing(8)
        # buttons
        self.btn_pdf = QPushButton("Generate PDF Report")
        self.btn_pdf.setMinimumHeight(46); self.btn_pdf.setCursor(Qt.PointingHandCursor)
        self.btn_pdf.setStyleSheet(
            f"QPushButton {{ background: {C_TEAL}; color: #fff; border: none; border-radius: 9px;"
            f"font-size: 15px; font-weight: 700; }} QPushButton:hover {{ background: {C_TEALH}; }}")
        self.btn_pdf.clicked.connect(self._generate)
        lay.addWidget(self.btn_pdf)

        row = QHBoxLayout(); row.setSpacing(8)
        self.btn_export = QPushButton("Export as CSV / XLSX")
        self.btn_print = QPushButton("Print")
        for b in (self.btn_export, self.btn_print):
            b.setMinimumHeight(40); b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(
                f"QPushButton {{ background: {C_BG}; color: {C_TEXT}; border: 1px solid {C_BORDER};"
                f"border-radius: 8px; font-size: 13px; padding: 0 14px; }}"
                f"QPushButton:hover {{ border: 1px solid {C_TEAL}; }}")
        self.btn_export.clicked.connect(self._export)
        self.btn_print.clicked.connect(self._print)
        row.addWidget(self.btn_export, 1); row.addWidget(self.btn_print, 0)
        lay.addLayout(row)

        scroll.setWidget(host)
        outer.addWidget(scroll)
        return panel

    def _heading(self, text):
        l = QLabel(text)
        l.setStyleSheet(f"color: {C_TEAL}; font-size: 14px; font-weight: 800; "
                        f"letter-spacing: 0.5px; padding-top: 8px; background: transparent;")
        return l

    def _toggle(self, text, on):
        c = QCheckBox(text); c.setChecked(on); c.setCursor(Qt.PointingHandCursor)
        c.setStyleSheet(_TOGGLE_QSS); c.stateChanged.connect(self._schedule)
        return c

    def _build_right(self):
        panel = QFrame(); panel.setStyleSheet(f"QFrame {{ background: {C_PANEL}; border: 1px solid {C_BORDER}; border-radius: 12px; }}")
        lay = QVBoxLayout(panel); lay.setContentsMargins(14, 12, 14, 14); lay.setSpacing(8)
        cap = QLabel("Live PDF Preview")
        cap.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; font-weight: 700; background: transparent;")
        lay.addWidget(cap)
        self.preview_scroll = QScrollArea(); self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setFrameShape(QFrame.NoFrame)
        self.preview_scroll.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.preview_scroll.setStyleSheet(f"QScrollArea {{ border: none; background: {C_BG}; border-radius: 8px; }}")
        self.preview_label = QLabel("Preview will appear here")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet(f"color: {C_TEXT_DIM}; background: {C_BG};")
        self.preview_scroll.setWidget(self.preview_label)
        lay.addWidget(self.preview_scroll, 1)
        return panel

    # ------------------------------------------------------------------
    def showEvent(self, e):
        # pull the latest defaults configured on the Settings page
        new_lab = _app_default("letterhead", self.in_letter.text())
        new_path = _app_default("pathologist", self.in_path.text())
        if new_lab != self.in_letter.text():
            self.in_letter.setText(new_lab)
        if new_path != self.in_path.text():
            self.in_path.setText(new_path)
        self._reload_runs()
        self._render_preview()
        super().showEvent(e)

    def _reload_runs(self):
        self._runs = report.list_recent_runs()
        cur = self.in_result.currentIndex() if self.in_result.count() else 0
        self.in_result.blockSignals(True)
        self.in_result.clear()
        if self._runs:
            for r in self._runs:
                n = sum(f.get("count", 0) for f in r.get("findings", []))
                self.in_result.addItem(f"{r.get('image_name', '—')} · {n} findings · {r.get('timestamp', '')}")
            self.in_result.setCurrentIndex(min(cur, len(self._runs) - 1))
        else:
            self.in_result.addItem("(no recent results — run Predict first)")
        self.in_result.blockSignals(False)

    def _selected_run(self):
        if not self._runs:
            return None
        i = self.in_result.currentIndex()
        if 0 <= i < len(self._runs):
            return self._runs[i]
        return None

    def _collect(self):
        return {
            "patient": {
                "name": self.in_name.text(),
                "id": self.in_id.text(),
                "age": self.in_age.text(),
                "gender": self.in_gender.currentText(),
                "sample_type": self.in_sample.currentText(),
                "collection_date": self.in_date.date().toString("dd MMM yyyy"),
                "referring_doctor": self.in_doctor.text(),
                "clinical_query": self.in_query.toPlainText(),
            },
            "options": {
                "include_image": self.opt_image.isChecked(),
                "include_table": self.opt_table.isChecked(),
                "include_explanation": self.opt_explain.isChecked(),
                "include_model": self.opt_model.isChecked(),
                "signature_line": self.opt_sign.isChecked(),
            },
            "letterhead": self.in_letter.text(),
            "pathologist": self.in_path.text(),
            "result": self._selected_run(),
        }

    # ------------------------------------------------------------------
    def _schedule(self):
        self._timer.start(350)

    def _render_preview(self):
        try:
            pdf_bytes = report.build_report_bytes(self._collect())
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = doc[0]
            target_w = max(360, self.preview_scroll.viewport().width() - 24)
            zoom = target_w / page.rect.width
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            # Render to PNG bytes and load that — avoids referencing fitz's raw
            # pixel buffer (which would be freed and crash on .copy()).
            png = pix.tobytes("png")
            doc.close()
            qpix = QPixmap()
            qpix.loadFromData(png, "PNG")
            self.preview_label.setPixmap(qpix)
            self.preview_label.setStyleSheet("background: white;")
        except Exception as e:
            self.preview_label.setText(f"Preview error:\n{e}")

    # ------------------------------------------------------------------
    def _generate(self):
        try:
            dest = report.save_report(self._collect())
        except Exception as e:
            QMessageBox.warning(self, "Could not generate", str(e))
            return
        if self.status_callback:
            self.status_callback(f"Report saved: {os.path.basename(dest)}")
        resp = QMessageBox.question(
            self, "Report saved",
            f"PDF saved automatically to:\n{dest}\n\nOpen it now?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if resp == QMessageBox.Yes:
            self._open_file(dest)

    def _export(self):
        run = self._selected_run()
        if not run or not run.get("findings"):
            QMessageBox.information(self, "Nothing to export",
                                    "Select a result with findings first (run Predict to create one).")
            return
        import pandas as pd
        df = pd.DataFrame([{"Class": f.get("class"), "Count": f.get("count"),
                            "Avg Confidence": f"{(f.get('avg_conf', 0) or 0)*100:.1f}%"}
                           for f in run["findings"]])
        default = os.path.join(APP_ROOT, "output", f"findings_{run.get('timestamp', '')}.xlsx")
        dest, sel = QFileDialog.getSaveFileName(
            self, "Export findings", default, "Excel (*.xlsx);;CSV (*.csv)")
        if not dest:
            return
        try:
            if dest.lower().endswith(".csv"):
                df.to_csv(dest, index=False)
            else:
                df.to_excel(dest, index=False)
            QMessageBox.information(self, "Exported", f"Saved to:\n{dest}")
        except Exception as e:
            QMessageBox.warning(self, "Could not export", str(e))

    def _print(self):
        try:
            dest = report.save_report(self._collect())
        except Exception as e:
            QMessageBox.warning(self, "Could not prepare", str(e))
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(dest, "print")  # noqa: hand off to default PDF printer
            elif sys.platform == "darwin":
                subprocess.Popen(["lpr", dest])
            else:
                subprocess.Popen(["lp", dest])
        except Exception:
            # fall back to just opening it so the user can print manually
            self._open_file(dest)
        if self.status_callback:
            self.status_callback("Sent report to printer")

    def _open_file(self, path):
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)  # noqa
            elif sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            QMessageBox.warning(self, "Could not open", str(e))
