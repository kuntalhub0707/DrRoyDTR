"""
Settings page.

Persists preferences to app_settings.json (read by the Reports page for its
letterhead/pathologist defaults, and by other pages over time). Every change
saves immediately; 'Reset to Defaults' restores the originals.
"""

import os
import json
from datetime import datetime

try:
    from zoneinfo import ZoneInfo, available_timezones
    _HAS_TZ = True
except Exception:
    _HAS_TZ = False

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QLineEdit,
    QCheckBox, QListWidget, QListWidgetItem, QScrollArea, QSizePolicy,
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
C_INACTIVE = "#3a4255"

from brain.paths import APP_ROOT
SETTINGS = os.path.join(APP_ROOT, "app_settings.json")

DEFAULTS = {
    "show_workflow_presets": False,
    "letterhead": "Tata 1mg Diagnostic Centre",
    "pathologist": "Dr. Kuntal Roy",
    "include_ai_disclaimer": True,
    "auto_save_best": True,
    "auto_delete_history_30d": False,
    "timezone": "Asia/Kolkata",
}

_FIELD_QSS = f"""
    QLineEdit {{ background: {C_BG}; color: {C_TEXT}; border: 1px solid {C_BORDER};
        border-radius: 6px; padding: 8px 10px; font-size: 13px; }}
    QLineEdit:focus {{ border: 1px solid {C_TEAL}; }}
"""
_TOGGLE_QSS = f"""
    QCheckBox {{ color: {C_TEXT}; font-size: 14px; spacing: 12px; }}
    QCheckBox::indicator {{ width: 44px; height: 22px; border-radius: 11px; background: {C_INACTIVE}; }}
    QCheckBox::indicator:checked {{ background: {C_TEAL}; }}
"""


def load_settings():
    data = dict(DEFAULTS)
    if os.path.isfile(SETTINGS):
        try:
            with open(SETTINGS, "r", encoding="utf-8") as fh:
                data.update(json.load(fh) or {})
        except Exception:
            pass
    return data


def save_settings(data):
    try:
        with open(SETTINGS, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except Exception:
        pass


class SettingsPage(QWidget):
    def __init__(self, status_callback=None):
        super().__init__()
        self.status_callback = status_callback
        self._loading = True
        self.settings = load_settings()
        self.setStyleSheet(f"background: {C_BG};")

        scroll = QScrollArea(self); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        host = QWidget(); host.setStyleSheet("background: transparent;")
        root = QVBoxLayout(host)
        root.setContentsMargins(40, 28, 40, 32)
        root.setSpacing(14)

        title = QLabel("Tune the App to Your Workflow")
        title.setStyleSheet(f"color: {C_TEXT}; font-size: 26px; font-weight: 800;")
        root.addWidget(title)

        # ---- Section 1: Predict Settings ----
        c1, s1 = self._section("Predict Settings"); root.addWidget(c1)
        self.chk_presets = self._toggle("Show Clinical Workflow Presets", self.settings["show_workflow_presets"])
        s1.addWidget(self.chk_presets)
        s1.addWidget(self._hint("Adds quick-select chips for common pathology tasks to the Predict page."))

        # ---- Section 2: Report Defaults ----
        c2, s2 = self._section("Report Defaults"); root.addWidget(c2)
        s2.addWidget(self._label("Default Lab Name"))
        self.in_lab = QLineEdit(self.settings["letterhead"]); self.in_lab.setStyleSheet(_FIELD_QSS)
        self.in_lab.textChanged.connect(self._save)
        s2.addWidget(self.in_lab)
        s2.addWidget(self._label("Default Pathologist Name"))
        self.in_path = QLineEdit(self.settings["pathologist"]); self.in_path.setStyleSheet(_FIELD_QSS)
        self.in_path.textChanged.connect(self._save)
        s2.addWidget(self.in_path)
        self.chk_disclaimer = self._toggle("Always include AI disclaimer", self.settings["include_ai_disclaimer"])
        s2.addWidget(self.chk_disclaimer)

        # ---- Section 3: Training Defaults ----
        c3, s3 = self._section("Training Defaults"); root.addWidget(c3)
        self.chk_autosave = self._toggle("Auto-save best model", self.settings["auto_save_best"])
        self.chk_autodelete = self._toggle("Auto-delete run history after 30 days", self.settings["auto_delete_history_30d"])
        s3.addWidget(self.chk_autosave)
        s3.addWidget(self.chk_autodelete)

        # ---- Section 4: Timezone ----
        c4, s4 = self._section("Timezone"); root.addWidget(c4)
        self.lbl_clock = QLabel("")
        self.lbl_clock.setStyleSheet(f"color: {C_TEALH}; font-size: 14px; font-weight: 700; background: transparent;")
        s4.addWidget(self.lbl_clock)
        self.tz_list = QListWidget()
        self.tz_list.setFixedHeight(220)
        self.tz_list.setStyleSheet(f"""
            QListWidget {{ background: {C_BG}; color: {C_TEXT}; border: 1px solid {C_BORDER};
                border-radius: 8px; font-size: 13px; outline: none; }}
            QListWidget::item {{ padding: 6px 10px; }}
            QListWidget::item:selected {{ background: {C_TEAL}; color: #fff; }}
        """)
        zones = sorted(available_timezones()) if _HAS_TZ else ["Asia/Kolkata", "UTC"]
        self.tz_list.addItems(zones)
        self._select_tz(self.settings["timezone"])
        self.tz_list.currentTextChanged.connect(self._on_tz_changed)
        s4.addWidget(self.tz_list)
        root.addWidget(s4.parent())

        # ---- Reset button ----
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.btn_reset = QPushButton("Reset to Defaults")
        self.btn_reset.setCursor(Qt.PointingHandCursor); self.btn_reset.setMinimumHeight(40)
        self.btn_reset.setStyleSheet(
            f"QPushButton {{ background: {C_BG}; color: {C_TEXT}; border: 1px solid {C_BORDER};"
            f"border-radius: 8px; padding: 8px 18px; font-size: 13px; font-weight: 600; }}"
            f"QPushButton:hover {{ border: 1px solid {C_TEALH}; color: {C_TEALH}; }}")
        self.btn_reset.clicked.connect(self._reset)
        bottom.addWidget(self.btn_reset)
        root.addLayout(bottom)
        root.addStretch(1)

        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(host); outer.addWidget(scroll)

        # connect toggles after construction
        for chk in (self.chk_presets, self.chk_disclaimer, self.chk_autosave, self.chk_autodelete):
            chk.stateChanged.connect(self._save)

        # live clock for the chosen zone
        self._loading = False
        self._tick()
        self._timer = QTimer(self); self._timer.timeout.connect(self._tick); self._timer.start(1000)

    # ------------------------------------------------------------------
    def _section(self, heading):
        card = QFrame()
        card.setStyleSheet(f"QFrame {{ background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 12px; }}")
        lay = QVBoxLayout(card); lay.setContentsMargins(20, 16, 20, 18); lay.setSpacing(8)
        hd = QLabel(heading)
        hd.setStyleSheet(f"color: {C_TEAL}; font-size: 15px; font-weight: 800; "
                         f"letter-spacing: 0.5px; background: transparent;")
        lay.addWidget(hd)
        return card, lay

    def _toggle(self, text, on):
        c = QCheckBox(text); c.setChecked(bool(on)); c.setCursor(Qt.PointingHandCursor)
        c.setStyleSheet(_TOGGLE_QSS)
        return c

    def _label(self, text):
        l = QLabel(text)
        l.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; font-weight: 600; background: transparent;")
        return l

    def _hint(self, text):
        l = QLabel(text); l.setWordWrap(True)
        l.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 12px; background: transparent;")
        return l

    def _select_tz(self, tz):
        items = self.tz_list.findItems(tz, Qt.MatchExactly)
        if items:
            self.tz_list.setCurrentItem(items[0])
            self.tz_list.scrollToItem(items[0])

    # ------------------------------------------------------------------
    def _collect(self):
        return {
            "show_workflow_presets": self.chk_presets.isChecked(),
            "letterhead": self.in_lab.text(),
            "pathologist": self.in_path.text(),
            "include_ai_disclaimer": self.chk_disclaimer.isChecked(),
            "auto_save_best": self.chk_autosave.isChecked(),
            "auto_delete_history_30d": self.chk_autodelete.isChecked(),
            "timezone": self.tz_list.currentItem().text() if self.tz_list.currentItem() else "Asia/Kolkata",
        }

    def _save(self, *_):
        if self._loading:
            return
        self.settings = self._collect()
        save_settings(self.settings)
        if self.status_callback:
            self.status_callback("Settings saved")

    def _on_tz_changed(self, _text):
        self._save()
        self._tick()

    def _tick(self):
        tz = self.tz_list.currentItem().text() if self.tz_list.currentItem() else "Asia/Kolkata"
        try:
            now = datetime.now(ZoneInfo(tz)) if _HAS_TZ else datetime.now()
            self.lbl_clock.setText(f"Current time in {tz}:  {now.strftime('%d %b %Y, %H:%M:%S')}")
        except Exception:
            self.lbl_clock.setText(f"Current time in {tz}:  —")

    def _reset(self):
        self._loading = True
        self.chk_presets.setChecked(DEFAULTS["show_workflow_presets"])
        self.in_lab.setText(DEFAULTS["letterhead"])
        self.in_path.setText(DEFAULTS["pathologist"])
        self.chk_disclaimer.setChecked(DEFAULTS["include_ai_disclaimer"])
        self.chk_autosave.setChecked(DEFAULTS["auto_save_best"])
        self.chk_autodelete.setChecked(DEFAULTS["auto_delete_history_30d"])
        self._select_tz(DEFAULTS["timezone"])
        self._loading = False
        self._save()
        self._tick()
        if self.status_callback:
            self.status_callback("Settings reset to defaults")
