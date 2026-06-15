"""
Changelog page.

A version history of Dr. Roy DT&R. Releases are listed newest-first as cards,
each with a version badge, date, and grouped highlights. Add a new dict to
RELEASES to record a future update.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QScrollArea, QSizePolicy,
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

APP_VERSION = "v1.0"

# Newest release first. Each: version, date, title, tag (or ""), sections[(heading, [items])]
RELEASES = [
    {
        "version": "v1.0",
        "date": "12 June 2026",
        "title": "First release",
        "tag": "Latest",
        "summary": "The complete desktop pathology AI workflow — train, predict, report, "
                   "and offload heavy training to a free cloud GPU.",
        "sections": [
            ("Workspace & Dashboard", [
                "Home dashboard with live counters (models, datasets, training runs, reports, Colab runs)",
                "Live Active Colab Sessions panel — in-progress, complete, and timed-out, refreshing every 2 minutes",
                "Sidebar navigation, live clock, and a system status bar",
            ]),
            ("Train your own AI", [
                "Fine-tune object-detection or image-classification models on your own slides",
                "Automatic dataset format detection — Roboflow YOLO, plain YOLO, COCO, and Pascal VOC",
                "Live training with accuracy and error-rate charts, progress and time-remaining",
                "Best model auto-saved, with a full searchable Training History",
            ]),
            ("Cloud training on Google Colab", [
                "Connect Google Drive, upload datasets, and launch a ready-to-run Colab notebook",
                "The app watches Drive and brings the finished model back automatically",
            ]),
            ("Predict & analyze", [
                "Run any model on a single slide or a whole folder of images",
                "Detection boxes, a findings table, and adjustable confidence / overlap settings",
                "Export annotated images, PDF reports, and CSV / XLSX results",
            ]),
            ("Datasets, models & reports", [
                "Dataset Library and a 16-model starter library (YOLO26 and YOLOv8, detection + classification)",
                "Clinical PDF reports with a live preview, patient details, and your letterhead",
                "Settings for report defaults, training defaults, and timezone",
            ]),
        ],
    },
]


class ChangelogPage(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet(f"background: {C_BG};")

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        page = QWidget(); page.setStyleSheet("background: transparent;")
        root = QVBoxLayout(page)
        root.setContentsMargins(40, 30, 40, 36)
        root.setSpacing(8)

        # header
        title_row = QHBoxLayout(); title_row.setSpacing(12)
        title = QLabel("Changelog")
        title.setStyleSheet(f"color: {C_TEXT}; font-size: 28px; font-weight: 800;")
        title_row.addWidget(title, 0)
        badge = QLabel(f"Current: {APP_VERSION}")
        badge.setStyleSheet(
            f"background: {C_PANEL2}; color: {C_TEALH}; border: 1px solid {C_BORDER};"
            f"border-radius: 11px; padding: 3px 12px; font-size: 12px; font-weight: 700;")
        badge.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        title_row.addWidget(badge, 0, Qt.AlignVCenter)
        title_row.addStretch(1)
        root.addLayout(title_row)

        sub = QLabel("Everything that's new in Dr. Roy DT&R.")
        sub.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 14px;")
        root.addWidget(sub)
        root.addSpacing(14)

        for rel in RELEASES:
            root.addWidget(self._release_card(rel))
            root.addSpacing(16)

        footer = QLabel("Dr. Roy — Data Training & Reporting · Built for Dr. Kuntal Roy")
        footer.setStyleSheet(f"color: {C_BORDER}; font-size: 11px;")
        root.addWidget(footer)
        root.addStretch(1)

        scroll.setWidget(page)
        wrap = QVBoxLayout(self)
        wrap.setContentsMargins(0, 0, 0, 0)
        wrap.addWidget(scroll)

    # ------------------------------------------------------------------
    def _release_card(self, rel):
        card = QFrame()
        card.setStyleSheet(
            f"QFrame#rel {{ background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 12px; }}")
        card.setObjectName("rel")
        outer = QHBoxLayout(card)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # left accent rail with the version
        rail = QFrame()
        rail.setFixedWidth(96)
        rail.setStyleSheet(
            f"background: {C_PANEL}; border-top-left-radius: 12px; border-bottom-left-radius: 12px;"
            f"border-right: 2px solid {C_TEAL};")
        rl = QVBoxLayout(rail)
        rl.setContentsMargins(14, 18, 14, 18)
        rl.setSpacing(4)
        ver = QLabel(rel["version"])
        ver.setStyleSheet(f"color: {C_TEALH}; font-size: 20px; font-weight: 800; background: transparent;")
        rl.addWidget(ver)
        date = QLabel(rel["date"])
        date.setWordWrap(True)
        date.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px; background: transparent;")
        rl.addWidget(date)
        rl.addStretch(1)
        outer.addWidget(rail, 0)

        # body
        body = QVBoxLayout()
        body.setContentsMargins(22, 18, 22, 20)
        body.setSpacing(8)

        head = QHBoxLayout(); head.setSpacing(10)
        t = QLabel(rel["title"])
        t.setStyleSheet(f"color: {C_TEXT}; font-size: 18px; font-weight: 800; background: transparent;")
        head.addWidget(t, 0)
        if rel.get("tag"):
            tag = QLabel(rel["tag"])
            tag.setStyleSheet(
                f"background: {C_BLUE}; color: #fff; border-radius: 9px; padding: 2px 10px;"
                f"font-size: 11px; font-weight: 700;")
            tag.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
            head.addWidget(tag, 0, Qt.AlignVCenter)
        head.addStretch(1)
        body.addLayout(head)

        if rel.get("summary"):
            s = QLabel(rel["summary"]); s.setWordWrap(True)
            s.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px; background: transparent;")
            body.addWidget(s)
            body.addSpacing(4)

        for heading, items in rel["sections"]:
            h = QLabel(heading)
            h.setStyleSheet(f"color: {C_TEAL}; font-size: 13px; font-weight: 800; "
                            f"letter-spacing: 0.3px; background: transparent; padding-top: 6px;")
            body.addWidget(h)
            for it in items:
                row = QHBoxLayout(); row.setSpacing(8); row.setContentsMargins(2, 0, 0, 0)
                dot = QLabel("•")
                dot.setStyleSheet(f"color: {C_TEALH}; font-size: 14px; background: transparent;")
                dot.setFixedWidth(12)
                txt = QLabel(it); txt.setWordWrap(True)
                txt.setStyleSheet(f"color: {C_TEXT}; font-size: 13px; background: transparent;")
                row.addWidget(dot, 0, Qt.AlignTop)
                row.addWidget(txt, 1)
                body.addLayout(row)

        body_wrap = QWidget(); body_wrap.setStyleSheet("background: transparent;")
        body_wrap.setLayout(body)
        outer.addWidget(body_wrap, 1)
        return card
