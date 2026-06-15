"""
Training History page.

Reads training_history.json (written by brain/trainer.py) and shows every run in
a filterable, sortable table. Click a row (or 'View Details') to see that run's
accuracy/loss charts + settings log. 'Re-run' restarts a run with the same
settings via the Train Model page.

Constructed with:
  navigate(index)        -> jump to another page (empty-state link, re-run)
  rerun_callback(entry)  -> re-run a history entry (provided by MainWindow)

Stack index of Train Model = 1.
"""

import os
import csv
import json
import shutil
from datetime import datetime, timedelta

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFrame, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSizePolicy,
    QDialog, QMessageBox, QScrollArea,
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- palette ---
C_BG        = "#0d1117"
C_PANEL     = "#161b27"
C_PANEL2    = "#1c2333"
C_BORDER    = "#2a3045"
C_TEXT      = "#e6edf3"
C_TEXT_DIM  = "#8b949e"
C_TEAL      = "#238f7a"
C_TEALH     = "#2aad93"
C_BLUE      = "#3b82f6"
C_GREEN     = "#3fb950"
C_RED       = "#e06c75"
C_AMBER     = "#e5c07b"

from brain.paths import APP_ROOT
HISTORY  = os.path.join(APP_ROOT, "training_history.json")
RUNS_DIR = os.path.join(APP_ROOT, "output", "training_runs")
MODELS_DIR = os.path.join(APP_ROOT, "models")

IDX_TRAIN_MODEL = 1

STATUS_COLOR = {
    "Completed": C_GREEN,
    "Completed on Colab": C_GREEN,
    "Failed": C_RED,
    "Stopped": C_AMBER,
    "Cancelled": C_AMBER,
    "Running": C_BLUE,
}
METRIC_STATUSES = ("Completed", "Completed on Colab", "Stopped")

COLUMNS = ["Run Name", "Task", "Dataset", "Status", "Rounds",
           "Best Accuracy", "Started", "Duration", "Actions"]


def _fmt_duration(secs):
    secs = int(secs or 0)
    m, s = divmod(secs, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _fmt_started(ts):
    try:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S").strftime("%d %b %Y, %H:%M")
    except Exception:
        return ts or "—"


def _load_history():
    if not os.path.isfile(HISTORY):
        return []
    try:
        with open(HISTORY, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_history(entries):
    with open(HISTORY, "w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)


def _run_dir(entry):
    return os.path.join(RUNS_DIR, f"train_{entry.get('task','detection')}_{entry.get('timestamp','')}")


# ----------------------------------------------------------------------
# Details dialog
# ----------------------------------------------------------------------
class _StaticChart(FigureCanvas):
    def __init__(self, title, ylabel, xs, ys, color):
        fig = Figure(figsize=(4.4, 2.6), facecolor=C_PANEL2)
        super().__init__(fig)
        ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.16, right=0.97, top=0.85, bottom=0.2)
        ax.set_facecolor(C_BG)
        ax.set_title(title, color=C_TEXT, fontsize=11, fontweight="bold")
        ax.set_xlabel("Round", color=C_TEXT_DIM, fontsize=8)
        ax.set_ylabel(ylabel, color=C_TEXT_DIM, fontsize=8)
        for sp in ax.spines.values():
            sp.set_color(C_BORDER)
        ax.tick_params(colors=C_TEXT_DIM, labelsize=7)
        ax.grid(True, color=C_BORDER, linewidth=0.4, alpha=0.5)
        if xs and ys:
            ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=3)
        else:
            ax.text(0.5, 0.5, "No per-round data", color=C_TEXT_DIM,
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
        self.draw_idle()


def _read_results_csv(entry):
    """Return (rounds, accuracy[], loss[]) from the run's results.csv if present."""
    path = os.path.join(_run_dir(entry), "results.csv")
    if not os.path.isfile(path):
        return [], [], []
    rounds, acc, loss = [], [], []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            cols = [c.strip() for c in (reader.fieldnames or [])]
            def find(*cands):
                for c in cands:
                    if c in cols:
                        return c
                return None
            acc_col = find("metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/accuracy_top1")
            loss_cols = [c for c in cols if c.startswith("val/") and "loss" in c] or \
                        [c for c in cols if c.startswith("train/") and "loss" in c]
            ep_col = find("epoch")
            for i, raw in enumerate(reader, start=1):
                row = {k.strip(): v for k, v in raw.items()}
                ep = int(float(row.get(ep_col, i))) if ep_col else i
                rounds.append(ep + 1 if ep_col and ep == 0 else ep if ep_col else i)
                try:
                    acc.append(float(row.get(acc_col, 0)) if acc_col else 0.0)
                except Exception:
                    acc.append(0.0)
                try:
                    loss.append(sum(float(row.get(c, 0) or 0) for c in loss_cols))
                except Exception:
                    loss.append(0.0)
    except Exception:
        return [], [], []
    return rounds, acc, loss


class RunDetailsDialog(QDialog):
    def __init__(self, entry, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run Details — " + (entry.get("model") or entry.get("timestamp", "")))
        self.resize(820, 640)
        self.setStyleSheet(f"QDialog {{ background: {C_BG}; }}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        body = QWidget(); body.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(body)
        lay.setContentsMargins(24, 22, 24, 22)
        lay.setSpacing(14)

        title = QLabel("📊  " + (entry.get("model") or f"Run {entry.get('timestamp','')}"))
        title.setStyleSheet(f"color: {C_TEXT}; font-size: 20px; font-weight: 800;")
        lay.addWidget(title)

        # summary grid
        summ = QFrame()
        summ.setStyleSheet(f"QFrame {{ background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 10px; }}")
        sg = QVBoxLayout(summ); sg.setContentsMargins(18, 14, 18, 14); sg.setSpacing(6)
        best = entry.get("metric", 0) or 0
        rows = [
            ("Task", "Object Detection" if entry.get("task") == "detection" else "Image Classification"),
            ("Dataset", entry.get("dataset") or "—"),
            ("Status", entry.get("status", "—")),
            ("Rounds", f"{entry.get('epochs', 0)} / {entry.get('requested_epochs', entry.get('epochs', 0))}"),
            ("Best Accuracy", f"{best*100:.1f}%"),
            ("Model Size", entry.get("model_size", "—")),
            ("Image Size", str(entry.get("img_size", "—"))),
            ("Started", _fmt_started(entry.get("timestamp", ""))),
            ("Duration", _fmt_duration(entry.get("duration_sec"))),
            ("Saved Model", entry.get("model") or "(not saved)"),
        ]
        for k, v in rows:
            r = QHBoxLayout()
            kl = QLabel(k); kl.setFixedWidth(140)
            kl.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 13px; background: transparent;")
            vl = QLabel(str(v))
            vl.setStyleSheet(f"color: {C_TEXT}; font-size: 13px; font-weight: 600; background: transparent;")
            r.addWidget(kl, 0); r.addWidget(vl, 1)
            sg.addLayout(r)
        lay.addWidget(summ)

        # charts
        rounds, acc, loss = _read_results_csv(entry)
        charts = QHBoxLayout(); charts.setSpacing(14)
        acc_label = "Top-1 Accuracy" if entry.get("task") == "classification" else "mAP@50-95"
        charts.addWidget(_StaticChart("Accuracy per Round", acc_label, rounds, acc, C_TEALH))
        charts.addWidget(_StaticChart("Error Rate per Round", "loss", rounds, loss, C_RED))
        lay.addLayout(charts)

        # log / settings
        log_hdr = QLabel("Run settings & log")
        log_hdr.setStyleSheet(f"color: {C_TEXT}; font-size: 14px; font-weight: 700;")
        lay.addWidget(log_hdr)
        args_path = os.path.join(_run_dir(entry), "args.yaml")
        if os.path.isfile(args_path):
            try:
                with open(args_path, "r", encoding="utf-8") as fh:
                    text = fh.read()
            except Exception:
                text = "(could not read run settings)"
        else:
            text = ("Per-run log files are kept in:\n  " + _run_dir(entry) +
                    "\n(They appear once a run has been executed on this computer.)")
        log = QLabel(text)
        log.setWordWrap(True)
        log.setStyleSheet(
            f"color: {C_TEXT_DIM}; font-size: 11px; font-family: Consolas, monospace; "
            f"background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 8px; padding: 12px;")
        lay.addWidget(log)

        scroll.setWidget(body)
        outer.addWidget(scroll)

        close = QPushButton("Close")
        close.setCursor(Qt.PointingHandCursor)
        close.setStyleSheet(
            f"QPushButton {{ background: {C_TEAL}; color: #fff; border: none; border-radius: 8px; "
            f"padding: 10px 20px; font-weight: 700; }} QPushButton:hover {{ background: {C_TEALH}; }}")
        close.clicked.connect(self.accept)
        row = QHBoxLayout(); row.addStretch(1); row.addWidget(close)
        outer.addLayout(row)


# ----------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------
class TrainingHistoryPage(QWidget):
    def __init__(self, navigate=None, rerun_callback=None):
        super().__init__()
        self.navigate = navigate or (lambda _i: None)
        self.rerun_callback = rerun_callback
        self._all = []
        self.setStyleSheet(f"background: {C_BG};")

        root = QVBoxLayout(self)
        root.setContentsMargins(40, 28, 40, 28)
        root.setSpacing(10)

        # ---- header row ----
        head = QHBoxLayout()
        left = QVBoxLayout(); left.setSpacing(2)
        title = QLabel("🕒  Training History")
        title.setStyleSheet(f"color: {C_TEXT}; font-size: 26px; font-weight: 800;")
        left.addWidget(title)
        head.addLayout(left, 1)

        self.btn_cleanup = QPushButton("🗑  Delete runs older than 30 days")
        self.btn_cleanup.setCursor(Qt.PointingHandCursor)
        self.btn_cleanup.setStyleSheet(
            f"QPushButton {{ background: {C_BG}; color: {C_TEXT_DIM}; border: 1px solid {C_BORDER}; "
            f"border-radius: 8px; padding: 9px 14px; font-size: 13px; }}"
            f"QPushButton:hover {{ border: 1px solid {C_RED}; color: {C_RED}; }}")
        self.btn_cleanup.clicked.connect(self._delete_old)
        head.addWidget(self.btn_cleanup, 0, Qt.AlignTop)
        root.addLayout(head)

        desc = QLabel("Every run appears here — completed, failed, cancelled, or still running. "
                      "Click any row to see its charts and replay. Re-run any session with the same settings.")
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 14px;")
        root.addWidget(desc)

        # ---- filter bar ----
        self.filter_bar = self._build_filter_bar()
        root.addWidget(self.filter_bar)

        # ---- empty state ----
        self.empty = self._build_empty_state()
        root.addWidget(self.empty, 1)

        # ---- table ----
        self.table = self._build_table()
        root.addWidget(self.table, 1)

        self.reload_from_disk()

    # ------------------------------------------------------------------
    def _combo(self, items):
        cb = QComboBox()
        cb.addItems(items)
        cb.setStyleSheet(f"""
            QComboBox {{ background: {C_BG}; color: {C_TEXT}; border: 1px solid {C_BORDER};
                border-radius: 7px; padding: 6px 10px; font-size: 13px; min-width: 130px; }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox QAbstractItemView {{ background: {C_PANEL2}; color: {C_TEXT};
                selection-background-color: {C_TEAL}; border: 1px solid {C_BORDER}; outline: none; }}
        """)
        cb.currentIndexChanged.connect(self._apply_filters)
        return cb

    def _build_filter_bar(self):
        bar = QFrame()
        bar.setStyleSheet(f"QFrame {{ background: {C_PANEL2}; border: 1px solid {C_BORDER}; border-radius: 10px; }}")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(16)

        def labelled(text, combo):
            box = QVBoxLayout(); box.setSpacing(3)
            l = QLabel(text); l.setStyleSheet(f"color: {C_TEXT_DIM}; font-size: 11px; font-weight: 700; background: transparent;")
            box.addWidget(l); box.addWidget(combo)
            return box

        self.cmb_task = self._combo(["All", "Detection", "Classification"])
        self.cmb_status = self._combo(["All", "Completed", "Failed", "Running"])
        self.cmb_dataset = self._combo(["All"])
        self.cmb_sort = self._combo(["Most Recent", "Oldest", "Best Accuracy"])

        lay.addLayout(labelled("Task", self.cmb_task))
        lay.addLayout(labelled("Status", self.cmb_status))
        lay.addLayout(labelled("Dataset", self.cmb_dataset))
        lay.addLayout(labelled("Sort By", self.cmb_sort))
        lay.addStretch(1)
        return bar

    def _build_empty_state(self):
        wrap = QFrame()
        wrap.setStyleSheet("background: transparent;")
        lay = QVBoxLayout(wrap)
        lay.addStretch(1)
        icon = QLabel("🗄️")
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet("font-size: 56px; background: transparent;")
        lay.addWidget(icon)
        msg = QLabel("No training runs yet")
        msg.setAlignment(Qt.AlignCenter)
        msg.setStyleSheet(f"color: {C_TEXT}; font-size: 18px; font-weight: 700; background: transparent;")
        lay.addWidget(msg)
        link = QPushButton("Start a training run →")
        link.setCursor(Qt.PointingHandCursor)
        link.setFlat(True)
        link.setStyleSheet(
            f"QPushButton {{ color: {C_BLUE}; font-size: 14px; font-weight: 600; background: transparent; border: none; }}"
            f"QPushButton:hover {{ text-decoration: underline; }}")
        link.clicked.connect(lambda: self.navigate(IDX_TRAIN_MODEL))
        row = QHBoxLayout(); row.addStretch(1); row.addWidget(link); row.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(2)
        return wrap

    def _build_table(self):
        t = QTableWidget(0, len(COLUMNS))
        t.setHorizontalHeaderLabels(COLUMNS)
        t.verticalHeader().setVisible(False)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionMode(QAbstractItemView.SingleSelection)
        t.setShowGrid(False)
        t.cellClicked.connect(self._on_cell_clicked)
        t.setStyleSheet(f"""
            QTableWidget {{ background: {C_PANEL2}; color: {C_TEXT}; border: 1px solid {C_BORDER};
                border-radius: 10px; gridline-color: {C_BORDER}; font-size: 13px; }}
            QHeaderView::section {{ background: {C_PANEL}; color: {C_TEXT_DIM}; padding: 10px 8px;
                border: none; border-bottom: 1px solid {C_BORDER}; font-weight: 700; font-size: 12px; }}
            QTableWidget::item {{ padding: 8px; border-bottom: 1px solid {C_BORDER}; }}
            QTableWidget::item:selected {{ background: {C_BG}; color: {C_TEXT}; }}
        """)
        hdr = t.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Stretch)
        act = COLUMNS.index("Actions")
        hdr.setSectionResizeMode(act, QHeaderView.Fixed)
        t.setColumnWidth(act, 210)
        return t

    # ------------------------------------------------------------------
    def showEvent(self, event):
        self.reload_from_disk()
        super().showEvent(event)

    def reload_from_disk(self):
        self._all = _load_history()
        # repopulate dataset filter, preserving selection
        cur = self.cmb_dataset.currentText() if self.cmb_dataset.count() else "All"
        datasets = sorted({(e.get("dataset") or "").strip() for e in self._all if (e.get("dataset") or "").strip()})
        self.cmb_dataset.blockSignals(True)
        self.cmb_dataset.clear()
        self.cmb_dataset.addItems(["All"] + datasets)
        idx = self.cmb_dataset.findText(cur)
        self.cmb_dataset.setCurrentIndex(idx if idx >= 0 else 0)
        self.cmb_dataset.blockSignals(False)
        self._apply_filters()

    def _apply_filters(self):
        entries = list(self._all)

        task = self.cmb_task.currentText()
        if task != "All":
            want = "detection" if task == "Detection" else "classification"
            entries = [e for e in entries if e.get("task") == want]

        status = self.cmb_status.currentText()
        if status != "All":
            entries = [e for e in entries if e.get("status") == status]

        ds = self.cmb_dataset.currentText()
        if ds != "All":
            entries = [e for e in entries if (e.get("dataset") or "").strip() == ds]

        sort = self.cmb_sort.currentText()
        if sort == "Most Recent":
            entries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
        elif sort == "Oldest":
            entries.sort(key=lambda e: e.get("timestamp", ""))
        elif sort == "Best Accuracy":
            entries.sort(key=lambda e: e.get("metric", 0) or 0, reverse=True)

        has_any = bool(self._all)
        self.empty.setVisible(not has_any)
        self.filter_bar.setVisible(has_any)
        self.table.setVisible(has_any)
        if has_any:
            self._fill_table(entries)

    def _fill_table(self, entries):
        self.table.setRowCount(0)
        for e in entries:
            r = self.table.rowCount()
            self.table.insertRow(r)
            run_name = e.get("model") or f"train_{e.get('task','')}_{e.get('timestamp','')}"
            best = e.get("metric", 0) or 0
            status = e.get("status", "—")
            cells = [
                run_name,
                "Detection" if e.get("task") == "detection" else "Classification",
                e.get("dataset") or "—",
                status,
                f"{e.get('epochs', 0)}/{e.get('requested_epochs', e.get('epochs', 0))}",
                f"{best*100:.1f}%" if status in METRIC_STATUSES else "—",
                _fmt_started(e.get("timestamp", "")),
                _fmt_duration(e.get("duration_sec")),
            ]
            for c, val in enumerate(cells):
                item = QTableWidgetItem(str(val))
                if COLUMNS[c] == "Status":
                    item.setForeground(__import__("PyQt5.QtGui", fromlist=["QColor"]).QColor(
                        STATUS_COLOR.get(status, C_TEXT)))
                self.table.setItem(r, c, item)
            self.table.setCellWidget(r, len(COLUMNS) - 1, self._actions_widget(e))
            self.table.setRowHeight(r, 46)

    def _actions_widget(self, entry):
        w = QWidget(); w.setStyleSheet("background: transparent;")
        lay = QHBoxLayout(w); lay.setContentsMargins(6, 4, 6, 4); lay.setSpacing(6)
        b_view = QPushButton("View Details")
        b_rerun = QPushButton("Re-run")
        b_view.setCursor(Qt.PointingHandCursor); b_rerun.setCursor(Qt.PointingHandCursor)
        b_view.setStyleSheet(
            f"QPushButton {{ background: {C_BG}; color: {C_TEXT}; border: 1px solid {C_BORDER};"
            f"border-radius: 6px; padding: 5px 10px; font-size: 12px; }}"
            f"QPushButton:hover {{ border: 1px solid {C_TEAL}; }}")
        b_rerun.setStyleSheet(
            f"QPushButton {{ background: {C_TEAL}; color: #fff; border: none;"
            f"border-radius: 6px; padding: 5px 12px; font-size: 12px; font-weight: 700; }}"
            f"QPushButton:hover {{ background: {C_TEALH}; }}")
        b_view.clicked.connect(lambda _c, e=entry: self._view_details(e))
        b_rerun.clicked.connect(lambda _c, e=entry: self._rerun(e))
        lay.addWidget(b_view); lay.addWidget(b_rerun)
        return w

    # ------------------------------------------------------------------
    def _on_cell_clicked(self, row, col):
        if col == len(COLUMNS) - 1:
            return  # action buttons handle themselves
        name_item = self.table.item(row, 0)
        if not name_item:
            return
        # find the matching entry by run name + started
        run_name = name_item.text()
        started = self.table.item(row, COLUMNS.index("Started")).text()
        for e in self._all:
            this_name = e.get("model") or f"train_{e.get('task','')}_{e.get('timestamp','')}"
            if this_name == run_name and _fmt_started(e.get("timestamp", "")) == started:
                self._view_details(e)
                return

    def _view_details(self, entry):
        dlg = RunDetailsDialog(entry, self)
        dlg.exec_()

    def _rerun(self, entry):
        if self.rerun_callback:
            self.rerun_callback(entry)
        else:
            self.navigate(IDX_TRAIN_MODEL)

    def _delete_old(self):
        if not self._all:
            return
        cutoff = datetime.now() - timedelta(days=30)
        old = []
        for e in self._all:
            try:
                when = datetime.strptime(e.get("timestamp", ""), "%Y%m%d_%H%M%S")
                if when < cutoff:
                    old.append(e)
            except Exception:
                continue
        if not old:
            QMessageBox.information(self, "Nothing to delete", "There are no runs older than 30 days.")
            return
        resp = QMessageBox.question(
            self, "Delete old runs",
            f"Delete {len(old)} run(s) older than 30 days?\n\n"
            "This removes their history entries, saved models and run folders. This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if resp != QMessageBox.Yes:
            return
        keep = [e for e in self._all if e not in old]
        for e in old:
            # remove saved model + run dir
            if e.get("model"):
                mp = os.path.join(MODELS_DIR, e["model"])
                if os.path.isfile(mp):
                    try: os.remove(mp)
                    except Exception: pass
            rd = _run_dir(e)
            if os.path.isdir(rd):
                shutil.rmtree(rd, ignore_errors=True)
        _save_history(keep)
        self.reload_from_disk()
        QMessageBox.information(self, "Done", f"Deleted {len(old)} old run(s).")
