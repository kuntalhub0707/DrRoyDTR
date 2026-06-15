"""
Colab session store.

A tiny persistent record of cloud-training sessions so the Home dashboard can
show live status even though it's a different page from Train on Colab.

colab_sessions.json is a list of:
  {run_name, date, task, launched_at, status, notebook_url, metric, completed_at}
status is one of: 'running', 'complete', 'timed_out', 'cancelled'.
"""

import os
import json
import time

from brain.paths import APP_ROOT
SESSIONS_FILE = os.path.join(APP_ROOT, "colab_sessions.json")


def load_sessions():
    if not os.path.isfile(SESSIONS_FILE):
        return []
    try:
        with open(SESSIONS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_sessions(items):
    try:
        with open(SESSIONS_FILE, "w", encoding="utf-8") as fh:
            json.dump(items, fh, indent=2)
    except Exception:
        pass


def add_session(run_name, date, task, notebook_url):
    items = [s for s in load_sessions()
             if not (s.get("run_name") == run_name and s.get("date") == date)]
    items.append({
        "run_name": run_name, "date": date, "task": task,
        "launched_at": time.strftime("%Y%m%d_%H%M%S"),
        "status": "running", "notebook_url": notebook_url,
        "metric": None, "completed_at": None,
    })
    save_sessions(items)


def update_session(run_name, date, **fields):
    items = load_sessions()
    for s in items:
        if s.get("run_name") == run_name and s.get("date") == date:
            s.update(fields)
    save_sessions(items)


def completed_count():
    return sum(1 for s in load_sessions() if s.get("status") == "complete")


def _parse(ts):
    try:
        return time.mktime(time.strptime(ts, "%Y%m%d_%H%M%S"))
    except Exception:
        return None


def visible_sessions(recent_hours=24):
    """Running sessions, plus complete/timed_out ones from the last `recent_hours`."""
    now = time.time()
    out = []
    for s in load_sessions():
        st = s.get("status")
        if st == "running":
            out.append(s)
        elif st in ("complete", "timed_out"):
            ref = _parse(s.get("completed_at") or "") or _parse(s.get("launched_at") or "")
            if ref and (now - ref) <= recent_hours * 3600:
                out.append(s)
    # newest first by launch time
    out.sort(key=lambda s: s.get("launched_at", ""), reverse=True)
    return out
