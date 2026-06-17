"""
Clinical report builder.

Turns patient details + a selected prediction run into a polished PDF
(reportlab). Also lists recent prediction runs from output/predictions/.

build_report_bytes(data) -> bytes        (used for the live preview)
save_report(data, dest)  -> dest path    (writes the PDF to disk)
list_recent_runs()       -> [record...]  (newest first)
"""

import os
import io
import glob
import json
import time

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

from brain.paths import APP_ROOT
PRED_DIR = os.path.join(APP_ROOT, "output", "predictions")
REPORTS  = os.path.join(APP_ROOT, "reports")

TEAL = colors.HexColor("#238f7a")
DARK = colors.HexColor("#1c2333")
GREY = colors.HexColor("#666666")


def list_recent_runs():
    """Return prediction records (newest first) read from result_*.json sidecars."""
    runs = []
    for p in glob.glob(os.path.join(PRED_DIR, "result_*.json")):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                runs.append(json.load(fh))
        except Exception:
            continue
    runs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return runs


def _styles():
    ss = getSampleStyleSheet()
    return {
        "letter": ParagraphStyle("letter", parent=ss["Title"], fontSize=18,
                                 textColor=TEAL, alignment=TA_CENTER, spaceAfter=2),
        "sub": ParagraphStyle("sub", parent=ss["Normal"], fontSize=9,
                              textColor=GREY, alignment=TA_CENTER, spaceAfter=2),
        "report_title": ParagraphStyle("rt", parent=ss["Heading2"], fontSize=13,
                                       textColor=DARK, alignment=TA_CENTER, spaceBefore=8, spaceAfter=6),
        "section": ParagraphStyle("section", parent=ss["Heading3"], fontSize=11,
                                  textColor=TEAL, spaceBefore=10, spaceAfter=4),
        "normal": ParagraphStyle("n", parent=ss["Normal"], fontSize=10, leading=14),
        "small": ParagraphStyle("s", parent=ss["Normal"], fontSize=8, textColor=GREY),
        "label": ParagraphStyle("lab", parent=ss["Normal"], fontSize=9, textColor=GREY),
        "value": ParagraphStyle("val", parent=ss["Normal"], fontSize=10, textColor=DARK),
    }


def _patient_table(patient, st):
    def cell(label, value):
        return [Paragraph(label, st["label"]), Paragraph(str(value or "—"), st["value"])]
    rows = [
        cell("Patient Name", patient.get("name")) + cell("Patient ID / Lab No.", patient.get("id")),
        cell("Age / Gender", f"{patient.get('age') or '—'} / {patient.get('gender') or '—'}")
            + cell("Sample Type", patient.get("sample_type")),
        cell("Date of Collection", patient.get("collection_date"))
            + cell("Referring Doctor", patient.get("referring_doctor")),
    ]
    t = Table(rows, colWidths=[32 * mm, 55 * mm, 32 * mm, 51 * mm])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#dddddd")),
    ]))
    return t


def _findings_table(findings, st):
    data = [["Class", "Count", "Avg Confidence"]]
    for f in findings:
        data.append([f.get("class", "—"), str(f.get("count", 0)),
                     f"{(f.get('avg_conf', 0) or 0) * 100:.1f}%"])
    if len(data) == 1:
        data.append(["No objects above threshold", "", ""])
    t = Table(data, colWidths=[80 * mm, 35 * mm, 45 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3f6f5")]),
    ]))
    return t


def _build_flow(data):
    st = _styles()
    patient = data.get("patient", {})
    opts = data.get("options", {})
    result = data.get("result")
    flow = []

    # letterhead
    flow.append(Paragraph(data.get("letterhead") or "Diagnostic Centre", st["letter"]))
    flow.append(Paragraph("AI-Assisted Pathology Report", st["sub"]))
    flow.append(Spacer(1, 4))
    flow.append(HRFlowable(width="100%", thickness=1.2, color=TEAL))
    flow.append(Paragraph("DIAGNOSTIC REPORT", st["report_title"]))

    # patient
    flow.append(_patient_table(patient, st))
    flow.append(Spacer(1, 6))

    # clinical query
    flow.append(Paragraph("Clinical Diagnosis / Query", st["section"]))
    flow.append(Paragraph(patient.get("clinical_query") or "—", st["normal"]))

    # AI analysis
    if result:
        flow.append(Paragraph("AI Analysis", st["section"]))
        flow.append(Paragraph(f"Analysed image: <b>{result.get('image_name', '—')}</b>", st["normal"]))

        if opts.get("include_image", True) and result.get("result_image") \
                and os.path.isfile(result["result_image"]):
            flow.append(Spacer(1, 4))
            flow.append(RLImage(result["result_image"], width=120 * mm, height=120 * mm, kind="proportional"))
            flow.append(Spacer(1, 4))

        if opts.get("include_table", True):
            flow.append(Paragraph("Detected Findings", st["section"]))
            flow.append(_findings_table(result.get("findings", []), st))

        if opts.get("include_explanation", True) and result.get("explain_image") \
                and os.path.isfile(result["explain_image"]):
            flow.append(Spacer(1, 6))
            flow.append(Paragraph("AI Explanation (heatmap)", st["section"]))
            flow.append(Paragraph(
                "Warmer (red/yellow) areas show the regions the AI model focused on "
                "when producing this result.", st["small"]))
            flow.append(Spacer(1, 3))
            flow.append(RLImage(result["explain_image"], width=110 * mm, height=110 * mm, kind="proportional"))

        if opts.get("include_model", True):
            flow.append(Spacer(1, 6))
            flow.append(Paragraph(
                f"<b>AI Model:</b> {result.get('model') or '—'} &nbsp;·&nbsp; "
                f"<b>Task:</b> {result.get('task', '—')} &nbsp;·&nbsp; "
                f"<b>Inference time:</b> {result.get('elapsed', '—')}s", st["small"]))
    else:
        flow.append(Paragraph("AI Analysis", st["section"]))
        flow.append(Paragraph("No analysis result selected.", st["normal"]))

    # signature
    flow.append(Spacer(1, 26))
    sig_bits = []
    if opts.get("signature_line", True):
        sig_bits.append(Paragraph("______________________________", st["normal"]))
    sig_bits.append(Paragraph(f"<b>{data.get('pathologist') or 'Pathologist'}</b>", st["normal"]))
    sig_bits.append(Paragraph("Consultant Pathologist", st["small"]))
    sig_tbl = Table([[b] for b in sig_bits], colWidths=[80 * mm], hAlign="RIGHT")
    sig_tbl.setStyle(TableStyle([("TOPPADDING", (0, 0), (-1, -1), 1),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    flow.append(sig_tbl)

    # footer
    flow.append(Spacer(1, 16))
    flow.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    flow.append(Paragraph(
        f"Generated by Dr. Roy DT&R on {time.strftime('%d %b %Y, %H:%M')} · "
        "AI-assisted result — to be reviewed and verified by a qualified pathologist.",
        st["small"]))
    return flow


def build_report_bytes(data):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, title="Dr. Roy DT&R Report",
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm)
    doc.build(_build_flow(data))
    return buf.getvalue()


def save_report(data, dest=None):
    os.makedirs(REPORTS, exist_ok=True)
    if dest is None:
        dest = os.path.join(REPORTS, f"report_{time.strftime('%Y%m%d_%H%M%S')}.pdf")
    with open(dest, "wb") as fh:
        fh.write(build_report_bytes(data))
    return dest
