from __future__ import annotations

import io
import hashlib
from pathlib import Path

import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from breast_cancer_detection.utils.file_store import render_download_button, save_output_bytes, save_output_text
from breast_cancer_detection.utils.logger import get_logger

logger = get_logger("report_viewer")


def render_markdown_report(report_md: str):
    st.markdown(report_md or "No report available.")


def report_download_buttons(report_md: str):
    report_md = report_md or "No report available."
    report_hash = hashlib.sha256(report_md.encode("utf-8")).hexdigest()
    md_path = st.session_state.get("report_md_path")
    pdf_path = st.session_state.get("report_pdf_path")
    previous_hash = st.session_state.get("report_content_hash")
    try:
        if report_hash != previous_hash or not md_path or not Path(md_path).exists():
            md_path = save_output_text(report_md, "oncovision_report.md")
            st.session_state["report_md_path"] = md_path
    except Exception as exc:
        logger.exception("Failed to save markdown report")
        st.error(f"Failed to persist markdown report: {exc}")
        md_path = st.session_state.get("report_md_path")

    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    text = c.beginText(40, 750)
    for line in report_md.splitlines():
        text.textLine(line[:110])
    c.drawText(text)
    c.showPage()
    c.save()
    try:
        if report_hash != previous_hash or not pdf_path or not Path(pdf_path).exists():
            pdf_path = save_output_bytes(pdf_buffer.getvalue(), "oncovision_report.pdf")
            st.session_state["report_pdf_path"] = pdf_path
        st.session_state["report_content_hash"] = report_hash
    except Exception as exc:
        logger.exception("Failed to save pdf report")
        st.error(f"Failed to persist PDF report: {exc}")
        pdf_path = st.session_state.get("report_pdf_path")

    render_download_button(md_path, f"Download .md ({Path(md_path).name if md_path else 'not-ready'})", "text/markdown", "download-report-md")
    render_download_button(pdf_path, f"Download .pdf ({Path(pdf_path).name if pdf_path else 'not-ready'})", "application/pdf", "download-report-pdf")

