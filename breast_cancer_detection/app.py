from __future__ import annotations

import os
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

# Ensure package imports work even when launched from inside breast_cancer_detection/.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from breast_cancer_detection.pages import (
    _01_upload as upload_page,
    _02_analysis as analysis_page,
    _03_results as results_page,
    _04_dashboard as dashboard_page,
    _05_chat as chat_page,
)
from breast_cancer_detection.utils.config import DEFAULT_STATE, ROOT_DIR
from breast_cancer_detection.utils.background_jobs import get_job
from breast_cancer_detection.utils.demo_data import demo_diagnosis_state, generate_seeded_history
from breast_cancer_detection.utils.file_store import cleanup_old_files, ensure_storage_dirs
from breast_cancer_detection.utils.logging_config import setup_logging
from breast_cancer_detection.utils.logger import get_logger
from breast_cancer_detection.utils.model_sync import sync_models_from_zip, validate_model_artifacts

PAGE_KEYS = {"app", "upload", "analysis", "results", "dashboard", "chat"}


def init_session():
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "app_initialized" not in st.session_state:
        st.session_state["current_page"] = "app"
        st.session_state["app_initialized"] = True
    st.session_state.setdefault("report_md_path", None)
    st.session_state.setdefault("report_pdf_path", None)
    st.session_state.setdefault("report_content_hash", None)
    st.session_state.setdefault("chat_txt_path", None)
    st.session_state.setdefault("chat_transcript_hash", None)
    st.session_state.setdefault("file_cleanup_last_run", 0.0)


def _load_css():
    css_path = ROOT_DIR / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _load_env():
    env_path = ROOT_DIR.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        os.environ.setdefault(key, value)


def _model_status():
    status = validate_model_artifacts()
    if not all(status.values()):
        sync_models_from_zip()
        status = validate_model_artifacts()
    return status


def _render_top_nav() -> str:
    current = st.session_state.get("current_page", "app")
    if current not in PAGE_KEYS:
        current = "app"
    model_status = "Ready" if st.session_state.get("models_loaded") else "Loading"
    llm_status = "Connected" if st.session_state.get("ollama_connected") else "Offline"
    analysis_job = get_job(st.session_state.get("analysis_job_id"))
    chat_job = get_job(st.session_state.get("chat_active_job_id"))
    analysis_badge = "Running" if analysis_job and analysis_job.get("status") == "running" else "Idle"
    chat_badge = "Running" if chat_job and chat_job.get("status") == "running" else "Idle"
    nav_items = [("app", "App"), ("upload", "Upload"), ("analysis", "Analysis"), ("results", "Results"), ("dashboard", "Dashboard"), ("chat", "Chat")]
    left, right = st.columns([5, 5], vertical_alignment="center")
    with left:
        st.markdown("<div class='brand-title'>OncoVision AI</div>", unsafe_allow_html=True)
        st.markdown("<div class='brand-subtitle'>Clinical multi-modal screening workspace</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='top-nav-compact'>", unsafe_allow_html=True)
        cols = st.columns([1, 1, 1, 1, 1.1, 1])
        for col, (key, label) in zip(cols, nav_items):
            if col.button(label, key=f"top-nav-btn-{key}", use_container_width=True, type=("primary" if current == key else "secondary")):
                st.session_state["current_page"] = key
                current = key
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        (
            "<div class='main-top-status'>"
            f"<span class='nav-pill'>Models: {model_status}</span>"
            f"<span class='nav-pill'>LLM: {llm_status}</span>"
            f"<span class='nav-pill'>Analysis: {analysis_badge}</span>"
            f"<span class='nav-pill'>Chat: {chat_badge}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )
    st.session_state["current_page"] = current
    return current


def _render_overview():
    st.markdown('<div class="hero-panel">', unsafe_allow_html=True)
    c1, c2 = st.columns([1.7, 1.3])
    with c1:
        st.markdown("### Clinical AI Workspace")
        st.markdown(
            "Use **Upload & Input** for real cases. Use **Load Demo** for a fully populated live preview "
            "of Analysis, Results, Dashboard, and Chat."
        )
        b1, b2 = st.columns(2)
        if b1.button("Load 25-point Demo", width="stretch"):
            demo = demo_diagnosis_state()
            st.session_state["diagnosis_state"] = demo
            st.session_state["analysis_complete"] = True
            st.session_state["patient_history"] = demo["seeded_history"]
            st.success("Demo workspace initialized.")
        if b2.button("Clear Current Session", width="stretch"):
            st.session_state["diagnosis_state"] = None
            st.session_state["last_complete_state"] = None
            st.session_state["analysis_complete"] = False
            st.session_state["chat_history"] = []
            st.info("Session reset complete.")
    with c2:
        history = st.session_state.get("patient_history") or generate_seeded_history(25, seed=42)
        risk_series = [h["risk_score"] for h in history]
        fig = go.Figure(go.Scatter(y=risk_series, mode="lines+markers", line={"color": "#00D4FF"}))
        fig.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=30, b=10), title="Live Risk Trend")
        st.plotly_chart(fig, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)


def _sync_background_analysis_state() -> None:
    """Promote completed background analysis result to canonical session state from any page."""
    logger = get_logger("app")
    analysis_job = get_job(st.session_state.get("analysis_job_id"))
    if not analysis_job:
        return

    status = analysis_job.get("status")
    if status == "done":
        result = analysis_job.get("result")
        logger.info("ANALYSIS_JOB done | has_result=%s", isinstance(result, dict))
        if isinstance(result, dict) and result.get("ensemble_result") is not None:
            st.session_state["diagnosis_state"] = result
            st.session_state["last_complete_state"] = result
            st.session_state["analysis_complete"] = True
            ensemble = result.get("ensemble_result") or {}
            history = st.session_state.get("patient_history") or []
            history.append(
                {
                    "idx": len(history) + 1,
                    "date": "Current",
                    "risk_score": float(ensemble.get("risk_score", 0.0)),
                    "confidence": float(ensemble.get("confidence", 0.0)),
                    "reliability": float(ensemble.get("reliability", 0.0)),
                    "risk_level": ensemble.get("risk_level", "Unknown"),
                }
            )
            st.session_state["patient_history"] = history[-25:]
            logger.info("ANALYSIS_JOB state promoted | prediction=%s", ensemble.get("prediction"))
        else:
            logger.warning("ANALYSIS_JOB result missing ensemble_result.")
        st.session_state["analysis_job_id"] = None
    elif status == "failed":
        st.session_state["analysis_complete"] = False
        st.session_state["analysis_job_error"] = analysis_job.get("error", "Unknown analysis error")
        st.session_state["analysis_job_id"] = None


def main():
    setup_logging("streamlit_app")
    st.set_page_config(page_title="OncoVision AI", page_icon="🔬", layout="wide", initial_sidebar_state="collapsed")
    _load_env()
    ensure_storage_dirs()
    init_session()
    statuses = _model_status()
    st.session_state["models_loaded"] = all(statuses.values())
    st.session_state["ollama_connected"] = True
    _sync_background_analysis_state()
    now = __import__("time").time()
    last_cleanup = float(st.session_state.get("file_cleanup_last_run", 0.0))
    if now - last_cleanup > 1800:
        cleanup_old_files(ROOT_DIR / "uploads", max_age=86400)
        cleanup_old_files(ROOT_DIR / "outputs", max_age=86400)
        st.session_state["file_cleanup_last_run"] = now
    _load_css()
    nav_page = _render_top_nav()

    if st.session_state.get("analysis_job_error"):
        st.error(f"Analysis failed: {st.session_state.get('analysis_job_error')}")

    if nav_page == "app":
        st.title("OncoVision AI")
        st.caption("Production-grade multi-modal breast screening workspace")
        _render_overview()
        st.info("Use the top-right navigation to open Upload, Analysis, Results, Dashboard, or Chat.")
    elif nav_page == "upload":
        st.session_state["current_page"] = "upload"
        upload_page.render()
    elif nav_page == "analysis":
        st.session_state["current_page"] = "analysis"
        analysis_page.render()
    elif nav_page == "results":
        st.session_state["current_page"] = "results"
        results_page.render()
    elif nav_page == "dashboard":
        st.session_state["current_page"] = "dashboard"
        dashboard_page.render()
    else:
        st.session_state["current_page"] = "chat"
        chat_page.render()


if __name__ == "__main__":
    main()

