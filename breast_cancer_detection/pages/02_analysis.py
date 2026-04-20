from __future__ import annotations

import streamlit as st
import pandas as pd
from breast_cancer_detection.utils.demo_data import demo_diagnosis_state


def render():
    st.title("⚙️ AI Analysis")
    state = st.session_state.get("diagnosis_state") or {}
    analysis_complete = bool(st.session_state.get("analysis_complete")) or bool(state.get("ensemble_result") is not None)
    if state and state.get("ensemble_result") is not None:
        st.session_state["last_complete_state"] = state
    if (not state) or (not analysis_complete) or (state.get("ensemble_result") is None):
        fallback = st.session_state.get("last_complete_state") or {}
        if fallback and fallback.get("ensemble_result") is not None:
            state = fallback
            st.warning("Showing last completed analysis (latest run still processing or incomplete).")
        else:
            state = demo_diagnosis_state()
            st.info("Showing seeded analysis preview until full pipeline output is available.")
    steps = [
        "🔄 Image Preprocessing",
        "🎯 ROI Segmentation (U-Net)",
        "🌡️ Thermal Analysis",
        "🔬 Feature Extraction",
        "🧠 Deep Learning Inference (EfficientNet + ViT)",
        "🎛️ Ensemble Fusion",
        "📝 Report Generation",
    ]
    progress = 0
    cols = st.columns(len(steps))
    for i, step in enumerate(steps):
        cols[i].markdown(f'<div class="pipeline-step done">{step}</div>', unsafe_allow_html=True)
        progress = int(((i + 1) / len(steps)) * 100)
    st.progress(progress)
    pre = state.get("preprocessed_image") or {}
    if pre:
        c1, c2, c3 = st.columns(3)
        if pre.get("original") is not None:
            c1.image(pre.get("original"), caption="Original", width="stretch")
        if pre.get("enhanced") is not None:
            c2.image(pre.get("enhanced"), caption="Enhanced", width="stretch")
        if pre.get("roi") is not None:
            c3.image(pre.get("roi"), caption="ROI", width="stretch")
        if state.get("segmentation_mask") is not None:
            st.subheader("Segmentation Mask")
            st.image(state.get("segmentation_mask"), caption="Foreground mask used for ROI extraction", width="stretch")
    st.subheader("Pipeline Logs")
    with st.expander("View logs", expanded=True):
        logs = state.get("pipeline_logs", []) or []
        if logs:
            for log in logs:
                st.code(str(log), language="text")
        else:
            st.caption("No logs yet. Start analysis from Upload page.")

    if not analysis_complete:
        seeded = st.session_state.get("patient_history", [])
        if seeded:
            st.subheader("Seeded Analysis Samples (25)")
            st.dataframe(pd.DataFrame(seeded).head(25), width="stretch")
    st.button("Go to Results", on_click=lambda: st.session_state.update({"current_page": "results"}))


if __name__ == "__main__":
    render()

