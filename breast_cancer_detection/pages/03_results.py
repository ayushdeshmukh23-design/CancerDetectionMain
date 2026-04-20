from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from breast_cancer_detection.components.charts import confidence_gauge, probability_bars
from breast_cancer_detection.components.heatmap_viewer import show_gradcam_base64
from breast_cancer_detection.components.report_viewer import render_markdown_report, report_download_buttons
from breast_cancer_detection.utils.demo_data import demo_diagnosis_state


def render():
    st.title("📊 Results & Report")
    state = st.session_state.get("diagnosis_state") or {}
    analysis_complete = bool(st.session_state.get("analysis_complete", False)) or bool(state.get("ensemble_result") is not None)
    if state and state.get("ensemble_result") is not None:
        st.session_state["last_complete_state"] = state
    if (not state) or (not analysis_complete) or (state.get("ensemble_result") is None):
        fallback = st.session_state.get("last_complete_state") or {}
        if fallback and fallback.get("ensemble_result") is not None:
            state = fallback
            st.warning("Showing last completed results (latest run still processing or incomplete).")
        else:
            demo = demo_diagnosis_state()
            if not analysis_complete:
                st.info("Showing seeded demo results until analysis completes.")
            else:
                st.warning("Analysis output incomplete; showing seeded demo results.")
            state = demo

    result = state.get("ensemble_result") or {}
    pred = result.get("prediction", "Unknown")
    conf = result.get("confidence", 0.0)
    reliability = result.get("reliability", 0.0)
    risk = result.get("risk_level", "Unknown")
    risk_score = result.get("risk_score", 0.0)
    probs = result.get("probabilities", {"Normal": 0.0, "Benign": 0.0, "Malignant": 0.0})

    st.markdown(f"## Diagnosis: **{pred}**")
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(confidence_gauge(conf, "Confidence"), width="stretch")
    c2.metric("Reliability", f"{reliability:.1%}")
    c3.metric("Risk Level", f"{risk} ({risk_score:.1f}/100)")

    st.subheader("Confidence Breakdown")
    st.plotly_chart(probability_bars(probs), width="stretch")

    st.subheader("Grad-CAM Heatmap")
    col1, col2, col3 = st.columns(3)
    pre = state.get("preprocessed_image") or {}
    original = pre.get("original")
    if original is not None:
        col1.image(original, caption="Original", width="stretch")
    else:
        col1.info("Original image unavailable for this run.")
    show_gradcam_base64(state.get("gradcam_heatmap"), "Heatmap Overlay")
    seg = state.get("segmentation_mask")
    if seg is not None:
        col3.image(seg, caption="Segmentation Mask", width="stretch")
    else:
        col3.info("Segmentation mask unavailable.")

    if state.get("shap_values"):
        st.subheader("SHAP Explanation")
        st.plotly_chart(go.Figure(state["shap_values"]), width="stretch")
        top_features = (result.get("xai") or {}).get("top_features", [])
        if top_features:
            st.markdown("**Top SHAP-supported factors:** " + ", ".join(top_features))
        st.caption(
            "SHAP bars indicate relative contribution magnitude of engineered thermal/image features "
            "to the model's risk estimate for this specific case."
        )
    if state.get("lime_explanation") is not None:
        with st.expander("LIME Explanation"):
            st.image(state["lime_explanation"], width="stretch")

    st.subheader("AI Report")
    report_md = state.get("markdown_report") or "No report."
    render_markdown_report(report_md)
    report_download_buttons(report_md)

    explanation = state.get("llm_response") or (result.get("explanation") if isinstance(result, dict) else None)
    if explanation:
        st.subheader("Patient-Friendly Explanation")
        st.markdown(explanation)

    st.subheader("Detailed Analysis Decisions")
    xai = result.get("xai") or {}
    top_features = xai.get("top_features", [])
    summary_features = ", ".join(top_features) if top_features else "Thermal/statistical/deep feature ensemble"
    detailed = [
        {"Stage": "Preprocessing", "Decision": "Applied CLAHE + luminance equalization + Gaussian denoising + ROI focus", "Impact": "Reduced acquisition noise and improved lesion-contrast visibility for downstream models"},
        {"Stage": "Feature Extraction", "Decision": f"Fused texture (GLCM/LBP), statistical, shape, deep CNN and latent embeddings; key factors: {summary_features}", "Impact": "High-dimensional multi-view representation improved robustness to modality variance"},
        {"Stage": "Inference", "Decision": "EfficientNet + ViT + calibrated thermal classifier fusion with weighted contribution tracking", "Impact": "Combined structural and thermal evidence into a single stable probability profile"},
        {"Stage": "Calibration", "Decision": f"Applied reliability-aware probability calibration; confidence={conf:.3f}, reliability={reliability:.3f}", "Impact": "Confidence better aligned with uncertainty and class overlap behavior"},
        {"Stage": "Risk Logic", "Decision": f"Mapped malignant posterior to risk score ({risk_score:.2f}) and risk level ({risk})", "Impact": "Created actionable triage indicator with explainable score semantics"},
    ]
    st.dataframe(pd.DataFrame(detailed), width="stretch")

    with st.expander("Raw Markdown Report Preview"):
        st.code(report_md, language="markdown")

    logs = state.get("pipeline_logs", []) or []
    if logs:
        with st.expander("Detailed Pipeline Audit Logs", expanded=False):
            for line in logs:
                st.code(str(line), language="text")

    malignant_prob = probs.get("Malignant", 0.0)
    if malignant_prob > 0.6:
        st.error("⚠️ HIGH RISK DETECTED — Please consult a medical professional immediately")
    elif pred.lower() == "benign":
        st.warning("🟠 Advisory: benign-like findings detected. Follow-up recommended.")
    else:
        st.success("✅ Low immediate concern detected. Continue routine screening.")


if __name__ == "__main__":
    render()

