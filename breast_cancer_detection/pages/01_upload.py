from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from breast_cancer_detection.agents.graph import build_diagnosis_graph
from breast_cancer_detection.utils.background_jobs import get_job, start_job
from breast_cancer_detection.utils.file_store import save_uploaded_file
from breast_cancer_detection.utils.logger import get_logger
from breast_cancer_detection.utils.pipeline_logs import append_pipeline_log
from breast_cancer_detection.utils.validators import ValidationError, validate_image_file, validate_thermal_file
from breast_cancer_detection.utils.config import CLASS_NAMES
from breast_cancer_detection.utils.demo_data import demo_diagnosis_state

logger = get_logger("upload_page")


def render():
    st.title("📤 Upload & Input")
    if "analysis_job_id" not in st.session_state:
        st.session_state["analysis_job_id"] = None
    if "analysis_job_error" not in st.session_state:
        st.session_state["analysis_job_error"] = None
    if "analysis_last_poll" not in st.session_state:
        st.session_state["analysis_last_poll"] = 0.0
    image_file = st.file_uploader("Upload breast image", type=["jpg", "jpeg", "png", "bmp", "tif"])
    thermal_file = st.file_uploader("Upload thermal matrix (optional)", type=["csv", "npy", "txt"])

    if not st.session_state.get("analysis_complete", False):
        demo = demo_diagnosis_state()
        st.caption("Preview mode: dummy visualizations until real analysis completes.")
        class_color_map = {"normal": "#10B981", "benign": "#F59E0B", "malignant": "#EF4444"}
        probability_map = demo["ensemble_result"]["probabilities"]
        ordered_x = [c for c in CLASS_NAMES if c in probability_map]
        ordered_y = [probability_map[c] for c in ordered_x]
        bar_colors = [class_color_map.get(c, "#00D4FF") for c in ordered_x]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=ordered_x,
                    y=ordered_y,
                    marker_color=bar_colors,
                )
            ]
        )
        fig.update_layout(
            title="Dummy Class Distribution Preview",
            xaxis_title="Class",
            yaxis_title="Probability",
            template="plotly_dark",
            showlegend=False,
        )
        st.plotly_chart(
            fig,
            width="stretch",
        )

    if image_file:
        try:
            image_path_str = save_uploaded_file(image_file, prefix="image")
            image_path = Path(image_path_str)
            width, height = validate_image_file(image_path)
            st.image(str(image_path), width=400)
            st.caption(f"Dimensions: {width}x{height} | Format: {image_path.suffix}")
            st.session_state["uploaded_image_path"] = str(image_path)
        except (ValidationError, Exception) as exc:
            logger.exception("Image upload processing failed")
            st.error(f"Image upload failed: {exc}")
            st.session_state["uploaded_image_path"] = None
            return

    if thermal_file:
        try:
            thermal_path_str = save_uploaded_file(thermal_file, prefix="thermal")
            thermal_path = Path(thermal_path_str)
            validate_thermal_file(thermal_path)
            st.session_state["uploaded_thermal_path"] = str(thermal_path)
            matrix = np.load(thermal_path) if thermal_path.suffix.lower() == ".npy" else np.loadtxt(thermal_path, delimiter=",")
            fig = px.imshow(matrix, color_continuous_scale="Jet", title="Thermal Matrix Preview")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, width="stretch")
        except (ValidationError, Exception) as exc:
            logger.exception("Thermal upload processing failed")
            st.error(f"Thermal upload failed: {exc}")
            st.session_state["uploaded_thermal_path"] = None
            return

    if st.button("Begin Analysis", type="primary", width="stretch"):
        if not st.session_state.get("uploaded_image_path"):
            st.error("Please upload an image first.")
            return
        st.session_state["analysis_job_error"] = None
        st.session_state["analysis_complete"] = False
        state = {
            "image_path": st.session_state["uploaded_image_path"],
            "thermal_matrix_path": st.session_state.get("uploaded_thermal_path"),
            "patient_context": "",
            "preprocessed_image": None,
            "segmentation_mask": None,
            "image_features": None,
            "thermal_features": None,
            "deep_embeddings": None,
            "model_predictions": None,
            "ensemble_result": None,
            "gradcam_heatmap": None,
            "shap_values": None,
            "lime_explanation": None,
            "markdown_report": None,
            "risk_assessment": None,
            "chat_history": [],
            "llm_response": None,
            "pipeline_logs": [],
            "errors": [],
            "processing_time": 0.0,
        }
        append_pipeline_log(state, "pipeline", "Pipeline request received", details={"image_path": state["image_path"]})
        if state.get("thermal_matrix_path"):
            append_pipeline_log(state, "pipeline", "Thermal modality attached", details={"thermal_path": state["thermal_matrix_path"]})
        else:
            append_pipeline_log(state, "pipeline", "Thermal modality missing; running image-only path", level="WARN")

        def _run_pipeline(payload):
            graph = build_diagnosis_graph()
            return graph.invoke(payload)

        st.session_state["analysis_job_id"] = start_job("analysis", _run_pipeline, state)

    active = get_job(st.session_state.get("analysis_job_id"))
    if active:
        if active["status"] == "running":
            st.info("Analysis running in background... You can switch tabs and return.")
            now = time.time()
            if now - float(st.session_state.get("analysis_last_poll", 0.0)) > 1.0:
                st.session_state["analysis_last_poll"] = now
                st.rerun()
        elif active["status"] == "done":
            result = active.get("result")
            if isinstance(result, dict) and result.get("ensemble_result") is not None:
                st.session_state["diagnosis_state"] = result
                st.session_state["analysis_complete"] = True
                st.session_state["current_page"] = "analysis"
                st.success("Analysis complete. Open Results.")
            else:
                st.session_state["analysis_complete"] = False
                st.session_state["analysis_job_error"] = "Analysis completed but output payload is incomplete."
                st.error(st.session_state["analysis_job_error"])
            st.session_state["analysis_job_id"] = None
        elif active["status"] == "failed":
            st.session_state["analysis_complete"] = False
            st.session_state["analysis_job_error"] = active.get("error", "Unknown error")
            st.error(f"Analysis failed: {st.session_state['analysis_job_error']}")
            st.session_state["analysis_job_id"] = None


if __name__ == "__main__":
    render()

