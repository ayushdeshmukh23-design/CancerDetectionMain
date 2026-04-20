from __future__ import annotations

import time

from breast_cancer_detection.agents.image_analyzer_agent import safe_agent_run
from breast_cancer_detection.utils.pipeline_logs import append_pipeline_log


@safe_agent_run
def explainability_node(state):
    start = time.perf_counter()
    append_pipeline_log(state, "explainability_agent", "Collecting XAI artifacts")
    ensemble = state.get("ensemble_result") or {}
    xai = ensemble.get("xai", {})
    state["gradcam_heatmap"] = xai.get("gradcam_base64")
    state["shap_values"] = xai.get("shap_figure")
    state["lime_explanation"] = xai.get("lime_image")
    elapsed = time.perf_counter() - start
    append_pipeline_log(
        state,
        "explainability_agent",
        "XAI artifacts ready",
        details={
            "elapsed_s": f"{elapsed:.3f}",
            "has_gradcam": bool(state.get("gradcam_heatmap")),
            "has_shap": bool(state.get("shap_values")),
            "has_lime": bool(state.get("lime_explanation") is not None),
        },
    )
    state["processing_time"] = state.get("processing_time", 0.0) + elapsed
    return state

