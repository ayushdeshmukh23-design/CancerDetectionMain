from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from breast_cancer_detection.agents.image_analyzer_agent import safe_agent_run
from breast_cancer_detection.inference.predictor import get_shared_predictor
from breast_cancer_detection.utils.pipeline_logs import append_pipeline_log


@safe_agent_run
def diagnosis_node(state):
    start = time.perf_counter()
    append_pipeline_log(state, "diagnosis_agent", "Starting ensemble inference")
    predictor = get_shared_predictor()
    thermal = None
    if state.get("thermal_features") and "matrix" in state["thermal_features"]:
        thermal = state["thermal_features"]["matrix"]

    preprocessed = state.get("preprocessed_image")
    features = state.get("image_features")
    if preprocessed is not None:
        output = predictor.predict_from_artifacts(
            processed=preprocessed,
            features=features,
            thermal_matrix=thermal,
        )
    else:
        output = predictor.predict(state["image_path"], thermal)

    state["model_predictions"] = output.get("model_contributions")
    state["ensemble_result"] = output
    state["preprocessed_image"] = output.get("processed", state.get("preprocessed_image"))
    state["segmentation_mask"] = (output.get("processed") or {}).get("segmentation_mask", state.get("segmentation_mask"))
    state["risk_assessment"] = {"risk_score": output["risk_score"], "risk_level": output["risk_level"]}
    elapsed = time.perf_counter() - start
    append_pipeline_log(
        state,
        "diagnosis_agent",
        "Inference complete",
        details={
            "elapsed_s": f"{elapsed:.3f}",
            "prediction": output.get("prediction"),
            "confidence": f"{float(output.get('confidence', 0.0)):.4f}",
            "risk_score": f"{float(output.get('risk_score', 0.0)):.2f}",
        },
    )
    state["processing_time"] = state.get("processing_time", 0.0) + elapsed
    return state

