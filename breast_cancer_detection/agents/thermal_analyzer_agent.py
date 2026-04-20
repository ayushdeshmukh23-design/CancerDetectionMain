from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from breast_cancer_detection.agents.image_analyzer_agent import safe_agent_run
from breast_cancer_detection.training.dataset_loader import load_thermal_matrix
from breast_cancer_detection.utils.pipeline_logs import append_pipeline_log


@safe_agent_run
def thermal_analyzer_node(state):
    start = time.perf_counter()
    thermal_path = state.get("thermal_matrix_path")
    if not thermal_path:
        append_pipeline_log(state, "thermal_analyzer", "Thermal data missing; skipped node", level="WARN")
        return state
    append_pipeline_log(state, "thermal_analyzer", "Loading thermal matrix", details={"thermal_path": thermal_path})
    matrix = load_thermal_matrix(Path(thermal_path))
    left = matrix[:, : matrix.shape[1] // 2]
    right = matrix[:, matrix.shape[1] // 2 :]
    asymmetry = float(abs(left.mean() - right.mean()))
    norm = cv2.normalize(matrix, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    state["thermal_features"] = {"asymmetry": asymmetry, "matrix": matrix, "heatmap": cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)}
    elapsed = time.perf_counter() - start
    append_pipeline_log(
        state,
        "thermal_analyzer",
        "Thermal analysis complete",
        details={"elapsed_s": f"{elapsed:.3f}", "shape": matrix.shape, "asymmetry": f"{asymmetry:.4f}"},
    )
    state["processing_time"] = state.get("processing_time", 0.0) + elapsed
    return state

