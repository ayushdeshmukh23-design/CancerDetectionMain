from __future__ import annotations

import time
import traceback
from functools import wraps

from breast_cancer_detection.inference.preprocessor import PreprocessingPipeline
from breast_cancer_detection.utils.logger import get_logger
from breast_cancer_detection.utils.pipeline_logs import append_pipeline_log

logger = get_logger("image_analyzer_agent")


class AgentError(Exception):
    ...


def safe_agent_run(func):
    @wraps(func)
    def wrapper(state):
        append_pipeline_log(state, func.__name__, "Node execution started")
        try:
            updated = func(state)
            append_pipeline_log(state, func.__name__, "Node execution finished")
            return updated
        except Exception as exc:
            state.setdefault("errors", []).append(f"{func.__name__}: {exc}")
            append_pipeline_log(
                state,
                func.__name__,
                "Node execution failed",
                level="ERROR",
                details={"error": str(exc), "traceback": traceback.format_exc().splitlines()[-1]},
            )
            logger.error("Agent %s failed: %s", func.__name__, exc)
            return state

    return wrapper


@safe_agent_run
def image_analyzer_node(state):
    start = time.perf_counter()
    append_pipeline_log(state, "image_analyzer", "Starting preprocessing", details={"image_path": state.get("image_path", "")})
    pipeline = PreprocessingPipeline(training=False)
    processed = pipeline.process(state["image_path"])
    state["preprocessed_image"] = processed
    state["segmentation_mask"] = processed["segmentation_mask"]
    elapsed = time.perf_counter() - start
    append_pipeline_log(
        state,
        "image_analyzer",
        "Preprocessing complete",
        details={"elapsed_s": f"{elapsed:.3f}", "keys": list(processed.keys())},
    )
    state["processing_time"] = state.get("processing_time", 0.0) + elapsed
    return state

