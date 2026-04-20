from __future__ import annotations

import time

import numpy as np

from breast_cancer_detection.agents.image_analyzer_agent import safe_agent_run
from breast_cancer_detection.inference.feature_extractor import FeatureExtractor
from breast_cancer_detection.inference.model_registry import ModelRegistry
from breast_cancer_detection.utils.pipeline_logs import append_pipeline_log


@safe_agent_run
def feature_extractor_node(state):
    start = time.perf_counter()
    append_pipeline_log(state, "feature_extractor", "Starting multimodal feature extraction")
    registry = ModelRegistry.get_instance()
    fx = FeatureExtractor(
        autoencoder_model=registry.get_autoencoder(),
        efficientnet_backbone=registry.get_feature_backbone(),
        device=registry.device,
    )
    pre = state["preprocessed_image"]
    thermal = None
    if state.get("thermal_features") and "matrix" in state["thermal_features"]:
        thermal = state["thermal_features"]["matrix"]
    features = fx.extract_all(pre["roi"], thermal)
    state["image_features"] = features
    state["deep_embeddings"] = {
        "efficientnet": features.get("deep_cnn"),
        "autoencoder": features.get("autoencoder_latent"),
    }
    elapsed = time.perf_counter() - start
    append_pipeline_log(
        state,
        "feature_extractor",
        "Feature extraction complete",
        details={
            "elapsed_s": f"{elapsed:.3f}",
            "has_thermal": thermal is not None,
            "texture_len": len(features.get("texture", [])),
            "stats_len": len(features.get("statistical", [])),
        },
    )
    state["processing_time"] = state.get("processing_time", 0.0) + elapsed
    return state

