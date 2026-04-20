from __future__ import annotations

import os
import time
from typing import Dict, Optional

import numpy as np
import torch

from breast_cancer_detection.inference.ensemble import EnsemblePredictor
from breast_cancer_detection.inference.explainability import ExplainabilityEngine
from breast_cancer_detection.inference.feature_extractor import FeatureExtractor
from breast_cancer_detection.inference.model_registry import ModelRegistry
from breast_cancer_detection.inference.preprocessor import PreprocessingPipeline
from breast_cancer_detection.utils.config import MODEL_PATHS
from breast_cancer_detection.utils.logger import get_logger
from breast_cancer_detection.utils.model_sync import sync_models_from_zip, validate_model_artifacts

logger = get_logger("predictor")


class Predictor:
    """Centralized, production-grade inference pipeline."""

    def __init__(self, registry: Optional[ModelRegistry] = None):
        status = validate_model_artifacts()
        if not all(status.values()):
            sync_models_from_zip()
            status = validate_model_artifacts()
        if not all(status.values()):
            missing = [k for k, v in status.items() if not v]
            raise RuntimeError(
                "Required model artifacts are missing. Expected trained models in 'breast_cancer_detection/models'. "
                f"Missing: {missing}"
            )
        self.registry = registry or ModelRegistry.get_instance()
        self.device = self.registry.device
        self.preprocessor = PreprocessingPipeline(training=False)
        self.feature_extractor = FeatureExtractor(
            autoencoder_model=self.registry.get_autoencoder(),
            efficientnet_backbone=self.registry.get_feature_backbone(),
            device=self.device,
        )
        self.efficientnet = self.registry.get_efficientnet_classifier()
        self.vit = self.registry.get_vit_classifier()
        self.xgb = self.registry.get_xgboost()
        self.ensemble = self.registry.get_ensemble()
        self.explainer = self.registry.get_explainer()
        manifest = self.registry.manifest or {}
        self.model_version = str(manifest.get("model_version", "unknown"))

    def load_models(self) -> None:
        """Compatibility method: models load via ModelRegistry."""
        _ = self.efficientnet, self.vit, self.xgb, self.ensemble, self.explainer

    def preprocess(self, image_path: str) -> Dict[str, object]:
        start = time.perf_counter()
        logger.info("PREPROCESS start | path=%s", image_path)
        processed = self.preprocessor.process(image_path)
        logger.info("PREPROCESS done | latency_ms=%.1f", (time.perf_counter() - start) * 1000)
        return processed

    def extract_features(self, image: np.ndarray, thermal_matrix: Optional[np.ndarray]) -> Dict[str, object]:
        start = time.perf_counter()
        logger.info("FEATURE_EXTRACTION start")
        features = self.feature_extractor.extract_all(image, thermal_matrix)
        logger.info("FEATURE_EXTRACTION done | latency_ms=%.1f", (time.perf_counter() - start) * 1000)
        return features

    def _predict_image(self, model: torch.nn.Module, tensor: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            logits = model(tensor.to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
        return probs

    def _thermal_feature_vector(self, features: Dict[str, object]) -> Optional[np.ndarray]:
        tf = features.get("thermal", {}) if features else {}
        vec = np.array(
            [
                tf.get("thermal_min", 0.0),
                tf.get("thermal_max", 0.0),
                tf.get("thermal_mean", 0.0),
                tf.get("thermal_std", 0.0),
                tf.get("thermal_asymmetry", 0.0),
                tf.get("thermal_hot_spots", 0.0),
                tf.get("thermal_grad_mean", 0.0),
                tf.get("thermal_entropy", 0.0),
            ],
            dtype=np.float32,
        )
        if vec.size == 0:
            return None
        return vec.reshape(1, -1)

    def predict(self, image_path: str, thermal_matrix: Optional[np.ndarray] = None) -> Dict[str, object]:
        start_total = time.perf_counter()
        logger.info("PREDICTION start")

        processed = self.preprocess(image_path)
        roi = processed["roi"]
        tensor = processed["normalized_tensor"].unsqueeze(0)
        features = self.extract_features(roi, thermal_matrix)

        eff_probs = self._predict_image(self.efficientnet, tensor)
        vit_probs = self._predict_image(self.vit, tensor)

        thermal_probs = None
        flat_thermal_features = None
        if self.xgb is not None and thermal_matrix is not None:
            flat_thermal_features = self._thermal_feature_vector(features)
            if flat_thermal_features is not None:
                expected = getattr(self.xgb, "n_features_in_", None)
                if expected is not None and flat_thermal_features.shape[1] != int(expected):
                    logger.error(
                        "FEATURE_MISMATCH expected=%s got=%s",
                        expected,
                        flat_thermal_features.shape[1],
                    )
                    flat_thermal_features = None
                else:
                    thermal_probs = self.xgb.predict_proba(flat_thermal_features).squeeze()
            else:
                logger.warning("Thermal features missing; skipping xgboost branch.")

        result = self.ensemble.predict(
            image_probs={"efficientnet": eff_probs, "vit": vit_probs},
            thermal_probs=thermal_probs,
            features_used=features,
        )

        feature_names = list(features.get("thermal", {}).keys()) or [f"f{i}" for i in range(8)]
        if flat_thermal_features is None:
            flat_thermal_features = np.zeros((1, len(feature_names)), dtype=np.float32)

        shap_model = None
        if self.xgb is not None:
            if hasattr(self.xgb, "named_steps"):
                shap_model = self.xgb.named_steps.get("xgb")
            elif hasattr(self.xgb, "xgb"):
                shap_model = self.xgb.xgb
        shap_fig = (
            self.explainer.shap_for_features(shap_model, flat_thermal_features, feature_names)
            if shap_model is not None
            else {}
        )

        lime_samples = int(os.getenv("ONCOVISION_LIME_SAMPLES", "200"))
        lime_img = self.explainer.lime_image_explanation(
            processed["resized"],
            predict_fn=lambda x: np.array(
                [
                    self._predict_image(
                        self.efficientnet,
                        torch.from_numpy(i).permute(2, 0, 1).float().unsqueeze(0) / 255.0,
                    )
                    for i in x
                ]
            ),
            num_samples=lime_samples,
        )
        gradcam = self.explainer.gradcam_plus_plus(processed["resized"], target_class=int(np.argmax(eff_probs)))

        result["xai"] = {
            "gradcam_base64": gradcam,
            "shap_figure": shap_fig,
            "lime_image": lime_img,
            "top_features": feature_names[:5],
        }
        result["processed"] = processed
        result["model_version"] = self.model_version
        result["latency_ms"] = float((time.perf_counter() - start_total) * 1000)
        result["reasoning_payload"] = {
            "prediction": result.get("prediction"),
            "confidence": float(result.get("confidence", 0.0)),
            "risk_score": float(result.get("risk_score", 0.0)),
            "key_features": feature_names[:5],
            "notes": "Model-based reasoning derived from ensemble probabilities and explainability features.",
        }

        logger.info(
            "PREDICTION done | prediction=%s confidence=%.4f risk_score=%.2f latency_ms=%.1f",
            result.get("prediction"),
            float(result.get("confidence", 0.0)),
            float(result.get("risk_score", 0.0)),
            result["latency_ms"],
        )
        return result

    def generate_explanation(self, result: Dict[str, object]) -> Dict[str, object]:
        """Return structured reasoning payload for downstream LLM explanation."""
        payload = result.get("reasoning_payload")
        if payload:
            return payload
        return {
            "prediction": result.get("prediction", "Unknown"),
            "confidence": float(result.get("confidence", 0.0)),
            "risk_score": float(result.get("risk_score", 0.0)),
            "key_features": (result.get("xai") or {}).get("top_features", []),
            "notes": "Model-based reasoning derived from ensemble probabilities and explainability features.",
        }

