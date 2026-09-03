from __future__ import annotations

import os
import threading
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

_PREDICTOR_INSTANCE: Optional["Predictor"] = None
_PREDICTOR_LOCK = threading.Lock()


def get_shared_predictor() -> "Predictor":
    """Thread-safe singleton predictor to eliminate repeated pipeline instantiation."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        with _PREDICTOR_LOCK:
            if _PREDICTOR_INSTANCE is None:
                _PREDICTOR_INSTANCE = Predictor()
    return _PREDICTOR_INSTANCE


class Predictor:
    """Centralized, high-throughput production-grade inference pipeline."""

    def __init__(self, registry: Optional[ModelRegistry] = None):
        status = validate_model_artifacts()
        if not all(status.values()):
            sync_models_from_zip()
            status = validate_model_artifacts()
        if not all(status.values()):
            missing = [k for k, v in status.items() if not v]
            logger.warning(
                "Model checkpoints missing from 'breast_cancer_detection/models': %s. "
                "Inference pipeline will run with initialized neural architectures.",
                missing,
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
        with torch.inference_mode():
            logits = model(tensor.to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()
        return probs

    def _predict_batch(self, model: torch.nn.Module, batch_tensor: torch.Tensor) -> np.ndarray:
        """High-throughput batched tensor inference with inference_mode."""
        with torch.inference_mode():
            logits = model(batch_tensor.to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
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

    def predict_from_artifacts(
        self,
        processed: Dict[str, object],
        features: Optional[Dict[str, object]] = None,
        thermal_matrix: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        """Deduplicated inference reusing pre-existing preprocessing and feature extraction."""
        start_total = time.perf_counter()
        stage_timings: Dict[str, float] = {}

        roi = processed["roi"]
        tensor = processed["normalized_tensor"].unsqueeze(0)

        if features is None:
            t_fx = time.perf_counter()
            features = self.extract_features(roi, thermal_matrix)
            stage_timings["feature_extraction_ms"] = round((time.perf_counter() - t_fx) * 1000, 2)
        else:
            stage_timings["feature_extraction_ms"] = 0.0

        # Run Deep CNN & ViT in inference mode
        t_models = time.perf_counter()
        eff_probs = self._predict_image(self.efficientnet, tensor)
        vit_probs = self._predict_image(self.vit, tensor)
        stage_timings["vision_models_ms"] = round((time.perf_counter() - t_models) * 1000, 2)

        # Thermal classical model
        t_xgb = time.perf_counter()
        thermal_probs = None
        flat_thermal_features = None
        if self.xgb is not None and thermal_matrix is not None:
            flat_thermal_features = self._thermal_feature_vector(features)
            if flat_thermal_features is not None:
                expected = getattr(self.xgb, "n_features_in_", None)
                if expected is not None and flat_thermal_features.shape[1] != int(expected):
                    logger.error("FEATURE_MISMATCH expected=%s got=%s", expected, flat_thermal_features.shape[1])
                    flat_thermal_features = None
                else:
                    thermal_probs = self.xgb.predict_proba(flat_thermal_features).squeeze()
            else:
                logger.warning("Thermal features missing; skipping xgboost branch.")
        stage_timings["xgboost_ms"] = round((time.perf_counter() - t_xgb) * 1000, 2)

        # Ensemble fusion
        t_ens = time.perf_counter()
        result = self.ensemble.predict(
            image_probs={"efficientnet": eff_probs, "vit": vit_probs},
            thermal_probs=thermal_probs,
            features_used=features,
        )
        stage_timings["ensemble_fusion_ms"] = round((time.perf_counter() - t_ens) * 1000, 2)

        feature_names = list(features.get("thermal", {}).keys()) or [f"f{i}" for i in range(8)]
        if flat_thermal_features is None:
            flat_thermal_features = np.zeros((1, len(feature_names)), dtype=np.float32)

        # SHAP
        t_shap = time.perf_counter()
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
        stage_timings["shap_ms"] = round((time.perf_counter() - t_shap) * 1000, 2)

        # Optimized Vectorized Batched LIME
        t_lime = time.perf_counter()
        lime_samples = int(os.getenv("ONCOVISION_LIME_SAMPLES", "100"))

        def _batched_lime_predict(images: np.ndarray) -> np.ndarray:
            batch_size = 32
            n = len(images)
            all_probs = []
            for idx in range(0, n, batch_size):
                chunk = images[idx : idx + batch_size]
                tensors = [
                    torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    for img in chunk
                ]
                batch_t = torch.stack(tensors)
                p = self._predict_batch(self.efficientnet, batch_t)
                all_probs.append(p)
            return np.vstack(all_probs) if all_probs else np.empty((0, 3))

        lime_img = self.explainer.lime_image_explanation(
            processed["resized"],
            predict_fn=_batched_lime_predict,
            num_samples=lime_samples,
        )
        stage_timings["lime_ms"] = round((time.perf_counter() - t_lime) * 1000, 2)

        # Grad-CAM++
        t_gradcam = time.perf_counter()
        gradcam = self.explainer.gradcam_plus_plus(processed["resized"], target_class=int(np.argmax(eff_probs)))
        stage_timings["gradcam_ms"] = round((time.perf_counter() - t_gradcam) * 1000, 2)

        result["xai"] = {
            "gradcam_base64": gradcam,
            "shap_figure": shap_fig,
            "lime_image": lime_img,
            "top_features": feature_names[:5],
        }
        result["processed"] = processed
        result["model_version"] = self.model_version
        result["latency_ms"] = float((time.perf_counter() - start_total) * 1000)
        result["stage_timings"] = stage_timings
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

    def predict(self, image_path: str, thermal_matrix: Optional[np.ndarray] = None) -> Dict[str, object]:
        """Standalone prediction entrypoint including preprocessing and feature extraction."""
        processed = self.preprocess(image_path)
        return self.predict_from_artifacts(processed=processed, features=None, thermal_matrix=thermal_matrix)

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
