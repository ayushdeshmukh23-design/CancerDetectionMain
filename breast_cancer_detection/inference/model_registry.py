from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import timm
import torch
import torchvision.models as tv_models

from breast_cancer_detection.inference.ensemble import EnsemblePredictor
from breast_cancer_detection.inference.explainability import ExplainabilityEngine
from breast_cancer_detection.inference.model_loader import (
    ModelCompatibilityError,
    load_autoencoder_safely,
    load_sklearn_model,
    preflight_model_validation,
)
from breast_cancer_detection.training.train_image_models import EfficientNetClassifier
from breast_cancer_detection.utils.config import MODEL_PATHS
from breast_cancer_detection.utils.logger import get_logger
from breast_cancer_detection.utils.version_check import ensure_model_metadata, load_model_manifest

logger = get_logger("model_registry")


class ModelRegistry:
    """Thread-safe singleton registry for all ML models."""

    _instance: Optional["ModelRegistry"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.RLock()
        self._autoencoder: Optional[torch.nn.Module] = None
        self._efficientnet: Optional[torch.nn.Module] = None
        self._vit: Optional[torch.nn.Module] = None
        self._xgb = None
        self._ensemble: Optional[EnsemblePredictor] = None
        self._feature_backbone: Optional[torch.nn.Module] = None
        self._explainer: Optional[ExplainabilityEngine] = None
        self._manifest = None
        self._preflight_done = False

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    @property
    def manifest(self):
        return self._manifest

    def _preflight(self) -> None:
        if self._preflight_done:
            return
        with self._lock:
            if self._preflight_done:
                return
            ensure_model_metadata()
            self._manifest = load_model_manifest()
            start = time.perf_counter()
            preflight_model_validation(log_warnings=True)
            elapsed = (time.perf_counter() - start) * 1000
            logger.info("MODEL_LOAD preflight ok | latency_ms=%.1f", elapsed)
            self._preflight_done = True

    def get_autoencoder(self) -> torch.nn.Module:
        self._preflight()
        with self._lock:
            if self._autoencoder is None:
                logger.info("MODEL_LOAD autoencoder")
                self._autoencoder = load_autoencoder_safely(MODEL_PATHS["autoencoder"], device=self.device)
            return self._autoencoder

    def get_efficientnet_classifier(self) -> torch.nn.Module:
        self._preflight()
        with self._lock:
            if self._efficientnet is None:
                logger.info("MODEL_LOAD efficientnet_classifier")
                model = EfficientNetClassifier().to(self.device).eval()
                if MODEL_PATHS["efficientnet"].exists():
                    model.load_state_dict(torch.load(MODEL_PATHS["efficientnet"], map_location=self.device))
                self._efficientnet = model
            return self._efficientnet

    def get_vit_classifier(self) -> torch.nn.Module:
        self._preflight()
        with self._lock:
            if self._vit is None:
                logger.info("MODEL_LOAD vit_classifier")
                model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=3)
                model = model.to(self.device).eval()
                if MODEL_PATHS["vit"].exists():
                    model.load_state_dict(torch.load(MODEL_PATHS["vit"], map_location=self.device))
                self._vit = model
            return self._vit

    def get_feature_backbone(self) -> torch.nn.Module:
        self._preflight()
        with self._lock:
            if self._feature_backbone is None:
                logger.info("MODEL_LOAD efficientnet_backbone")
                model = tv_models.efficientnet_b4(weights=tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1)
                self._feature_backbone = model.to(self.device).eval()
            return self._feature_backbone

    def get_xgboost(self):
        self._preflight()
        with self._lock:
            if self._xgb is None and MODEL_PATHS["xgboost"].exists():
                logger.info("MODEL_LOAD xgboost_classifier")
                json_path = MODEL_PATHS["xgboost"].with_suffix(".json")
                if json_path.exists() and MODEL_PATHS["scaler"].exists():
                    try:
                        import joblib
                        from xgboost import XGBClassifier

                        scaler = joblib.load(MODEL_PATHS["scaler"])
                        xgb_model = XGBClassifier()
                        xgb_model.load_model(str(json_path))

                        class _XGBScaledWrapper:
                            def __init__(self, scaler_obj, model_obj):
                                self.scaler = scaler_obj
                                self.xgb = model_obj
                                self.n_features_in_ = getattr(model_obj, "n_features_in_", None) or getattr(
                                    scaler_obj, "n_features_in_", None
                                )

                            def predict_proba(self, X):
                                Xs = self.scaler.transform(X)
                                return self.xgb.predict_proba(Xs)

                        self._xgb = _XGBScaledWrapper(scaler, xgb_model)
                        return self._xgb
                    except Exception as exc:
                        logger.warning("Failed loading JSON XGBoost; falling back to pickle. %s", exc)
                try:
                    self._xgb = load_sklearn_model(MODEL_PATHS["xgboost"], strict_version=False)
                    if self._xgb is not None:
                        try:
                            xgb_model = self._xgb.named_steps.get("xgb") if hasattr(self._xgb, "named_steps") else None
                            if xgb_model is not None:
                                xgb_model.save_model(str(json_path))
                                logger.info("XGBoost model exported to %s for warning-free loads.", json_path.name)
                        except Exception:
                            logger.warning("Unable to export XGBoost model to JSON format.")
                except ModelCompatibilityError as exc:
                    logger.warning("XGBoost model incompatible: %s", exc)
                    self._xgb = None
            return self._xgb

    def get_ensemble(self) -> EnsemblePredictor:
        self._preflight()
        with self._lock:
            if self._ensemble is None:
                logger.info("MODEL_LOAD ensemble")
                self._ensemble = EnsemblePredictor(MODEL_PATHS["ensemble"])
            return self._ensemble

    def get_explainer(self) -> ExplainabilityEngine:
        self._preflight()
        with self._lock:
            if self._explainer is None:
                logger.info("MODEL_LOAD explainability_engine")
                self._explainer = ExplainabilityEngine(self.get_efficientnet_classifier())
            return self._explainer
