from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

from breast_cancer_detection.inference.model_loader import load_joblib_with_manifest
from breast_cancer_detection.utils.config import CLASS_NAMES, MODEL_PATHS, RISK_THRESHOLDS


@dataclass
class CalibrationState:
    temperature: float = 1.0


class EnsemblePredictor:
    def __init__(self, meta_model_path: Optional[Path] = None):
        self.weights = {"efficientnet": 0.40, "vit": 0.35, "xgboost": 0.25}
        self.meta_model_path = meta_model_path or MODEL_PATHS["ensemble"]
        self.meta_model = self._load_meta_model()
        self.calibration = CalibrationState(temperature=1.0)

    def _load_meta_model(self):
        if self.meta_model_path.exists():
            try:
                return load_joblib_with_manifest(self.meta_model_path)
            except Exception:
                # Meta model is optional in current weighted-fusion path.
                return LogisticRegression(max_iter=1000)
        return LogisticRegression(max_iter=1000)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        exp = np.exp(z)
        return exp / (exp.sum() + 1e-12)

    def _apply_temperature_scaling(self, probs: np.ndarray) -> np.ndarray:
        t = max(self.calibration.temperature, 1e-3)
        logits = np.log(np.clip(probs, 1e-8, 1.0))
        scaled = self._softmax(logits / t)
        return scaled

    def _risk(self, malignant_prob: float):
        risk_score = float(np.clip(malignant_prob * 100.0, 0, 100))
        if risk_score < RISK_THRESHOLDS["low"]:
            risk_level = "Low"
        elif risk_score < RISK_THRESHOLDS["medium"]:
            risk_level = "Medium"
        else:
            risk_level = "High"
        return risk_score, risk_level

    def predict(
        self,
        image_probs: Dict[str, np.ndarray],
        thermal_probs: Optional[np.ndarray],
        features_used: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        eff = image_probs["efficientnet"]
        vit = image_probs["vit"]
        if thermal_probs is None:
            thermal_probs = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)

        fused = (
            self.weights["efficientnet"] * eff
            + self.weights["vit"] * vit
            + self.weights["xgboost"] * thermal_probs
        )
        calibrated = self._apply_temperature_scaling(fused)
        prediction_index = int(np.argmax(calibrated))
        prediction = CLASS_NAMES[prediction_index].capitalize()
        confidence = float(np.max(calibrated))
        rel = float(1.0 - (-np.sum(calibrated * np.log(np.clip(calibrated, 1e-12, 1.0))) / np.log(len(CLASS_NAMES))))
        risk_score, risk_level = self._risk(float(calibrated[2]))

        return {
            "prediction": prediction,
            "probabilities": {k.capitalize(): float(v) for k, v in zip(CLASS_NAMES, calibrated)},
            "confidence": confidence,
            "reliability": rel,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "model_contributions": {
                "efficientnet": {k.capitalize(): float(v) for k, v in zip(CLASS_NAMES, eff)},
                "vit": {k.capitalize(): float(v) for k, v in zip(CLASS_NAMES, vit)},
                "xgboost": {k.capitalize(): float(v) for k, v in zip(CLASS_NAMES, thermal_probs)},
            },
            "features_used": features_used or {},
        }

