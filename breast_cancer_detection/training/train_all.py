"""
RUN ONCE: python -m breast_cancer_detection.training.train_all
Trains all models and saves them to models/. Skips existing models by default.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from breast_cancer_detection.training.dataset_loader import discover_dataset
from breast_cancer_detection.utils.config import DATASETS_DIR as CONFIG_DATASETS_DIR, MODELS_DIR as CONFIG_MODELS_DIR

MODELS_DIR = CONFIG_MODELS_DIR
DATASETS_DIR = CONFIG_DATASETS_DIR
FORCE_RETRAIN = False


def check_models_exist() -> dict:
    return {
        "efficientnet": (MODELS_DIR / "efficientnet_model.pth").exists(),
        "vit": (MODELS_DIR / "vit_model.pth").exists(),
        "xgboost": (MODELS_DIR / "xgboost_thermal.pkl").exists(),
        "ensemble": (MODELS_DIR / "ensemble_meta_model.pkl").exists(),
        "unet": (MODELS_DIR / "unet_segmentation.pth").exists(),
        "autoencoder": (MODELS_DIR / "autoencoder.pth").exists(),
    }


def _write_evaluation_stub():
    df = discover_dataset()
    report = {
        "efficientnet": {},
        "vit": {},
        "xgboost": {},
        "ensemble": {},
        "training_completed_at": datetime.now(timezone.utc).isoformat(),
        "dataset_stats": {
            "total_images": int(df["image_path"].notna().sum()),
            "normal": int((df["label"] == "normal").sum()),
            "benign": int((df["label"] == "benign").sum()),
            "malignant": int((df["label"] == "malignant").sum()),
            "total_thermal_matrices": int(df["thermal_path"].notna().sum()),
        },
    }
    (MODELS_DIR / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("TrainingPipeline")
    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    status = check_models_exist()

    if not status["unet"] or FORCE_RETRAIN:
        logger.info("Training U-Net Segmentation model...")
        from breast_cancer_detection.training.train_segmentation import train_unet
        train_unet(DATASETS_DIR, MODELS_DIR)

    if not status["autoencoder"] or FORCE_RETRAIN:
        logger.info("Training Autoencoder...")
        from breast_cancer_detection.training.train_autoencoder import train_autoencoder
        train_autoencoder(DATASETS_DIR, MODELS_DIR)

    eff_metrics = {}
    if not status["efficientnet"] or FORCE_RETRAIN:
        logger.info("Training EfficientNet-B4...")
        from breast_cancer_detection.training.train_image_models import train_efficientnet
        eff_metrics = train_efficientnet(DATASETS_DIR, MODELS_DIR)

    vit_metrics = {}
    if not status["vit"] or FORCE_RETRAIN:
        logger.info("Training Vision Transformer (ViT-B/16)...")
        from breast_cancer_detection.training.train_image_models import train_vit
        vit_metrics = train_vit(DATASETS_DIR, MODELS_DIR)

    xgb_metrics = {}
    if not status["xgboost"] or FORCE_RETRAIN:
        logger.info("Training XGBoost Thermal Model...")
        from breast_cancer_detection.training.train_thermal_model import train_xgboost
        xgb_metrics = train_xgboost(DATASETS_DIR, MODELS_DIR)

    ens_metrics = {}
    if not status["ensemble"] or FORCE_RETRAIN:
        logger.info("Training Ensemble Meta-Learner...")
        from breast_cancer_detection.training.train_image_models import train_ensemble
        ens_metrics = train_ensemble(DATASETS_DIR, MODELS_DIR)

    report = _write_evaluation_stub()
    if eff_metrics:
        report["efficientnet"] = eff_metrics
    if vit_metrics:
        report["vit"] = vit_metrics
    if xgb_metrics:
        report["xgboost"] = xgb_metrics
    if ens_metrics:
        report["ensemble"] = ens_metrics
    (MODELS_DIR / "evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("ALL MODELS TRAINED AND SAVED TO models/")
    logger.info("Start API: uvicorn breast_cancer_detection.backend.api:app --reload")
    logger.info("Start UI: cd frontend && npm run dev")


if __name__ == "__main__":
    main()

