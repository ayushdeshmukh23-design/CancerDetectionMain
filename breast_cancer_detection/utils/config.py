from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = Path(os.getenv("ONCOVISION_DATASETS_DIR", ROOT_DIR / "Datasets"))
if not DATASETS_DIR.exists():
    parent_dataset = ROOT_DIR.parent / "Datasets"
    if parent_dataset.exists():
        DATASETS_DIR = parent_dataset
MODELS_DIR = Path(os.getenv("ONCOVISION_MODELS_DIR", ROOT_DIR / "models"))
ASSETS_DIR = Path(os.getenv("ONCOVISION_ASSETS_DIR", ROOT_DIR / "assets"))
UPLOADS_DIR = Path(os.getenv("ONCOVISION_UPLOADS_DIR", ROOT_DIR / "uploads"))
OUTPUTS_DIR = Path(os.getenv("ONCOVISION_OUTPUTS_DIR", ROOT_DIR / "outputs"))

CLASS_NAMES: List[str] = ["normal", "benign", "malignant"]
CLASS_TO_INDEX: Dict[str, int] = {k: i for i, k in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_INDEX.items()}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
THERMAL_EXTENSIONS = {".csv", ".npy", ".txt"}

DEFAULT_STATE = {
    "uploaded_image_path": None,
    "uploaded_thermal_path": None,
    "diagnosis_state": None,
    "last_complete_state": None,
    "analysis_complete": False,
    "current_page": "app",
    "chat_history": [],
    "patient_history": [],
    "models_loaded": False,
    "openrouter_connected": False,
    "llm_connected": False,
    "analysis_job_id": None,
    "analysis_job_error": None,
}

MODEL_PATHS = {
    "efficientnet": MODELS_DIR / "efficientnet_model.pth",
    "vit": MODELS_DIR / "vit_model.pth",
    "xgboost": MODELS_DIR / "xgboost_thermal.pkl",
    "ensemble": MODELS_DIR / "ensemble_meta_model.pkl",
    "unet": MODELS_DIR / "unet_segmentation.pth",
    "autoencoder": MODELS_DIR / "autoencoder.pth",
    "scaler": MODELS_DIR / "scaler.pkl",
    "evaluation": MODELS_DIR / "evaluation_report.json",
    "manifest": MODELS_DIR / "model_versions.json",
}

RISK_THRESHOLDS = {
    "low": 35.0,
    "medium": 65.0,
}

MAX_UPLOAD_SIZE_MB = 50
MIN_IMAGE_SIZE = (64, 64)

