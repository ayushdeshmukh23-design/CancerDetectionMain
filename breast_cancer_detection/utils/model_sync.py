from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, Iterable, List

from breast_cancer_detection.utils.config import MODEL_PATHS, MODELS_DIR

REQUIRED_MODEL_FILES = [
    "efficientnet_model.pth",
    "vit_model.pth",
    "xgboost_thermal.pkl",
    "ensemble_meta_model.pkl",
    "unet_segmentation.pth",
    "autoencoder.pth",
    "scaler.pkl",
    "evaluation_report.json",
    "model_versions.json",
]


def _find_zip_candidates(search_roots: Iterable[Path]) -> List[Path]:
    zips: List[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        zips.extend(root.rglob("*.zip"))
    return sorted(zips, key=lambda p: p.stat().st_mtime, reverse=True)


def sync_models_from_zip() -> Dict[str, object]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    roots = [MODELS_DIR, MODELS_DIR.parent.parent / "models", MODELS_DIR.parent.parent]
    zip_candidates = _find_zip_candidates(roots)
    selected_zip = None
    for z in zip_candidates:
        if "model" in z.name.lower() or "artifact" in z.name.lower():
            selected_zip = z
            break
    if selected_zip is None and zip_candidates:
        selected_zip = zip_candidates[0]
    if selected_zip is None:
        return {"synced": False, "zip_used": None, "missing": REQUIRED_MODEL_FILES}

    with zipfile.ZipFile(selected_zip, "r") as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        for req in REQUIRED_MODEL_FILES:
            if req in names:
                target = MODELS_DIR / req
                with zf.open(names[req]) as src, open(target, "wb") as dst:
                    dst.write(src.read())

    missing = [f for f in REQUIRED_MODEL_FILES if not (MODELS_DIR / f).exists()]
    return {"synced": True, "zip_used": str(selected_zip), "missing": missing}


def validate_model_artifacts() -> Dict[str, bool]:
    return {name: path.exists() for name, path in MODEL_PATHS.items() if name != "evaluation"}

