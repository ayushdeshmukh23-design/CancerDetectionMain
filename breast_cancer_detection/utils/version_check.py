from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Tuple, List

import sklearn
import torch

from breast_cancer_detection.utils.config import MODELS_DIR


class ModelCompatibilityError(RuntimeError):
    pass


DEFAULT_MANIFEST: Dict[str, object] = {
    "model_version": "1.0.0-runtime",
    "torch_version": torch.__version__,
    "sklearn_version": sklearn.__version__,
    "models": {
        "efficientnet": "efficientnet_model.pth",
        "vit": "vit_model.pth",
        "xgboost": "xgboost_thermal.pkl",
        "ensemble": "ensemble_meta_model.pkl",
        "unet": "unet_segmentation.pth",
        "autoencoder": "autoencoder.pth",
    },
}


def load_model_manifest(path: Path | None = None) -> Dict[str, object]:
    manifest_path = path or (MODELS_DIR / "model_versions.json")
    if not manifest_path.exists():
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(DEFAULT_MANIFEST, indent=2), encoding="utf-8")
        return DEFAULT_MANIFEST
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_MANIFEST


def _metadata_path() -> Path:
    return MODELS_DIR / "model_metadata.json"


def ensure_model_metadata() -> Path:
    """Ensure minimal model metadata file exists for runtime version checks."""
    path = _metadata_path()
    if path.exists():
        return path
    manifest = load_model_manifest()
    payload = {
        "model_version": manifest.get("model_version", "1.0.0-runtime"),
        "torch_version": manifest.get("torch_version", torch.__version__),
        "sklearn_version": manifest.get("sklearn_version", sklearn.__version__),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_model_metadata() -> Dict[str, object]:
    path = _metadata_path()
    if not path.exists():
        ensure_model_metadata()
    return json.loads(_metadata_path().read_text(encoding="utf-8"))


def runtime_versions() -> Dict[str, str]:
    return {
        "sklearn_version": sklearn.__version__,
        "torch_version": torch.__version__,
    }


def validate_runtime_versions(manifest: Dict[str, object]) -> Tuple[bool, str]:
    runtime = runtime_versions()
    expected_sklearn = str(manifest.get("sklearn_version", "")).strip()
    sklearn_compatible = manifest.get("sklearn_compatible_versions", [])
    if not isinstance(sklearn_compatible, list):
        sklearn_compatible = []
    expected_torch = str(manifest.get("torch_version", "")).strip()
    torch_compatible_prefixes = manifest.get("torch_compatible_prefixes", [])
    if not isinstance(torch_compatible_prefixes, list):
        torch_compatible_prefixes = []
    sklearn_allowed = [v for v in [expected_sklearn, *sklearn_compatible] if str(v).strip()]
    if sklearn_allowed and runtime["sklearn_version"] not in sklearn_allowed:
        return (
            False,
            f"scikit-learn runtime ({runtime['sklearn_version']}) not in compatible set ({sklearn_allowed}).",
        )
    torch_allowed = [v for v in [expected_torch, *torch_compatible_prefixes] if str(v).strip()]
    if torch_allowed and not any(runtime["torch_version"].startswith(str(prefix)) for prefix in torch_allowed):
        return (
            False,
            f"torch runtime ({runtime['torch_version']}) not in compatible prefixes ({torch_allowed}).",
        )
    return True, "ok"


def version_warnings(manifest: Dict[str, object]) -> List[str]:
    """Return non-fatal warnings for version mismatch cases that are still compatible."""
    runtime = runtime_versions()
    warnings: List[str] = []
    expected_sklearn = str(manifest.get("sklearn_version", "")).strip()
    if expected_sklearn and runtime["sklearn_version"] != expected_sklearn:
        warnings.append(
            f"scikit-learn runtime {runtime['sklearn_version']} differs from trained {expected_sklearn} (compatibility list applied)."
        )
    expected_torch = str(manifest.get("torch_version", "")).strip()
    if expected_torch and not runtime["torch_version"].startswith(expected_torch):
        warnings.append(
            f"torch runtime {runtime['torch_version']} differs from trained {expected_torch} (compatibility prefixes applied)."
        )
    return warnings


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_model_checksums(manifest: Dict[str, object]) -> Tuple[bool, str]:
    expected = manifest.get("checksums", {})
    if not isinstance(expected, dict):
        return False, "Invalid checksums block in model manifest."
    for name, checksum in expected.items():
        file_path = MODELS_DIR / str(name)
        if not file_path.exists():
            return False, f"Missing artifact: {file_path.name}"
        actual = file_sha256(file_path)
        if actual.lower() != str(checksum).lower():
            return False, f"Checksum mismatch for {file_path.name}"
    return True, "ok"

