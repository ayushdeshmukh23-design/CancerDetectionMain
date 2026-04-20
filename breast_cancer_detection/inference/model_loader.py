from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
import warnings

import joblib
import torch
from sklearn.exceptions import InconsistentVersionWarning

from breast_cancer_detection.inference.feature_extractor import ConvAutoencoder
from breast_cancer_detection.utils.config import MODEL_PATHS
from breast_cancer_detection.utils.logger import get_logger
from breast_cancer_detection.utils.version_check import (
    ModelCompatibilityError,
    load_model_manifest,
    validate_model_checksums,
    validate_runtime_versions,
    version_warnings,
)

logger = get_logger("model_loader")


def remap_state_dict_keys(state_dict: Dict[str, Any], logger_instance=None) -> Dict[str, Any]:
    """Remap legacy checkpoint keys to current autoencoder module names."""
    lg = logger_instance or logger
    remapped: Dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("enc."):
            new_key = "encoder." + key[len("enc.") :]
        elif key.startswith("dec."):
            new_key = "decoder." + key[len("dec.") :]
        elif key.startswith("fc."):
            new_key = "bottleneck." + key[len("fc.") :]
        elif key.startswith("fc1."):
            new_key = "bottleneck." + key[len("fc1.") :]
        elif key.startswith("fc2."):
            new_key = "expand." + key[len("fc2.") :]
        if new_key != key:
            lg.debug("Remapped checkpoint key: %s -> %s", key, new_key)
        remapped[new_key] = value
    return remapped


def _extract_checkpoint_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise RuntimeError("Checkpoint format invalid: expected state_dict dict or {'state_dict': ...}.")


def validate_state_dict(model: torch.nn.Module, state_dict: Dict[str, Any], logger_instance=None) -> Tuple[list[str], list[str]]:
    """Validate checkpoint keys against model keys and enforce critical layer presence."""
    lg = logger_instance or logger
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing_keys = sorted(model_keys - ckpt_keys)
    unexpected_keys = sorted(ckpt_keys - model_keys)

    if missing_keys:
        lg.warning("Autoencoder state_dict missing keys: %s", missing_keys)
    if unexpected_keys:
        lg.warning("Autoencoder state_dict unexpected keys: %s", unexpected_keys)

    critical_missing = [k for k in missing_keys if k.startswith("encoder.") or k.startswith("decoder.")]
    if critical_missing:
        raise RuntimeError(f"Critical autoencoder layers missing: {critical_missing[:20]}")

    return missing_keys, unexpected_keys


def load_autoencoder_model(
    model: ConvAutoencoder,
    path: Path,
    device: str = "cpu",
    logger_instance=None,
) -> ConvAutoencoder:
    """Load autoencoder checkpoint safely with explicit remapping and validation."""
    lg = logger_instance or logger
    lg.info("Loading model start | path=%s device=%s", path, device)
    try:
        checkpoint = torch.load(path, map_location=device)
        raw_state_dict = _extract_checkpoint_state_dict(checkpoint)
        remapped_state_dict = remap_state_dict_keys(raw_state_dict, logger_instance=lg)
        validate_state_dict(model, remapped_state_dict, logger_instance=lg)
        missing, unexpected = model.load_state_dict(remapped_state_dict, strict=False)
        if missing:
            lg.warning("Post-load missing keys: %s", missing)
        if unexpected:
            lg.warning("Post-load unexpected keys: %s", unexpected)
        model = model.to(device)
        model.eval()
        lg.info("Autoencoder model load successful | path=%s", path.name)
        return model
    except Exception as exc:
        lg.exception("Autoencoder loading failed: %s", exc)
        raise RuntimeError(f"Failed to load autoencoder model from {path}: {exc}") from exc


def load_autoencoder_safely(path: Path, device: str = "cpu") -> ConvAutoencoder:
    model = ConvAutoencoder()
    return load_autoencoder_model(model=model, path=path, device=device, logger_instance=logger)


def _is_sklearn_version_compatible(manifest: Dict[str, Any]) -> tuple[bool, str]:
    import sklearn

    runtime_version = sklearn.__version__
    expected = str(manifest.get("sklearn_version", "")).strip()
    compatibles = manifest.get("sklearn_compatible_versions", [])
    if not isinstance(compatibles, list):
        compatibles = []
    allowed = [v for v in [expected, *compatibles] if v]
    if not allowed:
        return True, "No explicit sklearn version policy in manifest."
    if runtime_version in allowed:
        return True, f"Runtime sklearn {runtime_version} is compatible."
    return False, f"Runtime sklearn {runtime_version} is not in compatible set {allowed}."


def load_sklearn_model(path: Path, *, strict_version: bool = False):
    manifest = load_model_manifest()
    ok_versions, msg_versions = _is_sklearn_version_compatible(manifest)
    if not ok_versions and strict_version:
        raise ModelCompatibilityError(f"Model/runtime version mismatch. {msg_versions}")
    if not ok_versions and not strict_version:
        logger.info("Sklearn version mismatch detected; trying safe fallback load. %s", msg_versions)
    ok_checksum, msg_checksum = validate_model_checksums(manifest)
    if not ok_checksum:
        raise ModelCompatibilityError(f"Artifact integrity check failed. {msg_checksum}")
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        try:
            return joblib.load(path)
        except InconsistentVersionWarning:
            if strict_version:
                raise ModelCompatibilityError(f"{path.name} is incompatible with current scikit-learn runtime.")
            logger.info("Skipping incompatible sklearn model %s for this runtime.", path.name)
            return None
        except Exception as exc:
            if strict_version:
                raise ModelCompatibilityError(f"Failed loading sklearn model {path.name}: {exc}") from exc
            logger.info("Unable to load sklearn model %s safely; using fallback path.", path.name)
            return None


def load_joblib_with_manifest(path: Path):
    return load_sklearn_model(path=path, strict_version=True)


def save_runtime_manifest(path: Path | None = None) -> Path:
    import json
    import hashlib
    import sklearn

    target = path or (MODEL_PATHS["manifest"])
    target.parent.mkdir(parents=True, exist_ok=True)
    files = [
        "efficientnet_model.pth",
        "vit_model.pth",
        "xgboost_thermal.pkl",
        "ensemble_meta_model.pkl",
        "unet_segmentation.pth",
        "autoencoder.pth",
        "scaler.pkl",
    ]
    checksums = {}
    for name in files:
        fp = MODEL_PATHS["efficientnet"].parent / name
        if fp.exists():
            checksums[name] = hashlib.sha256(fp.read_bytes()).hexdigest()
    payload = {
        "model_version": "v1",
        "sklearn_version": sklearn.__version__,
        "torch_version": ".".join(torch.__version__.split(".")[:2]),
        "checksums": checksums,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def preflight_model_validation(*, log_warnings: bool = False) -> None:
    manifest = load_model_manifest()
    ok_versions, msg_versions = validate_runtime_versions(manifest)
    if not ok_versions:
        raise ModelCompatibilityError(msg_versions)
    if log_warnings:
        for warn in version_warnings(manifest):
            logger.warning(warn)
    ok_checksum, msg_checksum = validate_model_checksums(manifest)
    if not ok_checksum:
        raise ModelCompatibilityError(msg_checksum)
    logger.info(
        "Model preflight passed | model_version=%s sklearn=%s torch=%s",
        manifest.get("model_version", "unknown"),
        manifest.get("sklearn_version", "unknown"),
        manifest.get("torch_version", "unknown"),
    )


def load_autoencoder_for_features(device: str):
    auto_path = MODEL_PATHS["autoencoder"]
    if not auto_path.exists():
        raise ModelCompatibilityError(f"Missing autoencoder artifact: {auto_path}")
    return load_autoencoder_safely(auto_path, device=device)

