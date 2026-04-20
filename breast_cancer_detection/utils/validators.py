from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import IMAGE_EXTENSIONS, MAX_UPLOAD_SIZE_MB, MIN_IMAGE_SIZE, THERMAL_EXTENSIONS


class ValidationError(ValueError):
    ...


def validate_file_size(file_path: Path) -> None:
    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise ValidationError(f"File exceeds {MAX_UPLOAD_SIZE_MB}MB limit ({size_mb:.2f}MB).")


def validate_image_file(file_path: Path) -> Tuple[int, int]:
    if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValidationError(f"Unsupported image type: {file_path.suffix}")
    validate_file_size(file_path)
    try:
        with Image.open(file_path) as img:
            width, height = img.size
    except UnidentifiedImageError as exc:
        raise ValidationError("Corrupt or unsupported image file.") from exc
    if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
        raise ValidationError(
            f"Image too small. Minimum required: {MIN_IMAGE_SIZE[0]}x{MIN_IMAGE_SIZE[1]}"
        )
    return width, height


def validate_thermal_file(file_path: Path) -> None:
    if file_path.suffix.lower() not in THERMAL_EXTENSIONS:
        raise ValidationError(f"Unsupported thermal type: {file_path.suffix}")
    validate_file_size(file_path)
    if file_path.suffix.lower() == ".npy":
        arr = np.load(file_path)
    else:
        arr = np.loadtxt(file_path, delimiter=",")
    if arr.ndim != 2:
        raise ValidationError("Thermal matrix must be 2D.")

