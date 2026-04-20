"""OncoVision AI package root."""

from __future__ import annotations

import os

# Prevent albumentations update-check warning noise in production runtime logs.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

