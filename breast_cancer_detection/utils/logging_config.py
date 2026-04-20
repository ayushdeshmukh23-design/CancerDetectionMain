from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from breast_cancer_detection.utils.config import ROOT_DIR

_LOGGING_INITIALIZED = False


def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """Set up structured application logging once and return a logger."""
    global _LOGGING_INITIALIZED
    logger_name = name or "oncovision"
    logger = logging.getLogger(logger_name)
    if _LOGGING_INITIALIZED:
        return logger

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")

    log_dir = ROOT_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    has_file = any(
        isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")).name == "app.log"
        for h in root_logger.handlers
    )
    if not has_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    has_stdout_stream = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        and getattr(h, "stream", None) in (sys.stdout, sys.__stdout__)
        for h in root_logger.handlers
    )
    if not has_stdout_stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    _LOGGING_INITIALIZED = True

    try:
        import sklearn
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.getLogger("runtime").info(
            "Runtime initialized | torch=%s sklearn=%s device=%s",
            torch.__version__,
            sklearn.__version__,
            device,
        )
    except Exception:
        pass

    return logger

