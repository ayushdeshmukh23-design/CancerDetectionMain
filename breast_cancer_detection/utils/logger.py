from __future__ import annotations

import logging
from .logging_config import setup_logging


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)

