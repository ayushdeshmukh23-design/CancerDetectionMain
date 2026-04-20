from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_PAGES_DIR = Path(__file__).resolve().parent


def _load(filename: str, module_name: str) -> ModuleType:
    path = _PAGES_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load page module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_01_upload = _load("01_upload.py", "pages._01_upload")
_02_analysis = _load("02_analysis.py", "pages._02_analysis")
_03_results = _load("03_results.py", "pages._03_results")
_04_dashboard = _load("04_dashboard.py", "pages._04_dashboard")
_05_chat = _load("05_chat.py", "pages._05_chat")

