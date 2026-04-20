from __future__ import annotations

import json
import threading
from datetime import datetime
from typing import Any, Dict, Optional


def append_pipeline_log(
    state: Dict[str, Any],
    stage: str,
    message: str,
    level: str = "INFO",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    logs = state.setdefault("pipeline_logs", [])
    seq = len(logs) + 1
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    thread_id = threading.get_ident()
    base = f"{ts} | #{seq:03d} | {level.upper():<5} | {stage:<22} | {message} | thread={thread_id}"
    if details:
        try:
            detail_str = json.dumps(details, default=str, ensure_ascii=False, sort_keys=True)
        except Exception:
            detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
        base = f"{base} | details={detail_str}"
    logs.append(base)

