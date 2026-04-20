from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import Callable, Dict, Optional

_JOB_LOCK = threading.RLock()
_JOBS: Dict[str, Dict] = {}


def _ensure_registry() -> Dict[str, Dict]:
    return _JOBS


def start_job(name: str, target: Callable, *args, **kwargs) -> str:
    jobs = _ensure_registry()
    job_id = f"{name}-{uuid.uuid4().hex[:8]}"
    with _JOB_LOCK:
        jobs[job_id] = {
            "id": job_id,
            "name": name,
            "status": "running",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "result": None,
            "error": None,
            "thread": None,
        }

    def runner():
        try:
            result = target(*args, **kwargs)
            with _JOB_LOCK:
                jobs[job_id]["result"] = result
                jobs[job_id]["status"] = "done"
        except Exception as exc:
            with _JOB_LOCK:
                jobs[job_id]["error"] = str(exc)
                jobs[job_id]["status"] = "failed"

    th = threading.Thread(target=runner, daemon=True)
    with _JOB_LOCK:
        jobs[job_id]["thread"] = th
    th.start()
    return job_id


def get_job(job_id: Optional[str]) -> Optional[Dict]:
    if not job_id:
        return None
    jobs = _ensure_registry()
    with _JOB_LOCK:
        job = jobs.get(job_id)
        return dict(job) if job else None

