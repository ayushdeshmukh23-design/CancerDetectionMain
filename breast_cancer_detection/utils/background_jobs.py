from __future__ import annotations

import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable, Dict, Optional

from breast_cancer_detection.utils.logger import get_logger

logger = get_logger("background_jobs")

_JOB_LOCK = threading.RLock()
_MAX_STORED_JOBS = 200
_JOBS: Dict[str, Dict] = {}

# Bounded worker pool prevents thread exhaustion under concurrent traffic
_MAX_WORKERS = min(16, (os.cpu_count() or 4) + 2)
_EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="onco-worker")


def _cleanup_old_jobs_locked() -> None:
    """Evict oldest finished jobs when store exceeds capacity."""
    if len(_JOBS) <= _MAX_STORED_JOBS:
        return
    # Find finished jobs sorted by creation time
    finished_keys = [
        k for k, v in _JOBS.items() if v.get("status") in ("done", "failed")
    ]
    finished_keys.sort(key=lambda k: _JOBS[k].get("created_at", 0.0))
    to_remove = len(_JOBS) - _MAX_STORED_JOBS
    for k in finished_keys[:to_remove]:
        _JOBS.pop(k, None)


def start_job(name: str, target: Callable, *args, **kwargs) -> str:
    job_id = f"{name}-{uuid.uuid4().hex[:8]}"
    now_epoch = time.time()
    now_iso = datetime.now().isoformat(timespec="seconds")

    with _JOB_LOCK:
        _cleanup_old_jobs_locked()
        _JOBS[job_id] = {
            "id": job_id,
            "name": name,
            "status": "running",
            "created_at": now_epoch,
            "created_at_iso": now_iso,
            "completed_at": None,
            "duration_s": None,
            "result": None,
            "error": None,
        }

    def _worker():
        t0 = time.perf_counter()
        try:
            result = target(*args, **kwargs)
            duration = time.perf_counter() - t0
            now_done = time.time()
            with _JOB_LOCK:
                if job_id in _JOBS:
                    _JOBS[job_id]["result"] = result
                    _JOBS[job_id]["status"] = "done"
                    _JOBS[job_id]["completed_at"] = now_done
                    _JOBS[job_id]["duration_s"] = round(duration, 3)
            logger.info("Background job finished | job_id=%s duration=%.2fs", job_id, duration)
        except Exception as exc:
            duration = time.perf_counter() - t0
            now_done = time.time()
            err_msg = str(exc)
            with _JOB_LOCK:
                if job_id in _JOBS:
                    _JOBS[job_id]["error"] = err_msg
                    _JOBS[job_id]["status"] = "failed"
                    _JOBS[job_id]["completed_at"] = now_done
                    _JOBS[job_id]["duration_s"] = round(duration, 3)
            logger.error("Background job failed | job_id=%s duration=%.2fs err=%s", job_id, duration, err_msg)

    _EXECUTOR.submit(_worker)
    return job_id


def get_job(job_id: Optional[str]) -> Optional[Dict]:
    if not job_id:
        return None
    with _JOB_LOCK:
        job = _JOBS.get(job_id)
        return dict(job) if job else None
