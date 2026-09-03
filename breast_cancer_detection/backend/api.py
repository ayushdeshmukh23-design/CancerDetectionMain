from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from breast_cancer_detection.agents.chat_agent import stream_chat_response
from breast_cancer_detection.agents.graph import get_compiled_diagnosis_graph
from breast_cancer_detection.llm.openrouter_client import OpenRouterLLMClient
from breast_cancer_detection.utils.background_jobs import get_job, start_job
from breast_cancer_detection.utils.config import OUTPUTS_DIR, ROOT_DIR, UPLOADS_DIR
from breast_cancer_detection.utils.demo_data import demo_diagnosis_state
from breast_cancer_detection.utils.file_store import ensure_storage_dirs, save_output_bytes, save_output_text, save_uploaded_file
from breast_cancer_detection.utils.logger import get_logger
from breast_cancer_detection.utils.model_sync import sync_models_from_zip, validate_model_artifacts
from breast_cancer_detection.utils.pipeline_logs import append_pipeline_log
from breast_cancer_detection.utils.validators import ValidationError, validate_image_file, validate_thermal_file

logger = get_logger("api")
ensure_storage_dirs()

# In-memory cache for demo analysis states to eliminate 280ms synthetic image generation overhead
_DEMO_CACHE: Dict[str, Dict] = {}
_DEMO_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up models and pre-compile DAG during application startup."""
    logger.info("FastAPI lifespan starting: warming up models and compiled graph...")
    try:
        from breast_cancer_detection.inference.predictor import get_shared_predictor

        # Pre-compile LangGraph DAG
        _ = get_compiled_diagnosis_graph()
        # Pre-load all ML/DL models into memory
        _ = get_shared_predictor()
        # Pre-seed demo malignant cache
        _ = _get_or_create_demo_state("case_malignant")
        logger.info("FastAPI lifespan: Warm-up complete.")
    except Exception as exc:
        logger.warning("FastAPI lifespan warm-up notice: %s", exc)
    yield
    logger.info("FastAPI lifespan: shutdown complete.")


app = FastAPI(
    title="OncoVision AI API",
    description="High-Throughput Production-Grade API for Multi-Modal Breast Screening & Explainable AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisStartRequest(BaseModel):
    image_path: str
    thermal_matrix_path: Optional[str] = None
    patient_context: Optional[str] = ""


class ChatStreamRequest(BaseModel):
    prompt: str
    report_context: Optional[str] = ""
    history: Optional[List[Dict[str, str]]] = []
    model: Optional[str] = None
    web_search: Optional[bool] = False


class ReportExportRequest(BaseModel):
    report_markdown: str
    format: str = "pdf"  # "pdf" or "md"


class ChatExportRequest(BaseModel):
    history: List[Dict[str, str]]


def _get_or_create_demo_state(case_id: str) -> Dict[str, Any]:
    if case_id in _DEMO_CACHE:
        return _DEMO_CACHE[case_id]
    with _DEMO_LOCK:
        if case_id in _DEMO_CACHE:
            return _DEMO_CACHE[case_id]
        demo = demo_diagnosis_state(case_id=case_id)
        payload = {
            "diagnosis_state": demo,
            "seeded_history": demo.get("seeded_history", []),
        }
        _DEMO_CACHE[case_id] = payload
        return payload


@app.get("/api/system/status")
def get_system_status():
    status = validate_model_artifacts()
    if not all(status.values()):
        sync_models_from_zip()
        status = validate_model_artifacts()

    openrouter_client = OpenRouterLLMClient.get_shared()
    return {
        "models_loaded": all(status.values()),
        "model_details": status,
        "llm_connected": openrouter_client.is_available,
        "llm_model": openrouter_client.primary_model,
        "timestamp": time.time(),
    }


@app.post("/api/upload/image")
async def upload_image(file: UploadFile = File(...)):
    try:
        saved_path_str = save_uploaded_file(file.file, filename=file.filename, prefix="image")
        image_path = Path(saved_path_str)
        width, height = validate_image_file(image_path)
        filename = image_path.name
        return {
            "success": True,
            "image_path": saved_path_str,
            "filename": filename,
            "preview_url": f"/api/files/uploads/{filename}",
            "width": width,
            "height": height,
            "format": image_path.suffix.lstrip(".").upper(),
        }
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process image upload: {exc}")


@app.post("/api/upload/thermal")
async def upload_thermal(file: UploadFile = File(...)):
    try:
        saved_path_str = save_uploaded_file(file.file, filename=file.filename, prefix="thermal")
        thermal_path = Path(saved_path_str)
        validate_thermal_file(thermal_path)

        if thermal_path.suffix.lower() == ".npy":
            matrix = np.load(thermal_path)
        else:
            matrix = np.loadtxt(thermal_path, delimiter=",")

        filename = thermal_path.name
        return {
            "success": True,
            "thermal_path": saved_path_str,
            "filename": filename,
            "matrix": matrix.tolist(),
            "shape": list(matrix.shape),
            "min_val": float(matrix.min()),
            "max_val": float(matrix.max()),
            "mean_val": float(matrix.mean()),
        }
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to process thermal upload: {exc}")


@app.get("/api/files/{category}/{filename}")
def serve_file(category: str, filename: str):
    if category == "uploads":
        file_path = UPLOADS_DIR / filename
    elif category == "outputs":
        file_path = OUTPUTS_DIR / filename
    else:
        raise HTTPException(status_code=404, detail="Category not recognized")

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@app.post("/api/analysis/start")
def start_analysis(req: AnalysisStartRequest):
    if not req.image_path or not Path(req.image_path).exists():
        raise HTTPException(status_code=400, detail="Invalid or missing image_path")

    state = {
        "image_path": req.image_path,
        "thermal_matrix_path": req.thermal_matrix_path,
        "patient_context": req.patient_context or "",
        "preprocessed_image": None,
        "segmentation_mask": None,
        "image_features": None,
        "thermal_features": None,
        "deep_embeddings": None,
        "model_predictions": None,
        "ensemble_result": None,
        "gradcam_heatmap": None,
        "shap_values": None,
        "lime_explanation": None,
        "markdown_report": None,
        "risk_assessment": None,
        "chat_history": [],
        "llm_response": None,
        "pipeline_logs": [],
        "errors": [],
        "processing_time": 0.0,
    }

    append_pipeline_log(state, "pipeline", "Pipeline request received via API", details={"image_path": state["image_path"]})
    if state.get("thermal_matrix_path"):
        append_pipeline_log(state, "pipeline", "Thermal modality attached", details={"thermal_path": state["thermal_matrix_path"]})
    else:
        append_pipeline_log(state, "pipeline", "Thermal modality missing; running image-only path", level="WARN")

    def _run_pipeline(payload):
        graph = get_compiled_diagnosis_graph()
        return graph.invoke(payload)

    job_id = start_job("analysis", _run_pipeline, state)
    return {"success": True, "job_id": job_id}


def _ndarray_to_b64(arr: np.ndarray) -> str:
    try:
        if arr is None or arr.size == 0:
            return ""
        if arr.ndim == 2:
            img = Image.fromarray(arr.astype(np.uint8), mode="L")
        elif arr.ndim == 3:
            if arr.shape[-1] == 1:
                img = Image.fromarray(arr.squeeze(-1).astype(np.uint8), mode="L")
            else:
                img = Image.fromarray(arr[:, :, :3].astype(np.uint8), mode="RGB")
        else:
            return ""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    except Exception:
        return ""


def clean_state_for_api(obj: Any) -> Any:
    """Recursively clean PyTorch tensors, NumPy ndarrays, and scalar types for zero-crash JSON serialization."""
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return None
    except ImportError:
        pass

    if isinstance(obj, np.ndarray):
        if obj.ndim in (2, 3) and obj.shape[0] > 10 and obj.shape[1] > 10:
            return _ndarray_to_b64(obj)
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            # Drop heavy duplicate tensor objects in nested processed dict
            if k in ("normalized_tensor", "processed"):
                continue
            cleaned[k] = clean_state_for_api(v)
        return cleaned
    if isinstance(obj, list):
        return [clean_state_for_api(v) for v in obj]
    return obj


@app.get("/api/analysis/status/{job_id}")
def check_analysis_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found")

    status = job.get("status", "unknown")
    result = job.get("result")
    error = job.get("error")

    # Safe float arithmetic with multi-type fallback protection
    raw_created = job.get("created_at")
    if isinstance(raw_created, (int, float)):
        elapsed = max(0.0, time.time() - float(raw_created))
    elif isinstance(raw_created, str):
        try:
            dt = datetime.fromisoformat(raw_created)
            elapsed = max(0.0, time.time() - dt.timestamp())
        except Exception:
            elapsed = 0.0
    else:
        elapsed = 0.0

    cleaned_result = clean_state_for_api(result) if isinstance(result, dict) else None

    return {
        "job_id": job_id,
        "status": status,
        "error": error,
        "result": cleaned_result,
        "elapsed": round(elapsed, 2),
    }


@app.get("/api/analysis/demo")
def get_demo_analysis(case_id: str = "case_malignant"):
    return _get_or_create_demo_state(case_id=case_id)


@app.post("/api/chat/stream")
def stream_chat(req: ChatStreamRequest):
    def event_generator():
        try:
            stream = stream_chat_response(
                user_message=req.prompt,
                report_context=req.report_context or "",
                chat_history=req.history or [],
                model=req.model,
                web_search=bool(req.web_search),
            )
            for token in stream:
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/reports/export")
def export_report(req: ReportExportRequest):
    report_md = req.report_markdown or "No report content."
    if req.format.lower() == "md":
        file_path = save_output_text(report_md, "oncovision_report.md")
        filename = Path(file_path).name
        return {
            "success": True,
            "filename": filename,
            "download_url": f"/api/files/outputs/{filename}",
        }

    # PDF format
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    text = c.beginText(40, 750)
    for line in report_md.splitlines():
        text.textLine(line[:110])
    c.drawText(text)
    c.showPage()
    c.save()

    file_path = save_output_bytes(pdf_buffer.getvalue(), "oncovision_report.pdf")
    filename = Path(file_path).name
    return {
        "success": True,
        "filename": filename,
        "download_url": f"/api/files/outputs/{filename}",
    }


@app.post("/api/chat/export")
def export_chat(req: ChatExportRequest):
    lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in req.history]
    transcript = "\n\n".join(lines)
    file_path = save_output_text(transcript, "chat_history.txt")
    filename = Path(file_path).name
    return {
        "success": True,
        "filename": filename,
        "download_url": f"/api/files/outputs/{filename}",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("breast_cancer_detection.backend.api:app", host="0.0.0.0", port=8000, reload=True)
