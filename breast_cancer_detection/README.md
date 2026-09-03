# breast_cancer_detection — Backend Package Reference

This document is the **technical reference** for the `breast_cancer_detection` Python package.
For the full project overview, deployment instructions, and workstation guide, see the
[root README](../README.md).

---

## Package Layout

```
breast_cancer_detection/
|
+-- backend/
|   +-- api.py                    # FastAPI application — all HTTP endpoints
|
+-- agents/
|   +-- graph.py                  # LangGraph DAG definition & state schema
|   +-- image_analyzer_agent.py   # Node 1: image load + preprocessing
|   +-- thermal_analyzer_agent.py # Node 2: thermal matrix feature extraction
|   +-- feature_extractor_agent.py# Node 3: full multi-modal feature extraction
|   +-- diagnosis_agent.py        # Node 4: ensemble inference (Eff + ViT + XGB)
|   +-- explainability_agent.py   # Node 5: Grad-CAM++ + SHAP + LIME synthesis
|   +-- report_generator_agent.py # Node 6: LLM report + patient guide
|   +-- chat_agent.py             # Clinical chat (streaming)
|
+-- inference/
|   +-- preprocessor.py           # CLAHE, GrabCut, U-Net masking, normalization
|   +-- feature_extractor.py      # GLCM, LBP, shape, deep CNN, autoencoder latent
|   +-- model_registry.py         # Thread-safe singleton for all models
|   +-- predictor.py              # End-to-end inference pipeline (Predictor class)
|   +-- ensemble.py               # Weighted fusion + temperature scaling
|   +-- explainability.py         # Grad-CAM++, SHAP TreeExplainer, LIME engine
|   +-- model_loader.py           # Safe checkpoint loading with version checks
|
+-- training/
|   +-- train_all.py              # Master orchestrator — trains missing models
|   +-- dataset_loader.py         # Recursive discovery + stratified split
|   +-- train_image_models.py     # EfficientNet-B4, ViT-B/16, ensemble meta-learner
|   +-- train_thermal_model.py    # XGBoost thermal classifier (StandardScaler pipeline)
|   +-- train_segmentation.py     # Attention U-Net with GrabCut pseudo-labels
|   +-- train_autoencoder.py      # Convolutional autoencoder (128-dim latent)
|
+-- llm/
|   +-- openrouter_client.py      # OpenRouter API client (generate + stream)
|   +-- prompts.py                # Clinical report, patient guide & chat templates
|
+-- utils/
|   +-- config.py                 # Paths, class names, risk thresholds, model paths
|   +-- logger.py                 # Structured application logger
|   +-- validators.py             # Upload file-type, size, and dimension checks
|   +-- model_sync.py             # ZIP artifact extraction fallback
|   +-- version_check.py          # SHA-256 integrity + manifest validation
|   +-- job_runner.py             # Background async job queue
|
+-- models/                       # Checkpoint files (git-ignored except manifests)
|   +-- model_versions.json       # Pinned version manifest (tracked in git)
|   +-- model_metadata.json       # Runtime metadata (tracked in git)
|
+-- logs/
|   +-- app.log                   # Structured runtime log (git-ignored)
```

---

## Quick Start (from repository root)

### 1. Environment Setup

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

```bash
# Linux / macOS
bash ./scripts/setup_env.sh && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

### 2. Configure `.env`

```env
OPENROUTER_API_KEY=your-openrouter-key-here
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct
OPENROUTER_FALLBACK_MODEL=mistralai/mistral-7b-instruct
```

### 3. Train Models (first run only)

```bash
# Place dataset under breast_cancer_detection/Datasets/
# Folder names must contain: normal | benign | malignant
python -m breast_cancer_detection.training.train_all
```

### 4. Start Backend Server

```bash
uvicorn breast_cancer_detection.backend.api:app --host 127.0.0.1 --port 8000 --reload
```

---

## Inference Pipeline

The `Predictor` class in `inference/predictor.py` exposes the full pipeline:

```python
from breast_cancer_detection.inference.predictor import get_shared_predictor

predictor = get_shared_predictor()  # thread-safe singleton

# Simple one-call inference
result = predictor.predict("path/to/image.jpg", thermal_matrix=None)

# Result keys:
# prediction         str         "Normal" | "Benign" | "Malignant"
# probabilities      dict        {Normal: 0.xx, Benign: 0.xx, Malignant: 0.xx}
# confidence         float       max(probabilities)
# reliability        float       1 - H(p)/log(K)
# risk_score         float       0–100
# risk_level         str         "Low" | "Medium" | "High"
# model_contributions dict       per-branch probabilities
# xai                dict        gradcam_base64, shap_figure, lime_image, top_features
# stage_timings      dict        latency_ms per pipeline stage
# latency_ms         float       total end-to-end ms
```

### Stage Timing (Observed, CPU)

| Stage | Time |
|---|---|
| Model load (cold, singleton) | ~2,500 ms |
| Preprocessing | ~1,000 ms |
| EfficientNet forward | ~350 ms |
| ViT forward | ~750 ms |
| LIME (100 samples) | ~5,000 ms |
| Grad-CAM++ | ~200 ms |
| SHAP | ~20 ms |
| Ensemble fusion | ~5 ms |
| **Total (CPU, cold start)** | **~8,834 ms** |

Set `ONCOVISION_LIME_SAMPLES=50` to roughly halve inference time.

---

## Feature Extraction

`inference/feature_extractor.py` computes a rich 463–471 dimensional feature vector:

| Group | Features | Count |
|---|---|---|
| GLCM Texture | contrast, dissimilarity, homogeneity, energy, correlation, ASM | 6 |
| LBP Histogram | uniform LBP (P=24, R=3) | 26 |
| Statistical | mean, var, skew, kurtosis, entropy per channel (R,G,B,Gray) | 20 |
| Shape | area, perimeter, eccentricity, solidity + 7 Hu moments + 8 Zernike | 19 |
| Deep CNN | EfficientNet-B4 AvgPool embedding (subsampled to 256-dim) | 256 |
| Autoencoder | 128-dim latent bottleneck vector | 128 |
| Thermal (optional) | min, max, mean, std, asymmetry, hot_spots, grad_mean, entropy | 8 |

---

## Model Artifacts

After training, the following files are saved to `breast_cancer_detection/models/`:

| File | Source | Size (approx.) |
|---|---|---|
| `efficientnet_model.pth` | EfficientNet-B4 classifier | ~75 MB |
| `vit_model.pth` | ViT-B/16 classifier | ~335 MB |
| `xgboost_thermal.pkl` | XGBoost pipeline (+ scaler) | < 5 MB |
| `ensemble_meta_model.pkl` | Logistic regression meta-learner | < 1 MB |
| `unet_segmentation.pth` | Attention U-Net | ~120 MB |
| `autoencoder.pth` | Convolutional autoencoder | ~90 MB |
| `scaler.pkl` | StandardScaler for thermal features | < 1 MB |
| `evaluation_report.json` | Training metrics per model | < 1 MB |
| `model_versions.json` | Version manifest **(git-tracked)** | < 1 KB |
| `model_metadata.json` | Runtime metadata **(git-tracked)** | < 1 KB |

---

## Reliability & Fault Tolerance

| Failure Scenario | System Response |
|---|---|
| Missing model checkpoint | Warning logged; inference continues with initialized (random-weight) architecture |
| Thermal matrix absent | XGBoost branch replaced with uniform prior `[1/3, 1/3, 1/3]` |
| LIME unavailable | Returns original image (graceful degradation) |
| Grad-CAM unavailable | Returns JET colormap heatmap blend |
| SHAP fails | Returns linearly-spaced fallback values |
| Feature count mismatch | `FEATURE_MISMATCH` logged; thermal branch skipped |
| LLM API unreachable | Returns structured fallback clinical text |
| Invalid upload | HTTP 422 with structured validation error |

---

## Logging

All pipeline stages emit structured log entries to `logs/app.log`:

```
[TIMESTAMP] [LEVEL] [MODULE] KEY=VALUE ...

Key events:
  MODEL_LOAD preflight ok        | latency_ms=0.4
  PREPROCESS start/done          | path=..., latency_ms=...
  FEATURE_EXTRACTION start/done  | latency_ms=...
  PREDICTION done                | prediction=Benign confidence=0.3976 risk_score=32.24 latency_ms=8833.9
  FEATURE_MISMATCH               | expected=8 got=N
```

---

## Dataset Structure

Place datasets under `breast_cancer_detection/Datasets/` using folder names that contain
`normal`, `benign`, or `malignant` for automatic label inference:

```
Datasets/
+-- normal/         # or: healthy/, saud/, saudave/
|   +-- img001.jpg
+-- benign/         # or: benig/
|   +-- img002.png
+-- malignant/      # or: doentes/, malign/, cancer/
    +-- img003.tif
    +-- img003.csv  # Optional paired thermal matrix (same stem)
```

The dataset loader pairs images and thermal matrices by normalized sample key.
Stratified split: **70% train / 15% validation / 15% test** (`random_state=42`).
