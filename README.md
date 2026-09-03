<div align="center">

# OncoVision AI

### Multi-Modal Breast Cancer Detection Clinical Workstation

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![License](https://img.shields.io/badge/License-Research%20Only-red?style=for-the-badge)](./LICENSE)

A **production-grade, end-to-end clinical decision support workstation** that fuses
**Full-Field Digital Mammography (FFDM)** with **calibrated infrared thermography matrices**
through a heterogeneous deep-learning ensemble, a 7-technique Explainable AI suite,
and an OpenRouter-powered LLM clinical copilot — all surfaced through a modern React 19 workstation.

</div>

---

> **Medical Disclaimer**
> OncoVision AI is **not a medical device** and **must not** be used as a standalone diagnostic system.
> All outputs are intended for **education, research, and decision-support only** and must always be reviewed
> by qualified, licensed clinicians before any clinical action is taken.

---

## Table of Contents

- [Why OncoVision AI?](#why-oncovision-ai)
- [Key Capabilities](#key-capabilities)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Training the Models](#training-the-models)
- [Configuration Reference](#configuration-reference)
- [Workstation Walkthrough](#workstation-walkthrough)
- [Explainability (XAI) Suite](#explainability-xai-suite)
- [LLM Clinical Copilot](#llm-clinical-copilot)
- [API Reference](#api-reference)
- [Outputs & Audit Trails](#outputs--audit-trails)
- [Troubleshooting](#troubleshooting)
- [Research Results](#research-results)
- [License & Citation](#license--citation)

---

## Why OncoVision AI?

Breast cancer screening interpretation is cognitively demanding and highly variable across radiologists.
OncoVision AI addresses this by:

- **Fusing two complementary modalities** — morphological mammography and functional thermography — into a single decision surface.
- **Providing layered explanations** so clinicians understand *why* the system reached a prediction, not just *what* it predicted.
- **Grounding every output** in structured ACR BI-RADS categories with priority clinical action recommendations.
- **Generating publication-quality reports** in seconds, reducing documentation overhead.

---

## Key Capabilities

### Multi-Modal Inference Engine

| Modality | Input Format | Processing |
|---|---|---|
| **FFDM Mammography** | `.jpg`, `.png`, `.tif`, `.bmp` | CLAHE + GrabCut + Attention U-Net ROI extraction |
| **Infrared Thermography** | `.csv`, `.npy`, `.txt` (2D matrices) | Contra-lateral ΔT extraction, 8D thermal feature engineering |

### Heterogeneous 3-Branch Ensemble

| Model | Role | Weight |
|---|---|---|
| **EfficientNet-B4** | Local texture, microcalcification, spiculation | 0.40 |
| **ViT-B/16** | Global context, mass distribution, bilateral symmetry | 0.35 |
| **XGBoost (thermal)** | Neovascular thermal signatures | 0.25 |

All branches are fused via **temperature-scaled weighted softmax** and mapped to a **risk score (0–100)** with ACR BI-RADS alignment.

### 7-Technique Explainability (XAI) Suite

1. **Grad-CAM++** — convolutional focal mass activation heatmaps with bounding-box annotation
2. **ViT Self-Attention Rollout** — global parenchymal distortion attention maps
3. **LIME Superpixel Perturbation** — locally faithful positive/negative boundary isolation
4. **TreeSHAP** — quantitative waterfall feature attributions for thermal descriptors
5. **Thermal Isotherm & ΔT Contours** — neo-angiogenesis focal heating visualization
6. **Attention U-Net Lesion Mask** — semantic segmentation overlay
7. **Integrated Gradients** — pixel-level attribution for clustered microcalcifications

### LangGraph Multi-Agent Orchestration

Typed state-machine workflow with per-node safe error isolation:

```
image_analyzer → [thermal_analyzer] → feature_extractor
    → diagnosis_agent → explainability_agent → report_generator
```

### Clinical Output Suite

- **ACR BI-RADS triage banner** with tentative TNM staging and priority management protocol
- **11-section structured clinical markdown report** (LLM-generated, exportable to PDF/MD)
- **4-pillar patient-friendly plain-language guide** with browser SpeechSynthesis audio readback
- **Interactive multi-turn clinical chat** grounded in active study findings
- **Real-time 3×3 multi-agent vision gallery** showing intermediate tensor artifacts

---

## System Architecture

```
+------------------------------------------------------------------+
|          USER UPLOADS IMAGE (+ optional thermal matrix)          |
+---------------------------+--------------------------------------+
                            |
              +-------------v--------------+
              |   FastAPI Ingestion Layer   |
              |   /api/upload/image         |
              |   /api/upload/thermal       |
              |   Validation + Storage      |
              +-------------+--------------+
                            |
              +-------------v--------------+
              |  LangGraph Orchestrator     |
              |                            |
              |  image_analyzer            |
              |      |                     |
              |  [thermal_analyzer]        |
              |      |                     |
              |  feature_extractor         |
              |      |                     |
              |  diagnosis_agent           |
              |  (EfficientNet + ViT       |
              |   + XGBoost + Ensemble)    |
              |      |                     |
              |  explainability_agent      |
              |  (GradCAM++ SHAP LIME)     |
              |      |                     |
              |  report_generator          |
              |  (LLM + patient guide)     |
              +-------------+--------------+
                            |
              +-------------v--------------+
              |  React 19 Workstation       |
              |                            |
              |  Overview  Upload          |
              |  Analysis  Results         |
              |  Dashboard Chat            |
              +----------------------------+
```

---

## Repository Structure

```
CancerDetectionProject-/
|
+-- breast_cancer_detection/         # Core backend Python package
|   +-- backend/
|   |   +-- api.py                   # FastAPI server (CORS, SSE, background jobs)
|   +-- agents/
|   |   +-- graph.py                 # LangGraph workflow definition
|   |   +-- image_analyzer_agent.py  # Stage 1 — preprocessing node
|   |   +-- thermal_analyzer_agent.py# Stage 2 — thermal analysis node
|   |   +-- feature_extractor_agent.py# Stage 3 — feature extraction
|   |   +-- diagnosis_agent.py       # Stage 4 — ensemble inference
|   |   +-- explainability_agent.py  # Stage 5 — XAI synthesis
|   |   +-- report_generator_agent.py# Stage 6 — LLM report generation
|   |   +-- chat_agent.py            # Clinical chat agent
|   +-- inference/
|   |   +-- preprocessor.py          # CLAHE + GrabCut + U-Net preprocessing
|   |   +-- feature_extractor.py     # GLCM + LBP + Shape + Deep features
|   |   +-- model_registry.py        # Thread-safe singleton model loader
|   |   +-- predictor.py             # End-to-end inference orchestrator
|   |   +-- ensemble.py              # Weighted fusion + temperature scaling
|   |   +-- explainability.py        # Grad-CAM++ + SHAP + LIME engine
|   |   +-- model_loader.py          # Safe loading with version checks
|   +-- training/
|   |   +-- train_all.py             # Master training orchestrator
|   |   +-- train_image_models.py    # EfficientNet-B4 + ViT-B/16 + ensemble
|   |   +-- train_thermal_model.py   # XGBoost thermal classifier
|   |   +-- train_segmentation.py    # Attention U-Net (pseudo-label GrabCut)
|   |   +-- train_autoencoder.py     # Convolutional autoencoder (128-dim latent)
|   |   +-- dataset_loader.py        # Recursive discovery + stratified splits
|   +-- llm/
|   |   +-- openrouter_client.py     # OpenRouter API client + streaming
|   |   +-- prompts.py               # Clinical report + patient guide templates
|   +-- utils/                       # Config, logging, validators, model sync
|   +-- models/                      # Trained weight files + manifests
|   +-- logs/                        # Runtime application logs
|
+-- frontend/                        # React 19 clinical workstation
|   +-- src/
|   |   +-- features/
|   |   |   +-- overview/            # Hero + 8 core system capabilities
|   |   |   +-- upload/              # Dual-modality ingestion studio
|   |   |   +-- analysis/            # Synaptic DAG + real-time vision gallery
|   |   |   +-- results/             # BI-RADS triage + XAI suite + patient guide
|   |   |   +-- dashboard/           # Schwartz kinetics + calibration analytics
|   |   |   +-- chat/                # Clinical AI copilot with telemetry
|   |   +-- store/                   # Zustand global state management
|   |   +-- types/                   # TypeScript clinical interfaces
|
+-- scripts/                         # setup_env.ps1  setup_env.sh
+-- Research_Results_Summary.md      # Publication-quality evaluation report
+-- PROJECT_DETAILED_EXPLANATION.md  # Deep technical documentation
+-- requirements.txt                 # Backend Python dependencies
+-- pyproject.toml                   # Package metadata
+-- .env                             # API keys (git-ignored)
+-- .gitignore
```

---

## Requirements

### Backend

| Requirement | Version |
|---|---|
| Python | >= 3.10 |
| PyTorch | 2.x (CPU or CUDA) |
| timm | >= 0.9.12 |
| scikit-learn | >= 1.4.0 |
| XGBoost | >= 2.0.3 |
| FastAPI | >= 0.110.0 |
| LangGraph | >= 0.0.60 |

Full list: [`requirements.txt`](./requirements.txt)

### Frontend

| Requirement | Version |
|---|---|
| Node.js | >= 18 |
| npm | >= 9 |

---

## Quick Start

### Step 1 — Set Up the Python Environment

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

**Linux / macOS**
```bash
bash ./scripts/setup_env.sh
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Step 2 — Configure the API Key

Create `.env` in the repository root (already git-ignored):

```env
OPENROUTER_API_KEY=your-openrouter-key-here
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct
OPENROUTER_FALLBACK_MODEL=mistralai/mistral-7b-instruct
```

Get your key at [openrouter.ai](https://openrouter.ai).

### Step 3 — Install Frontend Dependencies

```powershell
cd frontend
npm install
```

### Step 4 — Launch the Workstation

Open **two terminals** from the repository root:

**Terminal 1 — FastAPI Backend:**
```powershell
uvicorn breast_cancer_detection.backend.api:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — React Frontend:**
```powershell
cd frontend
npm run dev
```

Open **`http://localhost:5173`** in your browser.

---

## Training the Models

If no pre-trained checkpoints exist, train all models with one command:

```bash
python -m breast_cancer_detection.training.train_all
```

This trains sequentially (skipping already-trained models):

| Step | Script | Output |
|---|---|---|
| 1 | `train_segmentation.py` | `unet_segmentation.pth` |
| 2 | `train_autoencoder.py` | `autoencoder.pth` |
| 3 | `train_image_models.py` | `efficientnet_model.pth` |
| 4 | `train_image_models.py` | `vit_model.pth` |
| 5 | `train_thermal_model.py` | `xgboost_thermal.pkl`, `scaler.pkl` |
| 6 | `train_image_models.py` | `ensemble_meta_model.pkl` |

To force retraining of all models, set `FORCE_RETRAIN=True` in `training/train_all.py`.

> Place your dataset under `breast_cancer_detection/Datasets/` (or set `ONCOVISION_DATASETS_DIR`).
> Folder names must contain `normal`, `benign`, or `malignant` for automatic label inference.

---

## Configuration Reference

| Environment Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | — | OpenRouter API key (required for LLM features) |
| `OPENROUTER_MODEL` | `meta-llama/llama-3.1-8b-instruct` | Primary LLM model |
| `OPENROUTER_FALLBACK_MODEL` | `mistralai/mistral-7b-instruct` | Fallback model when primary is unavailable |
| `ONCOVISION_DATASETS_DIR` | `breast_cancer_detection/Datasets` | Dataset root directory |
| `ONCOVISION_MODELS_DIR` | `breast_cancer_detection/models` | Model checkpoint directory |
| `ONCOVISION_UPLOADS_DIR` | `breast_cancer_detection/uploads` | Uploaded file storage |
| `ONCOVISION_OUTPUTS_DIR` | `breast_cancer_detection/outputs` | Generated report/chat storage |
| `ONCOVISION_LIME_SAMPLES` | `100` | LIME perturbation sample count (reduce to speed up inference) |

---

## Workstation Walkthrough

The React 19 frontend is organized into six workflow sections:

| Section | Description |
|---|---|
| **Overview** | System capabilities summary and status dashboard |
| **Upload** | Dual-modality ingestion studio with clinical intake form, benchmark presets, and pre-flight validation |
| **Analysis** | Real-time synaptic DAG node execution progression + 3×3 live multi-agent vision gallery |
| **Results** | ACR BI-RADS triage, ensemble concordance, 7-technique XAI suite, 6-stage visual transformation matrix, patient guide |
| **Dashboard** | Schwartz tumor doubling kinetics, thermal ΔT trajectory, calibration curve, cohort distribution |
| **Chat** | Clinical AI copilot with web search, model selector, and transcript export |

---

## Explainability (XAI) Suite

The XAI pipeline runs automatically after every inference and delivers synchronized, multi-perspective explanations:

| Technique | Target Layer/Component | Output |
|---|---|---|
| **Grad-CAM++** | EfficientNet-B4 last conv block | Base64 heatmap overlay PNG |
| **ViT Self-Attention Rollout** | ViT-B/16 attention heads | Attention weight map |
| **LIME Superpixels** | EfficientNet (100-sample perturbation) | Superpixel attribution image |
| **TreeSHAP** | XGBoost thermal pipeline | Feature importance bar chart |
| **Thermal ΔT Contours** | Raw thermal matrix | Asymmetry heatmap |
| **U-Net Lesion Mask** | Attention U-Net output | Binary segmentation overlay |
| **Integrated Gradients** | EfficientNet pixel path | Pixel attribution map |

All outputs are rendered interactively in the Results section with opacity blending and colormap selection.

---

## LLM Clinical Copilot

The integrated AI assistant is powered by OpenRouter and provides:

- **Context-grounded responses** — grounded in the active study's prediction, risk score, XAI findings, and thermal features
- **Web search capability** — can perform real-time web searches to answer general clinical questions
- **11-section structured clinical report** generated automatically after each analysis
- **4-pillar patient guide** in plain, compassionate language with optional speech synthesis audio
- **Multi-model selector**: Llama 3.1 8B · Claude 3.5 Sonnet · GPT-4o · Mistral Large
- **Transcript export** to `.txt` for documentation

---

## API Reference

The FastAPI backend exposes the following key endpoints:

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload/image` | Upload mammography image |
| `POST` | `/api/upload/thermal` | Upload thermal matrix |
| `POST` | `/api/analyze` | Trigger background analysis job |
| `GET` | `/api/job/{job_id}` | Poll job status and results |
| `GET` | `/api/health` | Backend health check |
| `POST` | `/api/chat` | Clinical AI chat (SSE streaming) |
| `GET` | `/api/report/download` | Download generated PDF/MD report |

Interactive API docs available at **`http://localhost:8000/docs`** when the backend is running.

---

## Outputs & Audit Trails

| Output | Location | Format |
|---|---|---|
| Uploaded images | `breast_cancer_detection/uploads/` | Original format |
| Generated reports | `breast_cancer_detection/outputs/` | `.md` + `.pdf` |
| Chat transcripts | `breast_cancer_detection/outputs/` | `.txt` |
| Runtime logs | `breast_cancer_detection/logs/app.log` | Structured log lines |
| Model evaluation | `breast_cancer_detection/models/evaluation_report.json` | JSON |

---

## Troubleshooting

**Upload errors / unsupported format**
- Images: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff` (min 64×64 px, max 50 MB)
- Thermal matrices: `.csv`, `.npy`, `.txt` (must be a valid 2D array)

**LLM not responding**
- Check `OPENROUTER_API_KEY` is set in `.env`
- The workstation automatically falls back to deterministic structured clinical templates when offline

**Frontend won't start**
- Run `npm install` inside the `frontend/` directory before `npm run dev`
- Ensure Node.js >= 18 is installed

**Slow inference on CPU**
- Reduce `ONCOVISION_LIME_SAMPLES` (e.g. to `50`) to significantly reduce LIME computation time
- LIME is the dominant bottleneck (~57% of total inference time on CPU)

**Model checkpoints not found**
- Run `python -m breast_cancer_detection.training.train_all` to train all models
- Or place pre-trained `.pth` / `.pkl` files in `breast_cancer_detection/models/`

---

## Research Results

A comprehensive, publication-quality evaluation report is available at:

**[`Research_Results_Summary.md`](./Research_Results_Summary.md)**

It covers:
- Model architecture specifications and hyperparameter registry
- Expected performance ranges (ACC / Macro-F1 / AUC-OVR) per model
- Ablation study (feature groups, augmentation, temperature scaling)
- Measured inference latency breakdown (per stage, CPU vs GPU)
- Calibration analysis (Brier Score / ECE / reliability metric)
- Statistical significance framework (CI, McNemar, DeLong)
- Clinical decision threshold analysis and BI-RADS mapping

---

## License & Citation

This project is released for **research and educational purposes only**.

If you use OncoVision AI in academic work, please cite:

```bibtex
@software{oncovision_ai,
  title  = {OncoVision AI: Multi-Modal Breast Cancer Detection Clinical Workstation},
  author = {Ayush Deshmukh},
  year   = {2026},
  url    = {https://github.com/AyushDeshmukh18/CancerDetectionProject-}
}
```

---

<div align="center">

Built with PyTorch · FastAPI · LangGraph · React 19 · OpenRouter

</div>
