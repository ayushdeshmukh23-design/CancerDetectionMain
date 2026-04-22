# OncoVision AI — Multi‑Modal Breast Cancer Detection (Streamlit + LangGraph + XAI + LLM)

A production‑style, end‑to‑end **breast screening decision‑support** prototype that combines **medical image analysis (RGB)** with **optional thermal matrix analysis**, fuses multiple ML/DL models, generates **explainable outputs (Grad‑CAM++ / SHAP / LIME)**, and produces **LLM‑assisted clinical-style reports + chat** through a Streamlit interface.

> **Important medical disclaimer**  
> This project is **not a medical device** and **must not** be used as a standalone diagnostic system. Outputs are intended for **education/research and decision support only** and should always be reviewed by qualified clinicians.

---

## Table of Contents
- [Key Features](#key-features)
- [System Architecture (High Level)](#system-architecture-high-level)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Quick Start (Recommended)](#quick-start-recommended)
- [Configuration](#configuration)
- [Data & Artifacts](#data--artifacts)
- [Training](#training)
- [Run the Application](#run-the-application)
- [Explainability (XAI)](#explainability-xai)
- [LLM Report & Chat (Ollama)](#llm-report--chat-ollama)
- [Outputs](#outputs)
- [Troubleshooting](#troubleshooting)
- [Security & Privacy Notes](#security--privacy-notes)
- [Roadmap](#roadmap)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Key Features

### Multi‑modal inference (robust to missing thermal input)
- **RGB breast image** pipeline (preprocessing → feature extraction → model inference).
- **Optional thermal matrix** pipeline (feature engineering → classical ML inference).
- Fusion via a **weighted ensemble** to deliver:
  - Class probabilities (e.g., *normal / benign / malignant*)
  - Confidence, reliability (entropy proxy), and a risk score/level

### Explainability built into the UI
- **Grad‑CAM++** heatmaps for CNN-based evidence localization
- **SHAP** for thermal/tabular feature contributions
- **LIME** for local image perturbation explanations
- Structured explainability artifacts designed for direct rendering in Streamlit

### Orchestrated workflow with LangGraph agents
A staged, fault-tolerant workflow that runs in the background (non-blocking UI), with pipeline logs and error isolation per node.

### LLM‑assisted reporting and chat (Ollama)
- Generates a **clinically structured markdown report** (assistive, non-definitive language)
- Generates a **patient-friendly explanation**
- Provides a **context-aware chat** interface grounded in the current analysis/report

---

## System Architecture (High Level)

**Runtime flow:**
1. User uploads an image (and optionally a thermal matrix).
2. Inputs are validated and stored under `uploads/`.
3. A background job executes a **LangGraph** workflow:
   - preprocessing → thermal analysis (optional) → feature extraction → fusion → XAI → report generation
4. Results are persisted to session state and shown across:
   - Analysis → Results → Dashboard → Chat
5. Reports/transcripts can be exported to `outputs/`.

---

## Repository Structure

Key items in the repo root:
- `breast_cancer_detection/` — main Python package (Streamlit app + ML pipeline)
- `scripts/` — environment bootstrap scripts
- `pyproject.toml` — project metadata and packaging (`breast-cancer-detection`)
- `PROJECT_DETAILED_EXPLANATION.md` — deep technical walkthrough

Inside `breast_cancer_detection/` (core package):
- `app.py` — Streamlit entrypoint
- `pages/` — Upload / Analysis / Results / Dashboard / Chat pages
- `agents/` — LangGraph workflow nodes and graph assembly
- `inference/` — preprocessing, features, model registry, fusion, XAI
- `training/` — training pipeline scripts
- `llm/` — Ollama client + prompt templates
- `utils/` — validation, logging, storage, model sync, background jobs
- `models/` — model artifacts + manifest (not always committed)
- `assets/` — UI assets (CSS)

---

## Requirements

### Software
- **Python**: `>= 3.10` (per `pyproject.toml`)
- **Ollama**: installed and running locally (for report/chat features)

### Hardware (recommended)
- CPU-only works for inference if models are optimized, but **GPU recommended** for training and faster inference.

---

## Quick Start (Recommended)

### 1) Clone the repository
```bash
git clone https://github.com/ayushdeshmukh23-design/CancerDetectionProject-.git
cd CancerDetectionProject-
```

### 2) Create and prepare a virtual environment

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**
```bash
bash ./scripts/setup_env.sh
source .venv/bin/activate
```

### 3) Install the package (editable)
From repository root:
```bash
pip install -e .
```

> Note: This project uses `pyproject.toml` and installs `breast_cancer_detection` as a package, enabling clean absolute imports.

---

## Configuration

### Environment variables (`.env`)
Create a `.env` file in the **repository root**:

```env
OLLAMA_NUM_CTX=2048
OLLAMA_NUM_PREDICT=256
```

These configure Ollama context length and generation length for report/chat generation.

---

## Data & Artifacts

### Dataset layout
Place your dataset under:

- `Datasets/images/{normal,benign,malignant}`
- `Datasets/thermal_matrices/{normal,benign,malignant}`

Thermal matrices are optional. The pipeline supports **missing thermal modality** during inference.

### Required model artifacts
Place model artifacts under:

- `breast_cancer_detection/models/`

Expected artifacts include (names may be enforced by the loader/registry):
- `efficientnet_model.pth`
- `vit_model.pth`
- `xgboost_thermal.pkl`
- `ensemble_meta_model.pkl`
- `unet_segmentation.pth`
- `autoencoder.pth`
- `scaler.pkl`
- `model_versions.json`

If artifacts are not present, the application may attempt an internal “artifact sync” strategy (depending on your repo configuration), but you should treat explicit placement as the standard approach.

---

## Training

If you do **not** already have trained artifacts, you can train models (from repo root):

```bash
python -m breast_cancer_detection.training.train_all
```

Training is designed to skip already-existing artifacts unless forced (implementation-dependent).

---

## Run the Application

### Ensure Ollama is running and models are available
Pull required Ollama models:

```bash
ollama pull mistral
ollama pull llama3
```

### Start Streamlit
From repository root:

```bash
streamlit run breast_cancer_detection/app.py
```

### Quick validation checklist
1. Open **Upload** page and run one analysis.
2. Confirm outputs appear in:
   - Analysis
   - Results
   - Dashboard
3. Open **Chat** and ask one question to confirm Ollama responses.

---

## Explainability (XAI)

The system exposes explainability artifacts aligned with each modality:
- **Grad‑CAM++** overlays on the input image (CNN evidence localization)
- **SHAP** feature attributions for thermal/tabular inference
- **LIME** local image explanation overlays

These are presented inside the Streamlit UI and also packaged into the report context.

---

## LLM Report & Chat (Ollama)

### What the LLM does here
- Generates a structured markdown report (clinical-style sections)
- Produces a patient-friendly explanation
- Enables contextual Q&A in the Chat page

### Fallback behavior
If the LLM backend is unavailable, the app is designed to degrade gracefully (e.g., returning a safe fallback message rather than crashing).

---

## Outputs

Generated artifacts are typically stored in:
- `breast_cancer_detection/uploads/` — uploaded inputs
- `breast_cancer_detection/outputs/` — reports/transcripts/exports
- `breast_cancer_detection/logs/` — application logs (if enabled/configured)

---

## Troubleshooting

### Common issues

**1) Streamlit app starts but inference fails**
- Confirm artifacts exist in `breast_cancer_detection/models/`
- Confirm dataset paths match the expected `Datasets/` layout
- Check logs for missing file names or version mismatch messages

**2) Chat/Report not working**
- Ensure Ollama is installed and running
- Ensure models are pulled:
  - `ollama pull mistral`
  - `ollama pull llama3`
- Confirm `.env` exists and contains generation settings

**3) Virtual environment problems**
- Re-run the bootstrap script under `scripts/`
- Ensure you activated the correct environment (`.venv`)

---

## Security & Privacy Notes

- Do not upload real patient data to public or shared deployments.
- Treat generated reports and stored outputs as sensitive.
- Consider adding dataset/model directories to `.gitignore` if they contain private data.

---

## Roadmap (Suggested Enhancements)
- Add calibrated uncertainty quantification (beyond entropy proxy)
- Add external validation + dataset shift checks
- Improve segmentation using curated masks
- Integrate experiment tracking (e.g., MLflow / W&B) and reproducible runs
- Add Dockerfile and optional GPU container support

---

## License
No license file is currently detected in the repository. If you intend others to use or contribute to this project, add a `LICENSE` file (e.g., MIT, Apache-2.0, GPL-3.0) and update this section accordingly.

---

## Citation
If you use this project in academic work, cite it as:

- **OncoVision AI — Multi‑Modal Breast Cancer Detection**, GitHub repository: `ayushdeshmukh23-design/CancerDetectionProject-`

---

## Acknowledgements
- Streamlit for the interactive UI layer
- PyTorch + scikit-learn + XGBoost for modeling
- SHAP / LIME / Grad‑CAM++ ecosystem for explainability
- LangGraph/LangChain patterns for orchestration
- Ollama for local LLM inference
