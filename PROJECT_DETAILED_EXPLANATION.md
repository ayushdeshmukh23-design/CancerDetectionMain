# OncoVision AI — Complete Project Explanation (Detailed)

## 1. Project Overview

**OncoVision AI** is a multi-modal breast screening support system built with:
- **Medical image analysis** (RGB breast image)
- **Thermal matrix analysis** (optional second modality)
- **Deep learning + classical ML fusion**
- **Explainable AI surfaces** (Grad-CAM++, SHAP, LIME)
- **LLM-powered clinical report + chat assistant**
- **Streamlit UI with workflow pages**

This is an **assistive decision-support pipeline**, not a diagnostic replacement. The app is structured like a production prototype with model artifact checks, background jobs, logging, report export, and version-compatibility controls.

---

## 2. High-Level Architecture

## 2.1 Layers
- **UI layer (Streamlit):** `app.py`, `pages/*`, `components/*`, `assets/styles.css`
- **Orchestration layer (LangGraph):** `agents/graph.py`
- **Inference layer:** `inference/*` (preprocessing, features, model registry, fusion, XAI)
- **Training layer:** `training/*` (dataset discovery + model training scripts)
- **LLM layer:** `llm/*` (Ollama client + prompts)
- **Utilities layer:** `utils/*` (config, validators, storage, logging, model sync, job runner)
- **Compatibility backend wrappers:** `backend/services/*`, `backend/utils/*` (re-export interfaces)

## 2.2 End-to-End Runtime Flow
1. User uploads image (and optional thermal matrix) on **Upload** page.
2. Upload is validated and persisted under `uploads/`.
3. A background job runs the **LangGraph diagnosis workflow**.
4. Workflow stages:
   - image preprocessing
   - optional thermal analysis
   - feature extraction
   - ensemble diagnosis
   - explainability packaging
   - markdown report generation + patient-friendly explanation
5. Outputs are written into session state and visualized across Analysis/Results/Dashboard.
6. Report and chat transcript are exportable to files in `outputs/`.

---

## 3. Repository and Key Files

- **Root setup**
  - `pyproject.toml`: package metadata (`breast-cancer-detection`, script `oncovision-init`)
  - `README.md`: deployment/setup instructions
  - `scripts/setup_env.ps1`, `scripts/setup_env.sh`: one-command environment bootstrap
  - `.env`: Ollama generation params (`OLLAMA_NUM_CTX`, `OLLAMA_NUM_PREDICT`)

- **Core package: `breast_cancer_detection/`**
  - `app.py`: main Streamlit app, top nav, page routing, session initialization, periodic cleanup
  - `pages/*.py`: Upload, Analysis, Results, Dashboard, Chat pages
  - `agents/*.py`: workflow nodes + graph assembly
  - `inference/*.py`: model runtime and explainability engine
  - `training/*.py`: model training scripts
  - `llm/*.py`: Ollama integration + prompt templates
  - `utils/*.py`: state/config/IO/logging/validation/background jobs/model artifact sync
  - `models/`: trained model artifacts + manifest + metrics JSON
  - `assets/styles.css`: UI skinning

---

## 4. UI and Product Workflow

## 4.1 Main App (`app.py`)
- Initializes global session defaults from `utils.config.DEFAULT_STATE`
- Loads `.env` values into process environment
- Ensures uploads/outputs directories exist
- Validates required model artifacts and tries auto-sync from zip if missing
- Keeps status pills for **Models / LLM / Analysis / Chat**
- Handles completed background analysis job promotion into canonical session state
- Performs periodic stale-file cleanup (uploads/outputs older than 24h)

## 4.2 Pages
- **01_upload.py**
  - upload image + optional thermal file
  - file validation (type, size, shape)
  - starts async diagnosis graph run
  - shows preview plots and background-run status
- **02_analysis.py**
  - visual pipeline stage display
  - preprocessing/segmentation preview and pipeline logs
- **03_results.py**
  - prediction headline, confidence gauge, reliability, risk score/level
  - class probability chart
  - Grad-CAM/SHAP/LIME views
  - AI markdown report + PDF/MD download
  - patient-friendly explanation block
- **04_dashboard.py**
  - KPI cards, radar, donut, trend/rolling analytics
  - patient history accumulation (up to 25 points)
- **05_chat.py**
  - context-aware assistant using current report
  - suggested prompts
  - async response generation
  - chat transcript persistence and download

---

## 5. Orchestration with LangGraph

`agents/graph.py` defines a typed workflow state and node graph:
- Entry: `image_analyzer`
- Conditional edge:
  - if thermal exists -> `thermal_analyzer`
  - else skip to `feature_extractor`
- Then: `diagnosis_agent` -> `explainability_agent` -> `report_generator` -> END

All nodes use `safe_agent_run` decorator (from `image_analyzer_agent.py`) to:
- append pipeline logs
- catch and record errors into state
- avoid full pipeline crash from single-node exceptions

---

## 6. Inference Pipeline Internals

## 6.1 Preprocessing (`inference/preprocessor.py`)
- Image load and channel normalization (grayscale/alpha-safe conversion)
- Resize to 224x224
- CLAHE + histogram equalization + Gaussian denoise
- Edge map (Canny)
- Simulated thermal map for visual support
- Foreground extraction (GrabCut)
- Segmentation mask:
  - if segmentation model present, infer mask
  - else Otsu threshold pseudo-mask
- ROI extraction + ImageNet normalization tensor
- Optional train-time augmentations

## 6.2 Feature Extraction (`inference/feature_extractor.py`)
Extracts a rich mixed feature set:
- **Texture:** GLCM properties + LBP histogram
- **Statistical:** channel-wise mean/var/skew/kurtosis/entropy
- **Shape:** regionprops + Hu moments + approximate Zernike-style terms
- **Thermal (if available):** asymmetry, hotspot burden, gradients, entropy, dispersion
- **Deep features:** EfficientNet backbone embedding
- **Latent features:** convolutional autoencoder bottleneck vector

## 6.3 Model Registry (`inference/model_registry.py`)
Thread-safe singleton with lazy loading:
- Autoencoder, EfficientNet classifier, ViT classifier
- XGBoost thermal classifier (+ scaler wrapper if JSON export available)
- Ensemble fusion object
- Explainability engine
- Runtime preflight:
  - manifest/metadata checks
  - version compatibility
  - checksum integrity validation

## 6.4 Prediction Logic (`inference/predictor.py`)
- Ensures model artifacts exist (with zip-sync fallback attempt)
- Runs preprocess -> feature extraction
- Gets image probabilities from EfficientNet and ViT
- If thermal matrix exists and XGBoost is available:
  - computes 8D thermal vector
  - checks feature count compatibility (`n_features_in_`)
  - gets thermal class probabilities
- Sends branches to ensemble fusion
- Produces explainability outputs:
  - Grad-CAM++ overlay (base64)
  - SHAP bar-figure dict
  - LIME overlay image
- Adds latency, model version, reasoning payload

## 6.5 Ensemble (`inference/ensemble.py`)
- Fixed weighted fusion:
  - EfficientNet 0.40
  - ViT 0.35
  - XGBoost 0.25
- If thermal missing: uses neutral thermal prior `[1/3, 1/3, 1/3]`
- Applies temperature-scaled softmax
- Computes:
  - class probabilities
  - confidence (max prob)
  - reliability (entropy-derived certainty proxy)
  - risk score (malignant probability × 100)
  - risk level via thresholds (Low <35, Medium <65, else High)

---

## 7. Explainability System

`inference/explainability.py`:
- **Grad-CAM++** over CNN model (fallback: raw image if package unavailable)
- **SHAP** for tree model feature attributions; robust fallback values if explainer fails
- **LIME** local image perturbation explanations (fallback: original image)
- Returns structured outputs for direct UI rendering

Explainability is integrated in both:
- diagnostic payload (`result["xai"]`)
- report narrative context (`top_features`, xai summaries)

---

## 8. LLM Integration

## 8.1 Client (`llm/ollama_client.py`)
- Uses Ollama with host discovery and model availability checks
- Primary model: `mistral:latest`
- Fallback model: `llama3:latest`
- Supports both non-streamed and streamed generation
- Returns graceful fallback text when unreachable

## 8.2 Prompting (`llm/prompts.py`)
- Report prompt asks for 11-section clinically structured markdown
- Patient explanation prompt requires safe, supportive language
- Chat system prompt constrains behavior:
  - explain clearly
  - avoid definitive diagnosis
  - encourage professional consultation

## 8.3 Where LLM is used
- `report_generator_agent.py`: full markdown report + patient-friendly explanation
- `chat_agent.py`: ongoing contextual assistant answers based on report

---

## 9. Training Pipeline

## 9.1 Dataset Discovery (`training/dataset_loader.py`)
- Recursively scans dataset directory
- Supports label inference by folder/token patterns
- Supports pairing image + thermal by normalized sample key
- Produces unified dataframe with `sample_id`, `label`, `image_path`, `thermal_path`
- Stratified split: 70/15/15

## 9.2 Image Models (`training/train_image_models.py`)
- EfficientNet-B4 classifier head replacement and training with weighted loss
- ViT-B/16 fine-tuning with differential learning rates and mixup
- Class imbalance handling:
  - minority oversampling in dataframe
  - weighted random sampling in dataloader
- Metrics:
  - accuracy, macro-F1, multiclass ROC-AUC, confusion matrix, classification report
- Saves:
  - `efficientnet_model.pth`
  - `vit_model.pth`
  - `ensemble_meta_model.pkl` (logistic regression meta-learner trained on val outputs)

## 9.3 Thermal Model (`training/train_thermal_model.py`)
- Engineers thermal features from matrices
- Thermal augmentation with flips/rotation/scale/noise/contrast
- XGBoost pipeline with scaler
- Saves:
  - `xgboost_thermal.pkl`
  - `scaler.pkl`

## 9.4 Segmentation and Autoencoder
- `train_segmentation.py`: Attention U-Net using pseudo masks from GrabCut
- `train_autoencoder.py`: convolutional autoencoder reconstruction training
- Saves:
  - `unet_segmentation.pth`
  - `autoencoder.pth`

## 9.5 Orchestrator (`training/train_all.py`)
- Trains missing models unless `FORCE_RETRAIN=True`
- Writes `evaluation_report.json` including dataset stats + available metrics

---

## 10. Model Artifacts, Compatibility, and Reliability Engineering

## 10.1 Manifest/Metadata Strategy
- `models/model_versions.json` is authoritative manifest:
  - model version
  - expected sklearn/torch compatibility
  - checksums
- Runtime preflight validates:
  - library compatibility
  - file integrity (SHA256)

## 10.2 Artifact Sync
- `utils/model_sync.py` searches for model/artifact zip files and extracts required files if missing
- Enables easier deployment portability across machines

## 10.3 Metrics Present in Repository
- `models/evaluation_report.json` (overall training outputs)
- `models/xgboost_metrics.json` (per-model metrics)
- Example overall report includes ~0.94 accuracy and macro-F1 ~0.94 (artifact-dependent)

---

## 11. Requested Evaluation Points (Detailed)

## 11.1 Problem Understanding — Clarity of Problem & Objectives
**What problem it solves:**  
Support breast cancer screening triage by combining image morphology and thermal patterns into interpretable risk outputs.

**Objectives visible in code:**
1. Provide end-to-end inference from upload to report.
2. Fuse multi-model evidence (CNN + ViT + thermal XGBoost).
3. Improve trust with explainability and reliability indicators.
4. Keep workflow practical through Streamlit UI, exportable reports, and clinician/patient-friendly explanations.

**Clarity assessment:**  
Strong overall objective clarity. The project consistently frames itself as assistive and not definitive diagnosis, reflected in prompts, UI language, and fallback messaging.

## 11.2 Dataset Handling — Data Cleaning, Preprocessing
**Data handling strengths:**
- Recursive dataset discovery with automatic label inference.
- Image/thermal modality pairing through sample-key normalization.
- Invalid/corrupt images filtered during dataset construction.
- Stratified train/val/test split to preserve class ratios.
- Runtime upload validation: file type, file size, image minimum dimensions, thermal dimensionality checks.

**Preprocessing quality:**
- CLAHE + equalization + denoising + foreground extraction + segmentation-style masking gives robust ROI-focused inputs.
- Thermal branch computes physically meaningful summary features (asymmetry/hotspots/entropy/gradients).

**Observations:**
- Segmentation uses pseudo-labels (GrabCut-derived), good for bootstrap but weaker than manually curated masks.
- Dataset label inference by path tokens is flexible but depends on naming consistency.

## 11.3 Model Selection — Justification of Model Choice
**Selected models and rationale:**
- **EfficientNet-B4:** strong CNN baseline for fine-grained texture/shape patterns in medical images.
- **ViT-B/16:** complementary global-context modeling, often useful for structural distribution cues.
- **XGBoost (thermal features):** robust tabular learner for handcrafted thermal descriptors.
- **Ensemble fusion:** combines heterogeneous inductive biases for higher stability than a single model.
- **Autoencoder:** latent representation for richer deep features.
- **Attention U-Net:** segmentation-focused ROI support.

**Assessment:**  
Model choices are coherent for multi-modal clinical AI prototype design. The combination balances deep representation learning with interpretable tabular thermal modeling.

## 11.4 Implementation — Correctness & Completeness
**Correctness indicators from implementation:**
- Full state schema defined and carried through graph nodes.
- Conditional thermal path handling avoids hard-failure when modality is missing.
- Background job isolation avoids Streamlit UI blocking.
- Consistent logging and pipeline audit traces across nodes.
- Safe model loading with compatibility checks and artifact integrity verification.

**Completeness indicators:**
- Includes training, inference, UI, explainability, and report generation in one repository.
- Includes portability scripts and model artifact management.
- Includes downloadable outputs and chat transcript persistence.

**Gaps / caveats:**
- Some fallback branches return generic content instead of calibrated alternatives.
- Meta-model artifact is trained, but runtime ensemble path currently relies on weighted fusion rather than explicit meta-model inference logic.

## 11.5 Performance — Accuracy, Metrics, Evaluation
**Available evaluation artifacts show:**
- Overall reported performance around high-0.9 range for accuracy/macro-F1 in included JSON artifacts.
- Per-model metrics are tracked with confusion matrices and classification reports.
- Multiclass AUC-OVR also reported.

**Performance engineering in code:**
- Mixed precision training path (`torch.cuda.amp`) for efficiency.
- Weighted sampling + class weighting + augmentation for imbalance mitigation.
- Early stopping and cosine LR scheduling.
- Cached model registry and lazy loading to reduce repeated overhead.

**Interpretation caution:**
- Metrics are artifact-dependent and tied to discovered dataset composition.
- Benign class support appears much lower than malignant in included stats, so class-specific reliability should be interpreted with care.

## 11.6 Innovation — Creativity / Improvements
**Innovative elements already present:**
- Multi-agent LangGraph pipeline for deterministic stage orchestration.
- Multi-modal fallback logic (works even without thermal input).
- Hybrid explainability stack (visual + feature attribution + local perturbation).
- LLM-generated structured report + separate patient-friendly explanation + conversational assistant.
- Model manifest compatibility + checksum guardrails uncommon in basic seminar projects.

**Potential improvement directions:**
1. Integrate calibrated meta-model inference path at runtime (currently weighted fusion dominant).
2. Add external validation + domain-shift checks.
3. Upgrade from pseudo segmentation masks to expert-annotated masks.
4. Add uncertainty quantification beyond entropy-based reliability (e.g., MC dropout/deep ensemble spread).
5. Add systematic experiment tracking (e.g., run registry with fixed seed/version snapshots).

---

## 12. Data, Outputs, and Artifacts Summary

## 12.1 Expected Inputs
- Image files: `.jpg/.jpeg/.png/.bmp/.tif/.tiff`
- Thermal files: `.csv/.npy/.txt` (2D matrices)

## 12.2 Runtime Output Surfaces
- UI probability/risk panels
- Grad-CAM heatmap overlay
- SHAP figure dict rendered as plotly
- LIME explanation image
- Markdown AI report
- PDF report export
- Chat transcript export

## 12.3 Stored Outputs
- Uploads in `breast_cancer_detection/uploads/`
- Generated report/chat files in `breast_cancer_detection/outputs/`
- App logs in `breast_cancer_detection/logs/app.log`

---

## 13. Dependencies and Environment

The project combines:
- **DL/CV:** torch, torchvision, timm, opencv, scikit-image, albumentations
- **Classical ML:** scikit-learn, xgboost, lightgbm
- **XAI:** grad-cam, shap, lime
- **LLM stack:** langgraph/langchain + ollama
- **Frontend:** streamlit, plotly, matplotlib/seaborn
- **Reporting:** reportlab + markdown

Python target in `pyproject.toml`: `>=3.10`.

---

## 14. Final Technical Assessment

This repository is a **comprehensive, end-to-end clinical AI prototype** with stronger engineering maturity than a typical seminar demo. It covers training, multimodal inference, model management, explainability, reporting, and user interaction in one coherent system. The strongest parts are pipeline completeness, model-compatibility safeguards, and explainability integration. The main next maturity steps are stronger external validation, tighter calibration strategy integration, and richer uncertainty modeling.
