# OncoVision AI — Multi-Modal Breast Cancer Detection

Production-style end-to-end system for breast image + thermal matrix analysis using deep learning, classical ML, LangGraph agents, explainability, and Streamlit.

## Quick Start

### Option A (recommended): one-command virtual environment setup

From repository root:

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

1. Install dependencies:
   ```bash
   cd "D:\VS Codes\SeminarProject\breast_cancer_detection"
   pip install -r requirements.txt
   ```
2. Install package in editable mode (recommended for clean absolute imports):
   ```bash
   cd "D:\VS Codes\SeminarProject"
   pip install -e .
   ```
3. Pull Ollama models:
   ```bash
   ollama pull llama3
   ollama pull mistral
   ```
4. Place data under `Datasets/`.
5. Train once:
   ```bash
   python -m breast_cancer_detection.training.train_all
   ```
6. Run UI (from repository root):
   ```bash
   streamlit run breast_cancer_detection/app.py
   ```

### Run on another PC

1. Clone repository.
2. Run `scripts/setup_env.ps1` (Windows) or `scripts/setup_env.sh` (Linux/macOS).
3. Copy required model artifacts into `breast_cancer_detection/models` (if not versioned in repo).
4. Ensure Ollama is installed and running, then pull:
   ```bash
   ollama pull mistral
   ollama pull llama3
   ```
5. Start Streamlit:
   ```bash
   streamlit run breast_cancer_detection/app.py
   ```

## Notes

- `train_all.py` skips already-trained models unless `FORCE_RETRAIN=True`.
- Supports missing thermal modality during inference.
- Uses Llama3 primary and Mistral fallback for report and chat.
- Includes Grad-CAM++, SHAP, and LIME explainability surfaces.

