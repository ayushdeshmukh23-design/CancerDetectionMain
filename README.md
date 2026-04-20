# Seminar Project - OncoVision AI

Follow these exact steps on a fresh PC.

## 1. Clone the repository

```bash
git clone https://github.com/ayushdeshmukh23-design/Seminar-Project-.git
cd Seminar-Project-
```

## 2. Create and prepare virtual environment

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
bash ./scripts/setup_env.sh
source .venv/bin/activate
```

## 3. Configure environment variables

Create `.env` in repository root with:

```env
OLLAMA_NUM_CTX=2048
OLLAMA_NUM_PREDICT=256
```

## 4. Ensure Ollama is running and models are available

```bash
ollama pull mistral
ollama pull llama3
```

## 5. Place data and model artifacts

1. Put dataset in:
   - `Datasets/images/{normal,benign,malignant}`
   - `Datasets/thermal_matrices/{normal,benign,malignant}`
2. Put model artifacts in:
   - `breast_cancer_detection/models/`

Required artifacts:
- `efficientnet_model.pth`
- `vit_model.pth`
- `xgboost_thermal.pkl`
- `ensemble_meta_model.pkl`
- `unet_segmentation.pth`
- `autoencoder.pth`
- `scaler.pkl`
- `model_versions.json`

## 6. Train models (skip if artifacts already prepared)

```bash
python -m breast_cancer_detection.training.train_all
```

## 7. Run the app

```bash
streamlit run breast_cancer_detection/app.py
```

## 8. Quick validation after start

1. Open Upload page and run one analysis.
2. Confirm output appears in:
   - Analysis
   - Results
   - Dashboard
3. Open Chat and ask one question to verify Ollama response.
