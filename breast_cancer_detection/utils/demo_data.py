from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List


def generate_seeded_history(n: int = 25, seed: int = 42) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    base = 46.0
    rows: List[Dict[str, object]] = []
    start = datetime.now() - timedelta(days=n)
    for i in range(n):
        drift = rng.uniform(-4.5, 4.5)
        base = min(90.0, max(8.0, base + drift))
        confidence = min(0.98, max(0.55, 0.72 + rng.uniform(-0.18, 0.18)))
        reliability = min(0.98, max(0.45, 0.75 + rng.uniform(-0.22, 0.2)))
        risk_level = "Low" if base < 35 else "Medium" if base < 65 else "High"
        rows.append(
            {
                "idx": i + 1,
                "date": (start + timedelta(days=i)).strftime("%Y-%m-%d"),
                "risk_score": round(base, 2),
                "confidence": round(confidence, 4),
                "reliability": round(reliability, 4),
                "risk_level": risk_level,
            }
        )
    return rows


def demo_diagnosis_state() -> Dict[str, object]:
    history = generate_seeded_history(25, seed=100)
    latest = history[-1]
    malignant = min(0.95, max(0.05, latest["risk_score"] / 100.0))
    benign = max(0.03, min(0.7, 0.28 + (0.5 - abs(0.5 - malignant)) * 0.18))
    normal = max(0.02, 1.0 - malignant - benign)
    total = normal + benign + malignant
    normal, benign, malignant = normal / total, benign / total, malignant / total
    pred = "Malignant" if malignant >= max(normal, benign) else "Benign" if benign >= normal else "Normal"

    return {
        "preprocessed_image": {},
        "segmentation_mask": None,
        "pipeline_logs": [
            "Seeded demo mode active (no uploaded image).",
            "Preprocessing checkpoint simulated.",
            "Feature extraction checkpoint simulated.",
            "Inference and report generation checkpoint simulated.",
        ],
        "ensemble_result": {
            "prediction": pred,
            "confidence": float(latest["confidence"]),
            "reliability": float(latest["reliability"]),
            "risk_level": latest["risk_level"],
            "risk_score": float(latest["risk_score"]),
            "probabilities": {
                "Normal": float(normal),
                "Benign": float(benign),
                "Malignant": float(malignant),
            },
            "model_contributions": {
                "efficientnet": {"Normal": float(normal * 0.95), "Benign": float(benign * 1.03), "Malignant": float(malignant * 1.02)},
                "vit": {"Normal": float(normal * 1.04), "Benign": float(benign * 0.97), "Malignant": float(malignant * 0.99)},
                "xgboost": {"Normal": float(normal * 1.01), "Benign": float(benign * 1.01), "Malignant": float(malignant * 0.98)},
            },
        },
        "markdown_report": (
            "## Executive Summary\n"
            "This is a seeded demonstration report for UI continuity.\n\n"
            "## AI Diagnosis & Confidence\n"
            f"- Predicted class: **{pred}**\n"
            f"- Confidence: **{latest['confidence']:.1%}**\n"
            f"- Risk score: **{latest['risk_score']:.1f}/100**\n\n"
            "## Recommendation\n"
            "Upload a patient image from the app header and run full analysis for real inference."
        ),
        "shap_values": {},
        "lime_explanation": None,
        "gradcam_heatmap": None,
        "seeded_history": history,
        "chat_history": [],
    }

