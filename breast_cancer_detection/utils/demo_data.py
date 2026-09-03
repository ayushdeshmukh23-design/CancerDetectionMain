from __future__ import annotations

import base64
import io
import math
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_synthetic_mammogram_b64(
    mode: str = "original",
    lesion_type: str = "malignant",
    width: int = 380,
    height: int = 380,
) -> str:
    """Generate a realistic synthetic clinical mammogram/XAI image encoded in base64."""
    img = Image.new("RGBA", (width, height), (8, 12, 22, 255))
    draw = ImageDraw.Draw(img)

    # Breast profile outline / parenchyma
    for i in range(height):
        progress = i / height
        curve_x = int(width * (0.88 - 0.55 * (progress - 0.4) ** 2))
        intensity = int(40 + 60 * math.sin(progress * math.pi))
        draw.line([(0, i), (curve_x, i)], fill=(intensity, intensity + 5, intensity + 15, 255))

    # Add fibroglandular density variations
    for _ in range(120):
        gx = random.randint(30, int(width * 0.65))
        gy = random.randint(40, height - 50)
        gr = random.randint(15, 45)
        alpha = random.randint(20, 60)
        draw.ellipse([gx - gr, gy - gr, gx + gr, gy + gr], fill=(120, 130, 150, alpha))

    # Lesion center
    lx, ly, lr = int(width * 0.48), int(height * 0.44), 38

    if mode == "original":
        # Dense irregular lesion mass
        for r in range(lr, 0, -2):
            val = int(140 + (lr - r) * 2.8)
            draw.ellipse([lx - r, ly - r, lx + r, ly + r], fill=(val, val, val + 10, 180))
        # Spiculations
        if lesion_type == "malignant":
            for angle in range(0, 360, 25):
                rad = math.radians(angle)
                ex = lx + int((lr + random.randint(10, 26)) * math.cos(rad))
                ey = ly + int((lr + random.randint(10, 26)) * math.sin(rad))
                draw.line([(lx, ly), (ex, ey)], fill=(220, 220, 230, 140), width=2)

    elif mode == "enhanced":
        # High contrast CLAHE enhanced
        for r in range(lr + 10, 0, -2):
            val = min(255, int(160 + (lr - r) * 3.5))
            draw.ellipse([lx - r, ly - r, lx + r, ly + r], fill=(val, val + 5, min(255, val + 25), 230))
        # Enhanced micro-calcifications
        for _ in range(16):
            cx = lx + random.randint(-lr, lr)
            cy = ly + random.randint(-lr, lr)
            draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(255, 255, 255, 255))

    elif mode == "roi":
        # Cropped and focused ROI
        crop_box = [lx - lr - 15, ly - lr - 15, lx + lr + 15, ly + lr + 15]
        draw.rectangle(crop_box, outline=(0, 212, 255, 255), width=3)
        draw.text((crop_box[0] + 4, crop_box[1] + 4), "ROI: Lesion #1 (42mm)", fill=(0, 212, 255, 255))

    elif mode == "unet":
        # Pure binary & contour segmentation
        draw.ellipse([lx - lr, ly - lr, lx + lr, ly + lr], fill=(168, 85, 247, 90), outline=(168, 85, 247, 255), width=3)
        draw.text((lx - 30, ly - 8), "Dice: 0.941", fill=(255, 255, 255, 255))

    elif mode == "gradcam":
        # Grad-CAM++ activation colormap overlay
        for r in range(lr + 45, 0, -3):
            factor = (lr + 45 - r) / (lr + 45)
            # Jet/Turbomap: red at center, yellow at mid, cyan/blue at edge
            red = int(min(255, factor * 350))
            green = int(min(255, math.sin(factor * math.pi) * 255))
            blue = int(max(0, (1 - factor) * 200))
            alpha = int(factor * 170)
            draw.ellipse([lx - r, ly - r, lx + r, ly + r], fill=(red, green, blue, alpha))

    elif mode == "thermal":
        # Thermal isotherm gradient with concentric isotherms
        for r in range(lr + 60, 0, -4):
            factor = (lr + 60 - r) / (lr + 60)
            red = int(min(255, 80 + factor * 175))
            green = int(min(255, factor * 140))
            blue = int(max(0, (1 - factor) * 180))
            draw.ellipse([lx - r, ly - r, lx + r, ly + r], fill=(red, green, blue, 150))
        draw.text((lx - 35, ly - 8), "ΔT: +2.41°C", fill=(255, 255, 255, 255))

    elif mode == "lime":
        # Superpixel segmentation with green (supporting) and red (opposing) borders
        for i in range(6):
            angle = i * 60
            rad = math.radians(angle)
            sx = lx + int(lr * 0.9 * math.cos(rad))
            sy = ly + int(lr * 0.9 * math.sin(rad))
            color = (16, 185, 129, 220) if i in [0, 1, 3, 4] else (239, 68, 68, 220)
            draw.ellipse([sx - 20, sy - 20, sx + 20, sy + 20], outline=color, width=2)
            draw.text((sx - 12, sy - 6), f"+{i*12}%" if i in [0, 1, 3, 4] else f"-{i*8}%", fill=color)

    elif mode == "vit_attention":
        # Vision Transformer (ViT) Multi-Head Self-Attention Rollout (16x16 patch tokens)
        patch_size = width // 16
        for px in range(16):
            for py in range(16):
                dist = math.hypot(px * patch_size - lx, py * patch_size - ly)
                att = max(0.0, 1.0 - (dist / (lr * 2.2)))
                if att > 0.15:
                    alpha = int(att * 180)
                    r_val = int(min(255, att * 280))
                    g_val = int(min(255, (1 - att) * 160 + 80))
                    b_val = int(max(0, (1 - att) * 220))
                    x0, y0 = px * patch_size, py * patch_size
                    draw.rectangle([x0 + 1, y0 + 1, x0 + patch_size - 1, y0 + patch_size - 1], fill=(r_val, g_val, b_val, alpha), outline=(0, 212, 255, 60))
        draw.text((lx - 45, ly - 8), "ViT Attn: 0.884", fill=(255, 255, 255, 255))

    elif mode == "integrated_gradients":
        # Integrated Gradients path-interpolated pixel attribution (microcalcifications & spiculations)
        for _ in range(350):
            rx = lx + int(random.gauss(0, lr * 0.75))
            ry = ly + int(random.gauss(0, lr * 0.75))
            if 0 <= rx < width and 0 <= ry < height:
                intensity = random.randint(180, 255)
                color = (255, random.randint(100, 200), 50, random.randint(140, 230))
                draw.point((rx, ry), fill=color)
                draw.ellipse([rx - 1, ry - 1, rx + 1, ry + 1], fill=color)
        draw.text((lx - 50, ly + lr + 10), "Attr Mass: 91.2%", fill=(255, 215, 0, 255))

    elif mode == "spectral_fusion":
        # Multi-Spectral Composite: Co-registered FFDM + Thermal Radiometric Isotherm
        for r in range(lr + 50, 0, -3):
            factor = (lr + 50 - r) / (lr + 50)
            red = int(min(255, 100 + factor * 155))
            green = int(min(255, factor * 90))
            blue = int(max(0, (1 - factor) * 190))
            draw.ellipse([lx - r, ly - r, lx + r, ly + r], fill=(red, green, blue, int(factor * 130)))
        crop_box = [lx - lr - 8, ly - lr - 8, lx + lr + 8, ly + lr + 8]
        draw.rectangle(crop_box, outline=(245, 158, 11, 220), width=2)
        draw.text((crop_box[0], crop_box[1] - 14), "Fused Co-Reg (ΔT +2.4°C)", fill=(245, 158, 11, 255))

    elif mode == "edges":
        # Sobel / Canny spiculation gradient field
        for angle in range(0, 360, 15):
            rad = math.radians(angle)
            for step in range(lr - 10, lr + 25, 4):
                ex = lx + int(step * math.cos(rad) + random.randint(-2, 2))
                ey = ly + int(step * math.sin(rad) + random.randint(-2, 2))
                draw.point((ex, ey), fill=(0, 212, 255, 220))
                draw.ellipse([ex - 1, ey - 1, ex + 1, ey + 1], fill=(0, 212, 255, 180))
        draw.text((lx - 35, ly - 8), "Spiculation Map", fill=(0, 212, 255, 255))

    elif mode == "texture_map":
        # Stage 2: Texture Feature Map (GLCM / LBP gradient in vivid rainbow/jet)
        for r in range(120, 20, -6):
            c_idx = (r * 3) % 255
            # Jet style colormap: Red/Orange at edge, Yellow, Cyan, Blue
            if r > 90:
                color = (255, int((120 - r) * 7), 0, 200)
            elif r > 60:
                color = (255, 220, 0, 190)
            elif r > 40:
                color = (0, 230, 255, 180)
            else:
                color = (30, 80, 220, 190)
            draw.ellipse([lx - r - 20, ly - r + 10, lx + r + 20, ly + r - 10], outline=color, width=4)
        for i in range(25):
            tx = lx + random.randint(-90, 90)
            ty = ly + random.randint(-90, 90)
            draw.line([(tx, ty), (tx + random.randint(-15, 15), ty + random.randint(-15, 15))], fill=(0, 255, 220, 220), width=2)
        draw.text((15, 15), "GLCM Texture: 4.18", fill=(255, 255, 255, 230))

    elif mode == "deep_cnn_embedding":
        # Stage 2: Deep-CNN Embedding Map (EfficientNet proxy: 16x16 latent feature blocks)
        draw.rectangle([0, 0, width, height], fill=(15, 23, 42, 255))
        block_w = width // 16
        block_h = height // 16
        palette = [
            (30, 27, 75),   # dark indigo
            (49, 46, 129),  # indigo
            (67, 56, 202),  # purple
            (14, 116, 144), # cyan
            (13, 148, 136), # teal
            (16, 185, 129), # emerald
            (234, 179, 8),  # yellow
        ]
        rng_blocks = random.Random(42)
        for bx in range(16):
            for by in range(16):
                dist = math.hypot(bx * block_w - lx, by * block_h - ly)
                p_idx = min(len(palette) - 1, int(max(0, (1 - dist / (lr * 2.5))) * len(palette)) + rng_blocks.randint(0, 1))
                c = palette[p_idx]
                x0, y0 = bx * block_w, by * block_h
                draw.rectangle([x0 + 1, y0 + 1, x0 + block_w - 1, y0 + block_h - 1], fill=c)
        draw.text((15, 15), "Latent: [1, 1536]", fill=(255, 255, 255, 230))

    elif mode == "feature_overlay":
        # Stage 2: Feature Map Overlay (semi-transparent embedding grid on mammogram)
        block_w = width // 16
        block_h = height // 16
        palette_alpha = [
            (67, 56, 202, 70),
            (14, 116, 144, 90),
            (13, 148, 136, 110),
            (16, 185, 129, 130),
            (234, 179, 8, 150),
        ]
        for bx in range(16):
            for by in range(16):
                dist = math.hypot(bx * block_w - lx, by * block_h - ly)
                p_idx = min(len(palette_alpha) - 1, int(max(0, (1 - dist / (lr * 2.0))) * len(palette_alpha)))
                c = palette_alpha[p_idx]
                x0, y0 = bx * block_w, by * block_h
                draw.rectangle([x0 + 1, y0 + 1, x0 + block_w - 1, y0 + block_h - 1], fill=c)
        draw.text((15, 15), "Fused Features (Conv+ViT)", fill=(255, 255, 255, 230))

    elif mode == "shap_attribution_map":
        # Stage 3: SHAP-style Attribution Map (Soft magenta/pink and teal/cyan attribution clouds)
        for r in range(lr + 50, 0, -4):
            factor = (lr + 50 - r) / (lr + 50)
            red = int(min(255, 180 + factor * 75))
            green = int(min(255, 60 + factor * 80))
            blue = int(min(255, 190 + factor * 65))
            draw.ellipse([lx - r, ly - r, lx + r, ly + r], fill=(red, green, blue, int(factor * 160)))
        # Background negative attribution
        draw.rectangle([0, 0, width, height], fill=(13, 148, 136, 40))
        draw.text((15, 15), "SHAP Mass: +0.38", fill=(255, 255, 255, 230))

    # Smooth filter for realistic medical look
    img = img.filter(ImageFilter.SMOOTH_MORE)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_seeded_history(n: int = 25, seed: int = 42) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    base = 52.0
    rows: List[Dict[str, object]] = []
    start = datetime.now() - timedelta(days=n * 7)  # weekly scans
    for i in range(n):
        drift = rng.uniform(-4.0, 4.5)
        base = min(92.0, max(12.0, base + drift))
        confidence = min(0.98, max(0.65, 0.82 + rng.uniform(-0.12, 0.14)))
        reliability = min(0.98, max(0.60, 0.86 + rng.uniform(-0.15, 0.12)))
        risk_level = "Low" if base < 35 else "Moderate" if base < 65 else "High"
        birads = 2 if base < 30 else 3 if base < 50 else 4 if base < 75 else 5
        vol = round(1.2 + (base / 100.0) * 3.2 + math.sin(i * 0.4) * 0.3, 2)
        rows.append(
            {
                "idx": i + 1,
                "date": (start + timedelta(days=i * 7)).strftime("%Y-%m-%d"),
                "risk_score": round(base, 2),
                "confidence": round(confidence, 4),
                "reliability": round(reliability, 4),
                "risk_level": risk_level,
                "birads": f"BI-RADS {birads}",
                "doubling_time_days": round(rng.uniform(90, 240), 1) if base > 50 else None,
                "thermal_delta_t": round(rng.uniform(0.6, 2.9), 2),
                "tumor_volume_cm3": vol,
                "model_agreement": round(min(0.99, max(0.75, 0.90 + rng.uniform(-0.08, 0.08))), 3),
            }
        )
    return rows


def demo_diagnosis_state(case_id: str = "case_malignant") -> Dict[str, object]:
    history = generate_seeded_history(25, seed=101)
    latest = history[-1]

    if case_id == "case_benign":
        malignant, benign, normal = 0.12, 0.74, 0.14
        pred = "Benign"
        risk_score = 28.4
        risk_level = "Low"
        birads = "BI-RADS 3 (Probably Benign - 6 Month Surveillance)"
    elif case_id == "case_normal":
        malignant, benign, normal = 0.04, 0.08, 0.88
        pred = "Normal"
        risk_score = 9.2
        risk_level = "Low"
        birads = "BI-RADS 1 (Negative - Routine Screening)"
    else:
        # Default high-acuity malignant case
        malignant, benign, normal = 0.76, 0.18, 0.06
        pred = "Malignant"
        risk_score = 78.6
        risk_level = "High"
        birads = "BI-RADS 4C (High Suspicion of Malignancy - Core Biopsy Urged)"

    orig_b64 = generate_synthetic_mammogram_b64("original", "malignant")
    enh_b64 = generate_synthetic_mammogram_b64("enhanced", "malignant")
    roi_b64 = generate_synthetic_mammogram_b64("roi", "malignant")
    unet_b64 = generate_synthetic_mammogram_b64("unet", "malignant")
    cam_b64 = generate_synthetic_mammogram_b64("gradcam", "malignant")
    therm_b64 = generate_synthetic_mammogram_b64("thermal", "malignant")
    lime_b64 = generate_synthetic_mammogram_b64("lime", "malignant")
    vit_b64 = generate_synthetic_mammogram_b64("vit_attention", "malignant")
    ig_b64 = generate_synthetic_mammogram_b64("integrated_gradients", "malignant")
    fusion_b64 = generate_synthetic_mammogram_b64("spectral_fusion", "malignant")
    edges_b64 = generate_synthetic_mammogram_b64("edges", "malignant")
    texture_b64 = generate_synthetic_mammogram_b64("texture_map", "malignant")
    deep_cnn_b64 = generate_synthetic_mammogram_b64("deep_cnn_embedding", "malignant")
    overlay_b64 = generate_synthetic_mammogram_b64("feature_overlay", "malignant")
    shap_attr_b64 = generate_synthetic_mammogram_b64("shap_attribution_map", "malignant")

    now = datetime.now()

    pipeline_logs = [
        {
            "timestamp": (now - timedelta(seconds=12)).strftime("%H:%M:%S.%f")[:12],
            "stage": "Modality Ingestion",
            "level": "INFO",
            "latency_ms": 138,
            "device": "cuda:0",
            "tensor_shape": "[1, 3, 512, 512]",
            "message": "Dual-modality payload ingested: 16-bit mammogram scan (2840x2240) + 64x64 calibrated thermal matrix.",
        },
        {
            "timestamp": (now - timedelta(seconds=10)).strftime("%H:%M:%S.%f")[:12],
            "stage": "CLAHE Preprocessing",
            "level": "INFO",
            "latency_ms": 94,
            "device": "cpu",
            "tensor_shape": "[1, 3, 512, 512]",
            "message": "Adaptive histogram equalization completed. ClipLimit=2.0, TileGridSize=(8,8). Dynamic range enhanced.",
        },
        {
            "timestamp": (now - timedelta(seconds=9)).strftime("%H:%M:%S.%f")[:12],
            "stage": "U-Net ROI Segmentation",
            "level": "INFO",
            "latency_ms": 204,
            "device": "cuda:0",
            "tensor_shape": "[1, 1, 512, 512]",
            "message": "Isolated hyperdense lesion contour at (x: 182, y: 168). Area=1420px², Dice coefficient=0.941.",
        },
        {
            "timestamp": (now - timedelta(seconds=8)).strftime("%H:%M:%S.%f")[:12],
            "stage": "Thermal Asymmetry Analysis",
            "level": "WARN",
            "latency_ms": 72,
            "device": "cpu",
            "tensor_shape": "[64, 64]",
            "message": "Contra-lateral thermogram baseline ΔT = +2.41°C. Exceeds clinical hyperthermia threshold (1.5°C).",
        },
        {
            "timestamp": (now - timedelta(seconds=6)).strftime("%H:%M:%S.%f")[:12],
            "stage": "Multimodal Feature Extraction",
            "level": "INFO",
            "latency_ms": 118,
            "device": "cuda:0",
            "tensor_shape": "[1, 1536]",
            "message": "Fused 128 GLCM/LBP texture features + 1408 EfficientNet-B4 + ViT-B/16 hybrid latent embeddings.",
        },
        {
            "timestamp": (now - timedelta(seconds=4)).strftime("%H:%M:%S.%f")[:12],
            "stage": "Deep Ensemble Inference",
            "level": "INFO",
            "latency_ms": 242,
            "device": "cuda:0",
            "tensor_shape": "[1, 3]",
            "message": "Ensemble votes: EfficientNet-B4 (78.2%), Vision Transformer (75.4%), XGBoost Metaclassifier (81.0%).",
        },
        {
            "timestamp": (now - timedelta(seconds=2)).strftime("%H:%M:%S.%f")[:12],
            "stage": "Explainability (XAI) Synthesis",
            "level": "INFO",
            "latency_ms": 320,
            "device": "cuda:0",
            "tensor_shape": "N/A",
            "message": "Synthesized Grad-CAM++ layer4 gradients, ViT self-attention rollout, TreeSHAP force attributions, and LIME perturbation mask.",
        },
        {
            "timestamp": now.strftime("%H:%M:%S.%f")[:12],
            "stage": "OpenRouter Report Generation",
            "level": "INFO",
            "latency_ms": 860,
            "device": "openrouter:api",
            "tensor_shape": "N/A",
            "message": "Clinical structured screening report synthesized via meta-llama/llama-3.1-8b-instruct.",
        },
    ]

    markdown_report = f"""# ONCOVISION AI CLINICAL SCREENING REPORT
**Patient Study Date:** {now.strftime("%B %d, %Y")} | **Evaluation Protocol:** Multi-Modal Dual-Fusion v1.0
**Target Modalities:** Digital Mammography (DICOM MLO) + Infrared Thermography (FLIR Matrix)

---

### 1. CLINICAL VERDICT & TRIAGE ASSESSMENT
- **Primary AI Verdict:** **{pred.upper()}** (Suspicious for Neoplastic Infiltration)
- **Clinical BI-RADS Category:** **{birads}**
- **Composite Risk Score:** **{risk_score:.1f} / 100** (Tier 1 Priority)
- **Calibrated Ensemble Confidence:** **{(malignant if pred == 'Malignant' else benign)*100:.1f}%**
- **System Reliability Metric:** **94.2%** (Entropy Dispersion: 0.12)

---

### 2. ANATOMICAL & LESION LOCALIZATION
- **Laterality & View:** Right Breast, Mediolateral Oblique (R-MLO)
- **Clock Position:** 10:00 quadrant (Upper Outer Quadrant - UOQ)
- **Lesion Morphology:** Irregular architectural distortion with micro-lobulated and partially spiculated margins.
- **Max Diameter:** Estimated at **22.4 mm × 18.6 mm**.

---

### 3. DUAL-MODALITY EVIDENCE FUSION
- **Structural Mammography:** Hyperdense focal asymmetry identified in retro-areolar region extending into axillary tail. Micro-calcifications observed in clustered configuration.
- **Infrared Thermography:** Focal hyperthermic emission demonstrating positive contra-lateral thermal asymmetry of **ΔT = +2.41°C** (Normal threshold < 1.0°C). Indicative of localized neo-angiogenesis.

---

### 4. EXPLAINABLE AI (XAI) CONFIRMATION
- **Grad-CAM++ Spatial Attention:** 89.4% of total gradient mass localized specifically to the spiculated margins of the primary lesion.
- **TreeSHAP Feature Contributions:** Top drivers for malignant classification include *Vascular Asymmetry Index (+0.38)*, *Marginal Spiculation (+0.31)*, and *Thermal Delta-T (+0.28)*.
- **LIME Local Boundary Integrity:** Superpixel perturbation isolates peripheral microcalcification cluster as strongest evidence supporting malignancy.

---

### 5. RECOMMENDATIONS & CLINICAL MANAGEMENT
1. **Immediate Histopathological Correlation:** Ultrasound-guided core needle biopsy (14-gauge) recommended for definitive tissue sampling.
2. **Diagnostic Breast Ultrasound:** Targeted sonographic evaluation of Right UOQ and ipsilateral axillary lymph node stations (Level I/II).
3. **Multidisciplinary Tumor Board Review:** Case presentation recommended following biopsy confirmation.
"""

    return {
        "case_id": case_id,
        "birads": birads,
        "preprocessed_image": {
            "original": f"data:image/png;base64,{orig_b64}",
            "enhanced": f"data:image/png;base64,{enh_b64}",
            "roi": f"data:image/png;base64,{roi_b64}",
            "edges": f"data:image/png;base64,{edges_b64}",
            "spectral_fusion": f"data:image/png;base64,{fusion_b64}",
        },
        "segmentation_mask": f"data:image/png;base64,{unet_b64}",
        "gradcam_heatmap": cam_b64,
        "thermal_heatmap": therm_b64,
        "lime_explanation": f"data:image/png;base64,{lime_b64}",
        "vit_attention_map": vit_b64,
        "integrated_gradients_map": ig_b64,
        "spectral_fusion_map": fusion_b64,
        "agent_vision_outputs": {
            "input_image": f"data:image/png;base64,{orig_b64}",
            "stage1_segmentation": f"data:image/png;base64,{unet_b64}",
            "stage1_roi": f"data:image/png;base64,{roi_b64}",
            "stage2_texture": f"data:image/png;base64,{texture_b64}",
            "stage2_deep_cnn": f"data:image/png;base64,{deep_cnn_b64}",
            "stage2_overlay": f"data:image/png;base64,{overlay_b64}",
            "stage3_gradcam": f"data:image/png;base64,{cam_b64}",
            "stage3_lime": f"data:image/png;base64,{lime_b64}",
            "stage3_shap": f"data:image/png;base64,{shap_attr_b64}",
        },
        "shap_values": {
            "Vascular Flow Index": 0.38,
            "Thermal Delta-T": 0.31,
            "Marginal Spiculation": 0.28,
            "GLCM Contrast": 0.24,
            "Core Attenuation": 0.19,
            "Microcalcification Density": 0.15,
            "Perimeter-to-Area Ratio": -0.09,
            "Elliptical Eccentricity": -0.14,
        },
        "radiomics_profile": {
            "glcm_contrast": 3.42,
            "glcm_homogeneity": 0.61,
            "glcm_entropy": 4.18,
            "spiculation_index": 0.84,
            "compactness": 0.39,
            "perimeter_area_ratio": 0.28,
            "thermal_delta_t": 2.41,
            "vascular_flow_index": 0.73,
        },
        "uncertainty": {
            "epistemic": 0.048,
            "aleatoric": 0.034,
            "confidence_interval_95": [0.742, 0.828],
            "entropy": 0.142,
        },
        "model_concordance": {
            "cohens_kappa": 0.932,
            "fleiss_kappa": 0.914,
            "consensus_agreement_pct": 96.4,
        },
        "doubling_time_analytics": {
            "doubling_time_days": 142.5,
            "current_volume_cm3": 3.84,
            "baseline_volume_cm3": 1.62,
            "growth_rate_pct_month": 14.8,
            "kinetic_model": "Schwartz Exponential Model",
        },
        "calibration_metrics": {
            "brier_score": 0.042,
            "ece_pct": 2.1,
            "calibration_slope": 0.984,
        },
        "pipeline_logs": pipeline_logs,
        "ensemble_result": {
            "prediction": pred,
            "confidence": malignant if pred == "Malignant" else benign,
            "reliability": 0.942,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "birads": birads,
            "tnm_stage_est": "cT2 N0 M0 (Stage IIA)",
            "probabilities": {
                "Normal": float(normal),
                "Benign": float(benign),
                "Malignant": float(malignant),
            },
            "model_contributions": {
                "EfficientNet-B4": {"Normal": 0.05, "Benign": 0.17, "Malignant": 0.78, "weight": 0.35},
                "Vision Transformer (ViT)": {"Normal": 0.04, "Benign": 0.20, "Malignant": 0.76, "weight": 0.35},
                "XGBoost Metaclassifier": {"Normal": 0.08, "Benign": 0.11, "Malignant": 0.81, "weight": 0.30},
            },
            "xai": {
                "top_features": [
                    "Contra-lateral Hyperthermia (ΔT = +2.41°C)",
                    "Marginal Spiculation Index (0.84)",
                    "Clustered Pleomorphic Microcalcifications",
                    "GLCM Architectural Entropy (4.18)",
                    "ViT Patch Attention Concentration (0.884)",
                ]
            },
        },
        "markdown_report": markdown_report,
        "llm_response": (
            "Based on the multi-modal screening, the AI detected an area in the upper outer right breast "
            "with both tissue density changes and localized heat asymmetry (+2.4°C). In clinical guidelines, "
            "this corresponds to a BI-RADS 4C finding, which strongly warrants a targeted ultrasound and core biopsy "
            "to clarify the exact nature of the tissue. Please discuss these findings directly with your physician."
        ),
        "seeded_history": history,
        "chat_history": [],
    }
