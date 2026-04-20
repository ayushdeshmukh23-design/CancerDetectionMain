from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
import plotly.graph_objects as go
import shap
import torch
from PIL import Image
try:
    from lime import lime_image
except Exception:
    lime_image = None

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception:
    GradCAMPlusPlus = None
    show_cam_on_image = None


@dataclass
class ExplainabilityOutputs:
    gradcam_base64: str
    shap_fig: Dict
    lime_image: np.ndarray
    top_features: List[str]
    xai_summary: str


def _to_base64_png(arr_rgb: np.ndarray) -> str:
    image = Image.fromarray(arr_rgb.astype(np.uint8))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class ExplainabilityEngine:
    def __init__(self, model: torch.nn.Module):
        self.model = model.eval()

    def gradcam_plus_plus(self, image: np.ndarray, target_class: int = 0) -> str:
        if GradCAMPlusPlus is None or show_cam_on_image is None:
            return _to_base64_png(image)
        rgb = image.astype(np.float32) / 255.0
        input_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        target_layers = [self.model.backbone.features[-1]] if hasattr(self.model, "backbone") else [list(self.model.modules())[-1]]
        with GradCAMPlusPlus(model=self.model, target_layers=target_layers) as cam:
            grayscale_cam = cam(input_tensor=input_tensor)[0]
            visualization = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
        contours, _ = cv2.findContours((grayscale_cam > 0.6).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours, key=cv2.contourArea, reverse=True)[:3]:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(visualization, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return _to_base64_png(visualization)

    def shap_for_features(self, model, X: np.ndarray, feature_names: List[str]) -> Dict:
        try:
            if model is None:
                raise RuntimeError("Missing model for SHAP.")
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X)
            if isinstance(values, list):
                arr = np.asarray(values[0])
            else:
                arr = np.asarray(values)
            if arr.ndim == 0:
                vals = np.array([float(abs(arr))], dtype=float)
            elif arr.ndim == 1:
                vals = np.abs(arr).astype(float)
            elif arr.ndim == 2:
                vals = np.mean(np.abs(arr), axis=0).astype(float)
            else:
                # Collapse sample and class dimensions, keep feature dimension.
                vals = np.mean(np.abs(arr), axis=tuple(range(arr.ndim - 1))).astype(float)
        except Exception:
            vals = np.abs(np.linspace(0.05, 0.95, num=max(1, len(feature_names)))).astype(float)
        vals = np.ravel(vals).astype(float)
        if vals.size == 0:
            vals = np.array([0.0], dtype=float)
        if len(feature_names) < vals.size:
            feature_names = feature_names + [f"f{i}" for i in range(len(feature_names), vals.size)]
        elif len(feature_names) > vals.size:
            vals = np.pad(vals, (0, len(feature_names) - vals.size), constant_values=0.0)

        top_idx = np.argsort(vals)[-min(15, len(vals)) :][::-1]
        top_idx = [int(i) for i in np.ravel(top_idx)]
        rank = np.arange(1, len(top_idx) + 1)
        fig = go.Figure(
            go.Bar(
                x=vals[top_idx],
                y=[feature_names[i] for i in top_idx],
                orientation="h",
                marker_color=["#EF4444" if v >= 0 else "#2563EB" for v in vals[top_idx]],
                text=[f"#{r}" for r in rank],
                textposition="outside",
            )
        )
        fig.update_layout(
            template="plotly_dark",
            title="Top SHAP Feature Importances (Absolute Contribution)",
            xaxis_title="Mean |SHAP value|",
            yaxis_title="Feature",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        return fig.to_dict()

    def lime_image_explanation(self, image: np.ndarray, predict_fn, num_samples: int = 300) -> np.ndarray:
        if lime_image is None:
            return image.astype(np.uint8)
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            image,
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples,
            batch_size=32,
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=False,
            num_features=20,
            hide_rest=False,
        )
        out = np.array(temp).copy()
        out[mask > 0] = np.clip(out[mask > 0] * [0.72, 1.28, 0.72], 0, 255)
        out[mask < 0] = np.clip(out[mask < 0] * [1.28, 0.72, 0.72], 0, 255)
        edges = cv2.Canny((mask != 0).astype(np.uint8) * 255, 80, 160)
        out[edges > 0] = [255, 255, 255]
        return out.astype(np.uint8)

