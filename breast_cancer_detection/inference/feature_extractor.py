from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torchvision.models as models
from scipy.stats import entropy, kurtosis, skew
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import moments_hu, regionprops

from breast_cancer_detection.utils.logger import get_logger

logger = get_logger("feature_extractor")


class ConvAutoencoder(torch.nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, 3, stride=2, padding=1),
            torch.nn.ReLU(),
        )
        self.bottleneck = torch.nn.Linear(512 * 14 * 14, latent_dim)
        self.expand = torch.nn.Linear(latent_dim, 512 * 14 * 14)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.bottleneck(h.flatten(1))
        out = self.expand(z).view(-1, 512, 14, 14)
        return self.decoder(out), z


class FeatureExtractor:
    def __init__(
        self,
        autoencoder_path: Optional[Path] = None,
        autoencoder_model: Optional[torch.nn.Module] = None,
        efficientnet_backbone: Optional[torch.nn.Module] = None,
        device: Optional[str] = None,
    ):
        from breast_cancer_detection.inference.model_registry import ModelRegistry

        registry = ModelRegistry.get_instance()
        self.device = device or registry.device
        self.efficientnet = (efficientnet_backbone or registry.get_feature_backbone()).to(self.device).eval()
        if autoencoder_model is not None:
            self.autoencoder = autoencoder_model.to(self.device).eval()
        else:
            if autoencoder_path is None:
                autoencoder_path = None
            try:
                self.autoencoder = registry.get_autoencoder()
            except Exception:
                # Fallback to local load if registry fails and explicit path provided.
                self.autoencoder = ConvAutoencoder().to(self.device).eval()
                if autoencoder_path and autoencoder_path.exists():
                    from breast_cancer_detection.inference.model_loader import load_autoencoder_model

                    self.autoencoder = load_autoencoder_model(
                        self.autoencoder, autoencoder_path, device=self.device, logger_instance=logger
                    )

    def _glcm_features(self, gray: np.ndarray) -> Dict[str, float]:
        distances = [1, 3, 5]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
        props = {}
        for name in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
            props[name] = float(np.mean(graycoprops(glcm, name)))
        return props

    def _lbp_features(self, gray: np.ndarray) -> np.ndarray:
        lbp = local_binary_pattern(gray, P=24, R=3, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=26, range=(0, 26), density=True)
        return hist.astype(np.float32)

    def _stat_features(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, float]:
        stats: Dict[str, float] = {}
        for i, channel_name in enumerate(["r", "g", "b"]):
            ch = image[:, :, i].astype(np.float32)
            stats[f"{channel_name}_mean"] = float(np.mean(ch))
            stats[f"{channel_name}_var"] = float(np.var(ch))
            stats[f"{channel_name}_skew"] = float(skew(ch.reshape(-1)))
            stats[f"{channel_name}_kurtosis"] = float(kurtosis(ch.reshape(-1)))
            hist, _ = np.histogram(ch, bins=64, range=(0, 255), density=True)
            stats[f"{channel_name}_entropy"] = float(entropy(hist + 1e-9))
        g = gray.astype(np.float32)
        stats["gray_mean"] = float(np.mean(g))
        stats["gray_var"] = float(np.var(g))
        stats["gray_skew"] = float(skew(g.reshape(-1)))
        stats["gray_kurtosis"] = float(kurtosis(g.reshape(-1)))
        hist, _ = np.histogram(g, bins=64, range=(0, 255), density=True)
        stats["gray_entropy"] = float(entropy(hist + 1e-9))
        return stats

    def _shape_features(self, gray: np.ndarray) -> Dict[str, float]:
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        props = regionprops((bw > 0).astype(np.uint8))
        if not props:
            return {k: 0.0 for k in ["area", "perimeter", "eccentricity", "solidity"] + [f"hu_{i}" for i in range(7)] + [f"zernike_{i}" for i in range(8)]}
        region = max(props, key=lambda p: p.area)
        features = {
            "area": float(region.area),
            "perimeter": float(region.perimeter),
            "eccentricity": float(region.eccentricity),
            "solidity": float(region.solidity),
        }
        hu = moments_hu(region.image.astype(np.float32))
        for i, v in enumerate(hu):
            features[f"hu_{i}"] = float(v)
        # Lightweight Zernike approximation terms
        yy, xx = np.indices(region.image.shape)
        cy, cx = np.array(region.image.shape) / 2.0
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (max(region.image.shape) / 2 + 1e-9)
        theta = np.arctan2(yy - cy, xx - cx)
        mask = r <= 1
        for n in range(8):
            z = np.mean((r[mask] ** n) * np.cos(n * theta[mask]))
            features[f"zernike_{n}"] = float(z)
        return features

    def _thermal_features(self, thermal: np.ndarray) -> Dict[str, float]:
        left = thermal[:, : thermal.shape[1] // 2]
        right = thermal[:, thermal.shape[1] // 2 :]
        gx = cv2.Sobel(thermal, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(thermal, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.sqrt(gx**2 + gy**2)
        thr = thermal.mean() + 2 * thermal.std()
        hot_spots = np.sum(thermal > thr)
        hist, _ = np.histogram(thermal.reshape(-1), bins=64, density=True)
        return {
            "thermal_min": float(np.min(thermal)),
            "thermal_max": float(np.max(thermal)),
            "thermal_mean": float(np.mean(thermal)),
            "thermal_std": float(np.std(thermal)),
            "thermal_asymmetry": float(abs(np.mean(left) - np.mean(right))),
            "thermal_hot_spots": float(hot_spots),
            "thermal_grad_mean": float(np.mean(grad)),
            "thermal_entropy": float(entropy(hist + 1e-9)),
        }

    def _deep_features(self, image: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        with torch.no_grad():
            f = self.efficientnet.features(x)
            f = self.efficientnet.avgpool(f).flatten(1).cpu().numpy().squeeze()
        if f.size > 256:
            indices = np.linspace(0, f.size - 1, 256).astype(int)
            f = f[indices]
        return f.astype(np.float32)

    def _latent_features(self, image: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        with torch.no_grad():
            _, z = self.autoencoder(x)
        return z.cpu().numpy().squeeze().astype(np.float32)

    def extract_all(self, image: np.ndarray, thermal_matrix: Optional[np.ndarray] = None) -> Dict[str, object]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features: Dict[str, object] = {}
        features["glcm"] = self._glcm_features(gray)
        features["lbp"] = self._lbp_features(gray)
        features["statistical"] = self._stat_features(image, gray)
        features["shape"] = self._shape_features(gray)
        if thermal_matrix is not None:
            features["thermal"] = self._thermal_features(thermal_matrix.astype(np.float32))
        else:
            features["thermal"] = {}
        features["deep_cnn"] = self._deep_features(image)
        features["autoencoder_latent"] = self._latent_features(image)
        return features

