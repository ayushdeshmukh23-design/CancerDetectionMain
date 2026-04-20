from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from torchvision import transforms

from breast_cancer_detection.utils.logger import get_logger

logger = get_logger("preprocessor")


@dataclass
class PreprocessorConfig:
    image_size: int = 224
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple = (8, 8)
    canny_low: int = 50
    canny_high: int = 150


class PreprocessingPipeline:
    def __init__(self, segmentation_model: Optional[torch.nn.Module] = None, training: bool = False):
        self.cfg = PreprocessorConfig()
        self.segmentation_model = segmentation_model
        self.training = training
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.augment = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomAffine(degrees=15, scale=(0.9, 1.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ]
        )

    def _load_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Unable to load image: {image_path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _simulate_thermal(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return cv2.cvtColor(thermal, cv2.COLOR_BGR2RGB)

    def _grabcut_foreground(self, image: np.ndarray) -> np.ndarray:
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        return image * mask2[:, :, np.newaxis]

    def _segment_roi(self, image: np.ndarray) -> np.ndarray:
        if self.segmentation_model is None:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            _, pseudo = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return pseudo
        with torch.no_grad():
            tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0)
            pred = self.segmentation_model(tensor)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            pred = torch.sigmoid(pred)
            mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
            return mask

    def process(self, image_path: str) -> Dict[str, np.ndarray]:
        logger.info("Preprocessing image: %s", image_path)
        image = self._load_image(image_path)
        resized = cv2.resize(image, (self.cfg.image_size, self.cfg.image_size))
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.cfg.clahe_clip_limit, tileGridSize=self.cfg.clahe_tile_grid)
        l2 = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)
        ycrcb = cv2.cvtColor(enhanced, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        equalized = cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2RGB)
        denoised = cv2.GaussianBlur(equalized, (3, 3), 1)
        edges = cv2.Canny(cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY), self.cfg.canny_low, self.cfg.canny_high)
        thermal_sim = self._simulate_thermal(denoised)
        foreground = self._grabcut_foreground(denoised)
        segmentation_mask = self._segment_roi(foreground)
        roi = cv2.bitwise_and(foreground, foreground, mask=segmentation_mask)

        tensor = torch.from_numpy(roi).permute(2, 0, 1).float() / 255.0
        if self.training:
            tensor = self.augment(roi)
        normalized = self.normalize(tensor)

        return {
            "original": image,
            "resized": resized,
            "enhanced": enhanced,
            "equalized": equalized,
            "denoised": denoised,
            "edges": edges,
            "thermal_sim": thermal_sim,
            "foreground": foreground,
            "segmentation_mask": segmentation_mask,
            "roi": roi,
            "normalized_tensor": normalized,
        }

