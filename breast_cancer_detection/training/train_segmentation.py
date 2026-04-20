from __future__ import annotations

from pathlib import Path
from typing import Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from breast_cancer_detection.training.dataset_loader import discover_dataset
from breast_cancer_detection.utils.logger import get_logger

logger = get_logger("train_segmentation")


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + identity)


class AttentionGate(nn.Module):
    def __init__(self, g_ch: int, x_ch: int, inter: int):
        super().__init__()
        self.wg = nn.Conv2d(g_ch, inter, 1)
        self.wx = nn.Conv2d(x_ch, inter, 1)
        self.psi = nn.Conv2d(inter, 1, 1)

    def forward(self, g, x):
        if g.shape[-2:] != x.shape[-2:]:
            g = torch.nn.functional.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        h = torch.relu(self.wg(g) + self.wx(x))
        a = torch.sigmoid(self.psi(h))
        return x * a


class AttentionUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ResidualBlock(1, 64)
        self.enc2 = ResidualBlock(64, 128)
        self.enc3 = ResidualBlock(128, 256)
        self.enc4 = ResidualBlock(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.center = ResidualBlock(512, 512)
        self.up4 = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.att4 = AttentionGate(512, 512, 256)
        self.dec4 = ResidualBlock(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att3 = AttentionGate(256, 256, 128)
        self.dec3 = ResidualBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att2 = AttentionGate(128, 128, 64)
        self.dec2 = ResidualBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att1 = AttentionGate(64, 64, 32)
        self.dec1 = ResidualBlock(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        d4 = self.up4(c)
        if d4.shape[-2:] != e4.shape[-2:]:
            e4 = torch.nn.functional.interpolate(e4, size=d4.shape[-2:], mode="bilinear", align_corners=False)
        e4a = self.att4(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4a], dim=1))
        d3 = self.up3(d4)
        if d3.shape[-2:] != e3.shape[-2:]:
            e3 = torch.nn.functional.interpolate(e3, size=d3.shape[-2:], mode="bilinear", align_corners=False)
        e3a = self.att3(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3a], dim=1))
        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            e2 = torch.nn.functional.interpolate(e2, size=d2.shape[-2:], mode="bilinear", align_corners=False)
        e2a = self.att2(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2a], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            e1 = torch.nn.functional.interpolate(e1, size=d1.shape[-2:], mode="bilinear", align_corners=False)
        e1a = self.att1(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1a], dim=1))
        return self.out(d1)


def _grabcut_pseudo_mask(img: np.ndarray) -> np.ndarray:
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (5, 5, img.shape[1] - 10, img.shape[0] - 10)
    cv2.grabCut(img, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == 2) | (mask == 0), 0, 1).astype(np.float32)


class SegDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = [p for p in image_paths if Path(p).exists()]
        self.tf = A.Compose(
            [
                A.Resize(256, 256),
                A.ElasticTransform(alpha=1.0, sigma=50, p=0.3),
                A.GridDistortion(p=0.3),
                A.Normalize(mean=(0.5,), std=(0.5,)),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = np.array(Image.open(self.image_paths[idx]).convert("L"))
        except Exception:
            img = np.zeros((256, 256), dtype=np.uint8)
        img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        m = _grabcut_pseudo_mask(img3)
        out = self.tf(image=img, mask=m)
        return out["image"].float(), out["mask"].unsqueeze(0).float()


def _dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    denom = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1 - ((2 * inter + eps) / (denom + eps)).mean()


def train_unet(dataset_dir: Path, models_dir: Path):
    _ = dataset_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    df = discover_dataset()
    image_paths = df[df["image_path"].notna()]["image_path"].tolist()
    if not image_paths:
        raise RuntimeError("No images found for segmentation training.")
    ds = SegDataset(image_paths)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AttentionUNet().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(4):
        losses = []
        for x, m in tqdm(loader, desc=f"UNet Epoch {epoch+1}/4"):
            x, m = x.to(device), m.to(device)
            optim.zero_grad(set_to_none=True)
            pred = model(x)
            if pred.shape[-2:] != m.shape[-2:]:
                pred = torch.nn.functional.interpolate(pred, size=m.shape[-2:], mode="bilinear", align_corners=False)
            loss = bce(pred, m) + _dice_loss(pred, m)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        logger.info("epoch=%s seg_loss=%.4f", epoch + 1, float(np.mean(losses)))
    torch.save(model.state_dict(), models_dir / "unet_segmentation.pth")
    logger.info("Saved unet_segmentation.pth")

