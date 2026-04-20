from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from breast_cancer_detection.training.dataset_loader import discover_dataset
from breast_cancer_detection.utils.logger import get_logger

logger = get_logger("train_autoencoder")


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.bottleneck = nn.Linear(512 * 14 * 14, latent_dim)
        self.expand = nn.Linear(latent_dim, 512 * 14 * 14)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.bottleneck(h.flatten(1))
        h2 = self.expand(z).view(-1, 512, 14, 14)
        return self.decoder(h2), z


class AutoDataset(Dataset):
    def __init__(self, image_paths):
        valid = []
        for p in image_paths:
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is not None:
                valid.append(p)
        self.image_paths = valid
        self.tf = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out = self.tf(image=img)["image"]
        return out.float()


def train_autoencoder(dataset_dir: Path, models_dir: Path):
    _ = dataset_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    df = discover_dataset()
    image_paths = df[df["image_path"].notna()]["image_path"].tolist()
    if not image_paths:
        raise RuntimeError("No images found for autoencoder training.")
    ds = AutoDataset(image_paths)
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvAutoencoder().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(5):
        losses = []
        for x in tqdm(loader, desc=f"AE Epoch {epoch+1}/5"):
            x = x.to(device)
            optim.zero_grad(set_to_none=True)
            rec, _ = model(x)
            loss = loss_fn(rec, x)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        logger.info("epoch=%s recon_loss=%.4f", epoch + 1, float(np.mean(losses)))
    torch.save(model.state_dict(), models_dir / "autoencoder.pth")
    logger.info("Saved autoencoder.pth")

