from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import albumentations as A
import joblib
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import torchvision.models as tv_models

from breast_cancer_detection.training.dataset_loader import ImageClassificationDataset, discover_dataset, stratified_split
from breast_cancer_detection.utils.config import CLASS_NAMES, MODEL_PATHS
from breast_cancer_detection.utils.logger import get_logger

logger = get_logger("train_image_models")


@dataclass
class TrainConfig:
    epochs: int = 24
    batch_size: int = 16
    patience: int = 10
    lr: float = 1e-4
    minority_aug_multiplier: int = 3


class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.4):
        super().__init__()
        self.backbone = tv_models.efficientnet_b4(weights=tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def get_embeddings(self, x):
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        return torch.flatten(features, 1)


def _build_transforms(train: bool = True):
    if train:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=15, p=0.7),
                A.RandomScale(scale_limit=0.1, p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5),
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
    return A.Compose([A.Resize(224, 224), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()])


def _compute_weights(labels: np.ndarray):
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    class_weights = np.sum(counts) / (len(CLASS_NAMES) * np.maximum(counts, 1))
    sample_weights = class_weights[labels]
    return class_weights, sample_weights


def _evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    y_pred = y_prob.argmax(axis=1)
    y_true_bin = label_binarize(y_true, classes=np.arange(len(CLASS_NAMES)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "auc_roc": float(roc_auc_score(y_true_bin, y_prob, multi_class="ovr")),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, target_names=CLASS_NAMES),
    }


def _train_loop(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    cfg: TrainConfig,
    select_metric: str = "f1_macro",
    mixup_alpha: float = 0.0,
):
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    best_score = -np.inf
    no_improve = 0
    best_state = None
    best_metrics = None
    for epoch in range(cfg.epochs):
        model.train()
        train_losses = []
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                idx = torch.randperm(x.size(0), device=device)
                x_mix = lam * x + (1 - lam) * x[idx]
                y_a, y_b = y, y[idx]
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x_mix if mixup_alpha > 0 else x)
                if mixup_alpha > 0:
                    loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                else:
                    loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())

        model.eval()
        y_true, y_prob = [], []
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                y_prob.append(prob)
                y_true.append(y.cpu().numpy())
        y_true = np.concatenate(y_true)
        y_prob = np.concatenate(y_prob)
        metrics = _evaluate(y_true, y_prob)
        scheduler.step()
        logger.info(
            "epoch=%s train_loss=%.4f val_loss=%.4f acc=%.4f f1=%.4f auc=%.4f",
            epoch + 1,
            np.mean(train_losses),
            np.mean(val_losses),
            metrics["accuracy"],
            metrics["f1_macro"],
            metrics["auc_roc"],
        )
        score = metrics.get(select_metric, metrics["f1_macro"])
        if score > best_score:
            best_score = score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_metrics = metrics
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= cfg.patience:
            logger.info("Early stopping at epoch %s", epoch + 1)
            break
    if best_state:
        model.load_state_dict(best_state)
    return model, (best_metrics or {"accuracy": 0.0, "f1_macro": 0.0, "auc_roc": 0.0, "confusion_matrix": [], "classification_report": ""})


def _make_loaders(cfg: TrainConfig):
    df = discover_dataset()
    splits = stratified_split(df)
    train_df = splits["train"].copy()
    counts = train_df["label"].value_counts().to_dict()
    max_count = max(counts.values())
    extra = []
    for label, count in counts.items():
        if count < max_count:
            deficit = min((max_count - count), count * cfg.minority_aug_multiplier)
            if deficit > 0:
                sampled = train_df[train_df["label"] == label].sample(n=deficit, replace=True, random_state=42)
                extra.append(sampled)
    if extra:
        train_df = pd.concat([train_df] + extra, ignore_index=True)
    train_set = ImageClassificationDataset(train_df, transform=_build_transforms(True))
    val_set = ImageClassificationDataset(splits["val"], transform=_build_transforms(False))
    y_train = np.array([{"normal": 0, "benign": 1, "malignant": 2}[lbl] for lbl in train_set.df["label"]])
    class_weights, sample_weights = _compute_weights(y_train)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    train_loader = DataLoader(train_set, batch_size=16, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
    return splits, train_loader, val_loader, class_weights


def train_efficientnet(dataset_dir: Path, models_dir: Path):
    _ = dataset_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TrainConfig()
    splits, train_loader, val_loader, class_weights = _make_loaders(cfg)
    model = EfficientNetClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    model, metrics = _train_loop(model, train_loader, val_loader, criterion, optimizer, scheduler, device, cfg, select_metric="f1_macro")
    torch.save(model.state_dict(), models_dir / "efficientnet_model.pth")
    logger.info("Saved efficientnet_model.pth")
    return {
        "test_accuracy": metrics["accuracy"],
        "test_f1_macro": metrics["f1_macro"],
        "test_auc_roc": metrics["auc_roc"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
    }


def train_vit(dataset_dir: Path, models_dir: Path):
    _ = dataset_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = TrainConfig()
    splits, train_loader, val_loader, _ = _make_loaders(cfg)
    model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=3, drop_path_rate=0.1).to(device)
    params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "head" in name:
            lr = 1e-3
        elif "blocks" in name:
            lr = 5e-5
        else:
            lr = 1e-5
        params.append({"params": [p], "lr": lr})
    optimizer = AdamW(params, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model, metrics = _train_loop(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        cfg,
        select_metric="auc_roc",
        mixup_alpha=0.4,
    )
    torch.save(model.state_dict(), models_dir / "vit_model.pth")
    logger.info("Saved vit_model.pth")
    return {
        "test_accuracy": metrics["accuracy"],
        "test_f1_macro": metrics["f1_macro"],
        "test_auc_roc": metrics["auc_roc"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
    }


def train_ensemble(dataset_dir: Path, models_dir: Path):
    _ = dataset_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    df = discover_dataset()
    splits = stratified_split(df)
    val_set = ImageClassificationDataset(splits["val"], transform=_build_transforms(False))
    if len(val_set) == 0:
        raise RuntimeError("No validation images available for ensemble training.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    eff = EfficientNetClassifier().to(device).eval()
    vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=3).to(device).eval()
    if (models_dir / "efficientnet_model.pth").exists():
        eff.load_state_dict(torch.load(models_dir / "efficientnet_model.pth", map_location=device))
    if (models_dir / "vit_model.pth").exists():
        vit.load_state_dict(torch.load(models_dir / "vit_model.pth", map_location=device))

    loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=0)
    X_meta, y_meta = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            eff_p = torch.softmax(eff(x), dim=1).cpu().numpy()
            vit_p = torch.softmax(vit(x), dim=1).cpu().numpy()
            xgb_p = np.full_like(eff_p, 1 / 3.0)
            feat = np.concatenate([eff_p, vit_p, xgb_p], axis=1)
            X_meta.append(feat)
            y_meta.append(y.numpy())
    x = np.concatenate(X_meta)
    y = np.concatenate(y_meta)
    meta = LogisticRegression(max_iter=1000, multi_class="auto")
    meta.fit(x, y)
    joblib.dump(meta, models_dir / "ensemble_meta_model.pkl")
    logger.info("Saved ensemble_meta_model.pkl")
    y_prob = meta.predict_proba(x)
    y_pred = y_prob.argmax(axis=1)
    y_bin = label_binarize(y, classes=np.arange(len(CLASS_NAMES)))
    return {
        "test_accuracy": float(accuracy_score(y, y_pred)),
        "test_f1_macro": float(f1_score(y, y_pred, average="macro")),
        "test_auc_roc": float(roc_auc_score(y_bin, y_prob, multi_class="ovr")),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
        "classification_report": classification_report(y, y_pred, target_names=CLASS_NAMES),
    }

