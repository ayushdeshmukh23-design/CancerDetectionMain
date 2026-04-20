from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from breast_cancer_detection.utils.config import CLASS_NAMES, DATASETS_DIR, IMAGE_EXTENSIONS, THERMAL_EXTENSIONS
from breast_cancer_detection.utils.logger import get_logger

logger = get_logger("dataset_loader")


@dataclass
class SampleRecord:
    sample_id: str
    label: str
    image_path: Optional[Path] = None
    thermal_path: Optional[Path] = None


def _infer_label_from_path(path: Path) -> Optional[str]:
    lowered = [p.lower() for p in path.parts]
    joined = str(path).lower()
    for cls in CLASS_NAMES:
        if cls in lowered:
            return cls
    if any(tok in joined for tok in ("healthy", "saud", "normal", "sauda", "saudave")):
        return "normal"
    if any(tok in joined for tok in ("doentes", "doent", "disease", "malign", "cancer")):
        return "malignant"
    if any(tok in joined for tok in ("benig",)):
        return "benign"
    return None


def _infer_sample_key(path: Path) -> str:
    stem = path.stem.lower()
    for token in ("_anterior", "_oblleft", "_oblright", "-esq", "-dir"):
        stem = stem.replace(token, "")
    return stem


def discover_dataset(dataset_dir: Path = DATASETS_DIR) -> pd.DataFrame:
    records: Dict[Tuple[str, str], SampleRecord] = {}
    for path in dataset_dir.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in IMAGE_EXTENSIONS and ext not in THERMAL_EXTENSIONS:
            continue
        label = _infer_label_from_path(path)
        if label is None:
            continue
        sample_key = _infer_sample_key(path)
        key = (label, sample_key)
        if key not in records:
            records[key] = SampleRecord(sample_id=sample_key, label=label)
        if ext in IMAGE_EXTENSIONS and records[key].image_path is None:
            records[key].image_path = path
        if ext in THERMAL_EXTENSIONS and records[key].thermal_path is None:
            records[key].thermal_path = path

    data = [
        {
            "sample_id": rec.sample_id,
            "label": rec.label,
            "image_path": str(rec.image_path) if rec.image_path else None,
            "thermal_path": str(rec.thermal_path) if rec.thermal_path else None,
        }
        for rec in records.values()
    ]
    df = pd.DataFrame(data)
    if df.empty:
        raise RuntimeError(f"No discoverable dataset files found in {dataset_dir}")
    logger.info("Discovered samples: %s", len(df))
    logger.info("Class counts: %s", df["label"].value_counts().to_dict())
    logger.info("Image samples: %s | Thermal samples: %s", df["image_path"].notna().sum(), df["thermal_path"].notna().sum())
    return df


def stratified_split(df: pd.DataFrame, random_state: int = 42) -> Dict[str, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=random_state,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=random_state,
    )
    return {"train": train_df.reset_index(drop=True), "val": val_df.reset_index(drop=True), "test": test_df.reset_index(drop=True)}


class ImageClassificationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, class_to_index: Optional[Dict[str, int]] = None):
        filtered = df[df["image_path"].notna()].copy()
        valid_rows = []
        for _, row in filtered.iterrows():
            image_path = Path(row["image_path"])
            if not image_path.exists():
                continue
            try:
                with Image.open(image_path) as img:
                    img.verify()
                valid_rows.append(row)
            except Exception:
                continue
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        self.transform = transform
        self.class_to_index = class_to_index or {name: i for i, name in enumerate(CLASS_NAMES)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        arr = np.array(img)
        if self.transform:
            arr = self.transform(image=arr)["image"]
        return arr, self.class_to_index[row["label"]]


def load_thermal_matrix(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".txt":
        return np.loadtxt(path)
    return np.loadtxt(path, delimiter=",")

