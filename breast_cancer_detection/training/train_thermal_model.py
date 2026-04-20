from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import albumentations as A
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from xgboost import XGBClassifier

from breast_cancer_detection.training.dataset_loader import discover_dataset, load_thermal_matrix
from breast_cancer_detection.utils.config import CLASS_NAMES
from breast_cancer_detection.utils.logger import get_logger

logger = get_logger("train_thermal_model")


def _thermal_features(matrix: np.ndarray) -> List[float]:
    matrix = matrix.astype(np.float32)
    left = matrix[:, : matrix.shape[1] // 2]
    right = matrix[:, matrix.shape[1] // 2 :]
    gx = np.gradient(matrix, axis=1)
    gy = np.gradient(matrix, axis=0)
    grad = np.sqrt(gx**2 + gy**2)
    thr = matrix.mean() + 2 * matrix.std()
    hot_spots = float(np.sum(matrix > thr))
    hist, _ = np.histogram(matrix.reshape(-1), bins=64, density=True)
    ent = float(-np.sum(hist * np.log(hist + 1e-9)))
    return [
        float(matrix.min()),
        float(matrix.max()),
        float(matrix.mean()),
        float(matrix.std()),
        float(abs(left.mean() - right.mean())),
        hot_spots,
        float(grad.mean()),
        ent,
    ]


def _augment_thermal_matrix(matrix: np.ndarray, n_aug: int = 3) -> List[np.ndarray]:
    tf = A.Compose(
        [
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.4),
            A.Rotate(limit=12, p=0.7),
            A.RandomScale(scale_limit=0.08, p=0.6),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.4),
        ]
    )
    mats = [matrix]
    m_norm = matrix.astype(np.float32)
    if np.max(m_norm) > np.min(m_norm):
        m_norm = (m_norm - np.min(m_norm)) / (np.max(m_norm) - np.min(m_norm))
    for _ in range(n_aug):
        aug = tf(image=m_norm)["image"].astype(np.float32)
        mats.append(aug)
    return mats


def train_xgboost(dataset_dir: Path, models_dir: Path):
    _ = dataset_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    df = discover_dataset()
    tdf = df[df["thermal_path"].notna()].copy()
    if tdf.empty:
        raise RuntimeError("No thermal matrices found for thermal model training.")

    X = []
    y = []
    label_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    for _, row in tdf.iterrows():
        try:
            mat = load_thermal_matrix(Path(row["thermal_path"]))
            if mat.ndim != 2:
                continue
            for aug_mat in _augment_thermal_matrix(mat, n_aug=4):
                X.append(_thermal_features(aug_mat))
                y.append(label_to_idx[row["label"]])
        except Exception:
            continue
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    class_counts = np.bincount(y_train, minlength=3)
    scale_pos_weight = float(np.max(class_counts) / max(np.min(class_counts[class_counts > 0]), 1))

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric=["mlogloss", "merror"],
        objective="multi:softprob",
        num_class=3,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    pipeline = Pipeline([("scaler", StandardScaler()), ("xgb", model)])
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)
    y_pred = y_prob.argmax(axis=1)
    y_test_bin = label_binarize(y_test, classes=np.arange(3))
    metrics: Dict[str, object] = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "test_auc_roc": float(roc_auc_score(y_test_bin, y_prob, multi_class="ovr")),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, target_names=CLASS_NAMES),
    }
    logger.info("Thermal model metrics: %s", metrics)
    joblib.dump(pipeline, models_dir / "xgboost_thermal.pkl")
    joblib.dump(pipeline.named_steps["scaler"], models_dir / "scaler.pkl")
    logger.info("Saved xgboost_thermal.pkl and scaler.pkl")
    return metrics

