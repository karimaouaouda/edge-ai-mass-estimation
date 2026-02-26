"""Dataset classes for training the mass regression model and fine-tuning detection.

Provides:
- ``MassDataset``: tabular (features + mass label) for the regression head.
- ``WasteDetectionDataset``: image + bboxes + masks for YOLO / detection training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class MassDataset(Dataset):
    """Tabular dataset: each row is a feature vector + ground-truth mass (kg).

    Expected CSV columns (order matters):
        bbox_w, bbox_h, area, mask_area, mask_ratio,
        depth_mean, depth_median, depth_std, depth_min, depth_max,
        ..., class_id, mass_kg
    """

    def __init__(self, csv_path: str | Path, input_dim: int = 32) -> None:
        import pandas as pd

        df = pd.read_csv(csv_path)
        self.labels = torch.tensor(df["mass_kg"].values, dtype=torch.float32)
        self.class_ids = torch.tensor(df["class_id"].values, dtype=torch.long)

        feature_cols = [c for c in df.columns if c not in ("mass_kg", "class_id")]
        raw = df[feature_cols].values.astype(np.float32)
        # Pad or truncate to input_dim
        if raw.shape[1] < input_dim:
            pad = np.zeros((raw.shape[0], input_dim - raw.shape[1]), dtype=np.float32)
            raw = np.concatenate([raw, pad], axis=1)
        else:
            raw = raw[:, :input_dim]
        self.features = torch.tensor(raw)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "class_id": self.class_ids[idx],
            "mass_kg": self.labels[idx],
        }


class WasteDetectionDataset(Dataset):
    """Image dataset for detection / segmentation training.

    Works with the YOLO-style directory layout::

        images/
            train/  val/  test/
        labels/
            train/  val/  test/
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        transform: Any = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.image_paths = sorted(self.images_dir.glob("*.[jp][pn]g"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import cv2

        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        label_path = self.labels_dir / img_path.with_suffix(".txt").name

        labels: list[list[float]] = []
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                parts = [float(x) for x in line.split()]
                labels.append(parts)

        sample = {"image": image, "labels": labels, "path": str(img_path)}
        if self.transform:
            sample = self.transform(sample)
        return sample
