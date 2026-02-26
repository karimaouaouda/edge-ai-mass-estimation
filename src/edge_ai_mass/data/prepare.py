"""Data preparation utilities — download, convert, and split datasets.

Supports:
- TrashNet download and conversion to YOLO format.
- TACO download and conversion.
- Custom dataset splitting (train / val / test).
- Feature extraction for mass regression training.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def create_splits(
    data_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[Path]]:
    """Split image files into train / val / test sets.

    Creates subdirectories under *output_dir*: ``images/{train,val,test}``
    and ``labels/{train,val,test}`` (if label files exist alongside images).
    """
    rng = np.random.default_rng(seed)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    images = sorted(data_dir.glob("*.[jp][pn]g"))
    rng.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    for split_name, paths in splits.items():
        img_dir = output_dir / "images" / split_name
        lbl_dir = output_dir / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for src in paths:
            shutil.copy2(src, img_dir / src.name)
            label_src = src.with_suffix(".txt")
            if label_src.exists():
                shutil.copy2(label_src, lbl_dir / label_src.name)

    logger.info(
        "Split %d images → train=%d  val=%d  test=%d",
        n, len(splits["train"]), len(splits["val"]), len(splits["test"]),
    )
    return splits


def generate_yolo_dataset_yaml(
    output_dir: str | Path,
    class_names: list[str],
    yaml_path: str | Path | None = None,
) -> Path:
    """Generate a YOLO-format dataset.yaml file."""
    import yaml

    output_dir = Path(output_dir)
    yaml_path = Path(yaml_path) if yaml_path else output_dir / "dataset.yaml"

    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path.write_text(yaml.dump(data, default_flow_style=False))
    logger.info("Dataset YAML written to %s", yaml_path)
    return yaml_path
