"""Fine-tune YOLOv8 on a custom waste detection dataset.

Usage::

    python scripts/train_detector.py --config configs/models/yolov8_training.yaml
"""

from __future__ import annotations

import argparse

from edge_ai_mass.utils.config import load_config
from edge_ai_mass.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)

    from ultralytics import YOLO

    model = YOLO(cfg.get("model", "yolov8n-seg.pt"))
    model.train(
        data=cfg["data"],
        epochs=cfg.get("epochs", 100),
        batch=cfg.get("batch", 16),
        imgsz=cfg.get("imgsz", 640),
        patience=cfg.get("patience", 20),
        device=cfg.get("device", 0),
        project=cfg.get("project", "runs/train"),
        name=cfg.get("name", "yolov8-waste"),
        augment=cfg.get("augment", True),
        mosaic=cfg.get("mosaic", 1.0),
        mixup=cfg.get("mixup", 0.1),
        copy_paste=cfg.get("copy_paste", 0.3),
        optimizer=cfg.get("optimizer", "AdamW"),
        lr0=cfg.get("lr0", 0.001),
        lrf=cfg.get("lrf", 0.01),
        weight_decay=cfg.get("weight_decay", 0.0005),
    )


if __name__ == "__main__":
    main()
