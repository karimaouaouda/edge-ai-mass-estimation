"""Collect feature vectors for mass regression training.

Runs the detection + depth stages on a directory of images (with
ground-truth mass labels in a CSV), extracts per-object features,
and writes them to a training-ready CSV.

Usage::

    python scripts/collect_mass_features.py \\
        --images data/raw/mass_dataset/ \\
        --labels data/raw/mass_labels.csv \\
        --config configs/pipeline/default.yaml \\
        --output data/splits/mass_features.csv

The labels CSV must have columns: filename, class_id, mass_kg
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import cv2
import numpy as np

from edge_ai_mass.pipeline.factory import build_pipeline
from edge_ai_mass.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--config", default="configs/pipeline/default.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    setup_logging()

    # Load labels
    import pandas as pd

    labels_df = pd.read_csv(args.labels)

    pipeline = build_pipeline(args.config)
    pipeline.load_all()

    rows = []
    images_dir = Path(args.images)

    for _, row in labels_df.iterrows():
        img_path = images_dir / row["filename"]
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning("Skipping %s", img_path)
            continue

        result = pipeline.run(image)
        if not result.objects:
            logger.warning("No detections in %s", img_path)
            continue

        # Take the top-confidence detection
        obj = max(result.objects, key=lambda o: o.detection.confidence)
        d = obj.detection
        bbox = d.bbox
        w = float(bbox[2] - bbox[0])
        h = float(bbox[3] - bbox[1])
        area = w * h
        mask_area = float(np.sum(d.mask > 0)) if d.mask is not None else area
        mask_ratio = mask_area / max(area, 1.0)
        ds = obj.depth_stats

        rows.append({
            "bbox_w": w,
            "bbox_h": h,
            "area": area,
            "mask_area": mask_area,
            "mask_ratio": mask_ratio,
            "depth_mean": ds.get("mean", 0),
            "depth_median": ds.get("median", 0),
            "depth_std": ds.get("std", 0),
            "depth_min": ds.get("min", 0),
            "depth_max": ds.get("max", 0),
            "class_id": int(row["class_id"]),
            "mass_kg": float(row["mass_kg"]),
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    keys = rows[0].keys()
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d feature rows to %s", len(rows), output_path)


if __name__ == "__main__":
    main()
