# Selected Figures for Thesis

Recommended figures are ranked by thesis value and traceability.

| rank | figure | artifact_path | why |
| --- | --- | --- | --- |
| 1 | results.png | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.png | Best overview of convergence, losses, and overall validation metrics. |
| 2 | MaskPR_curve.png | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/MaskPR_curve.png | Most relevant class-level segmentation evidence; includes image-derived AP@0.5 by class. |
| 3 | confusion_matrix_normalized.png | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/confusion_matrix_normalized.png | Useful for discussing class confusion and background-related misses. |
| 4 | labels.jpg | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/diagnostics/labels.jpg | Supports dataset imbalance and object geometry discussion. |
| 5 | val_batch0_pred.jpg with val_batch0_labels.jpg | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/annotated_batches/val_batch0_pred.jpg; kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/annotated_batches/val_batch0_labels.jpg | Provides qualitative prediction examples and visible failure modes. |

Do not treat visual figure legends as a replacement for structured test metrics. For class AP@0.5 values read from PR curve legends, use `needs confirmation` if the final thesis requires fully machine-readable per-class metrics.
