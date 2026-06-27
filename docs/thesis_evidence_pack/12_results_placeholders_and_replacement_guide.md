# 12 Results Placeholders And Replacement Guide

<!-- BEGIN TRAINING_RESULTS_ARTIFACT_ANALYSIS -->
## Extracted YOLO Segmentation Training Results

Artifact root: `kaggle-output`

| Value | Extracted result | Source |
| --- | --- | --- |
| Dataset size | 6460 images, 9702 instances | `kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/lineage/dataset_manifest.json` |
| Train/val/test images | 4522/969/969 | `kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/lineage/dataset_manifest.json` |
| Task | segment | `kaggle-output/artifacts/training/yolo/waste-seg-yolo/training_runs/waste-seg-yolo/args.yaml` |
| Model checkpoint | /kaggle/working/yolo26m-seg.pt | `kaggle-output/artifacts/training/yolo/waste-seg-yolo/training_runs/waste-seg-yolo/args.yaml` |
| Completed epochs | 54 | `kaggle-output/artifacts/training/yolo/waste-seg-yolo/checkpoints/latest.json` |
| Best-model mask mAP@50 | 0.627705 | `kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json` |
| Best-model mask mAP@50:95 | 0.530008 | `kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json` |
| Best checkpoint | `kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/best.pt` | `kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json` |

Values still missing: test metrics, class-level structured precision/recall and mAP@50:95, ONNX/TensorRT exports, Jetson Nano benchmarks, and mass-estimation metrics.
<!-- END TRAINING_RESULTS_ARTIFACT_ANALYSIS -->
