# Prism AI Results Integration Notes

## Safe Sources

- Dataset counts and sources: `kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/lineage/dataset_manifest.json`
- Training configuration: `kaggle-output/artifacts/training/yolo/waste-seg-yolo/training_runs/waste-seg-yolo/args.yaml` and `kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/configuration/resolved_config.json`
- Validation metrics: `kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json` and `kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv`
- Checkpoint identity: `kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`, `kaggle-output/artifacts/training/yolo/waste-seg-yolo/checkpoints/latest.json`
- Visual evidence: `kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/` and `kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/annotated_batches/`

## Values Safe to Use

- Total dataset size and split sizes from `dataset_manifest.json`.
- Overall validation box/mask precision, recall, mAP@50, and mAP@50:95 from `training_summary.json` and `results.csv`.
- Best checkpoint path `models/best.pt`.
- Optuna tuning metadata, only as tuning metadata, from `optuna_best.json`.

## Keep as Placeholders

- Test-set metrics.
- Jetson Nano FPS, latency, power, and memory.
- End-to-end mass-estimation accuracy.
- ONNX/TensorRT deployment results.
- Structured per-class precision, recall, and mAP@50:95.

## Writing Guidance

- Cite artifact paths beside every numeric claim.
- Distinguish validation segmentation metrics from full system performance.
- Discuss weak classes professionally as observed validation limitations.
- Do not overclaim deployment readiness; no export or Jetson benchmark artifact was found.
- Mark image-derived class AP@0.5 and confusion-matrix interpretations as needing confirmation if the thesis requires strict numeric tables.
