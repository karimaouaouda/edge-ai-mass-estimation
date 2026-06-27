# Validation and Segmentation Metrics

The installed artifacts contain validation metrics from the YOLO training run. A test split exists in `dataset_manifest.json`, but no structured test-set evaluation report or `reports/evaluation.json` was found.

## Bounding-Box Detection Performance

- Best-model box precision: 0.803843 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)
- Best-model box recall: 0.590039 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)
- Best-model box mAP@50: 0.649014 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)
- Best-model box mAP@50:95: 0.599596 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)

## Mask/Segmentation Performance

- Best-model mask precision: 0.793789 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)
- Best-model mask recall: 0.582536 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)
- Best-model mask mAP@50: 0.627705 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)
- Best-model mask mAP@50:95: 0.530008 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)
- Final installed epoch mask mAP@50: 0.6208 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv`, epoch 54)
- Final installed epoch mask mAP@50:95: 0.52755 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv`, epoch 54)

## Validation Versus Test Status

- Validation metrics are present in `results.csv`, `training_summary.json`, and checkpoint metric files.
- Test metrics are missing from the installed artifacts and must remain `[REAL_VALUE_REQUIRED]` until a test evaluation artifact is supplied.
- The presence of `dataset_manifest.json` test split counts does not prove test-set model performance.
