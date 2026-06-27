# Training Metrics Analysis

Primary metric source: `kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv`  
Best-model summary source: `kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`

## Final Installed Epoch Metrics

| metric | value | source | epoch |
| --- | --- | --- | --- |
| metrics/precision(B) | 0.80488 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv | 54 |
| metrics/recall(B) | 0.5701 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv | 54 |
| metrics/mAP50(B) | 0.64057 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv | 54 |
| metrics/mAP50-95(B) | 0.59791 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv | 54 |
| metrics/precision(M) | 0.79622 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv | 54 |
| metrics/recall(M) | 0.56408 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv | 54 |
| metrics/mAP50(M) | 0.6208 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv | 54 |
| metrics/mAP50-95(M) | 0.52755 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv | 54 |
| train/box_loss | 0.36609 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv | 54 |
| train/seg_loss | 0.4904 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv | 54 |
| train/cls_loss | 1.62279 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv | 54 |
| val/box_loss | 0.28647 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv | 54 |
| val/seg_loss | 0.36691 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv | 54 |
| val/cls_loss | 2.33211 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv | 54 |

## Best Epochs by Selected Metrics

| metric | best_value | epoch | source |
| --- | --- | --- | --- |
| metrics/mAP50-95(M) | 0.53016 | 44 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv |
| metrics/mAP50(M) | 0.62829 | 38 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv |
| metrics/mAP50-95(B) | 0.59979 | 44 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv |
| metrics/mAP50(B) | 0.64911 | 44 | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv |

## Best-Model Summary Metrics

- Box mAP@50: 0.649014 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)
- Box mAP@50:95: 0.599596 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)
- Mask mAP@50: 0.627705 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)
- Mask mAP@50:95: 0.530008 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)
- Box precision: 0.803843 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)
- Box recall: 0.590039 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)
- Mask precision: 0.793789 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)
- Mask recall: 0.582536 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`)

## Metric Meaning Notes

- Box metrics evaluate object localization using bounding boxes.
- Mask metrics evaluate instance segmentation masks and are the more relevant results for this thesis section.
- mAP@50 uses an IoU threshold of 0.50 and is more permissive.
- mAP@50:95 averages AP over IoU thresholds from 0.50 to 0.95 and is stricter.
- Precision indicates how many predicted positives were correct.
- Recall indicates how many ground-truth objects were detected.

The table `tables/overall_metrics.csv` contains the extracted metric rows with source paths and epochs.
