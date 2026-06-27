# Curves and Visual Evidence

The installed run includes quantitative curves, confusion matrices, label diagnostics, dataset mosaics, and annotated train/validation batches. Visual interpretation is secondary evidence; numeric claims should cite CSV/JSON artifacts when available.

| figure | artifact_path | what_it_shows | thesis_use | concerns |
| --- | --- | --- | --- | --- |
| results.png | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.png | training/validation losses and aggregate box/mask metrics across epochs | supports learning-curve discussion | losses trend downward and mAP trends upward; recall and validation losses show some jitter; overfitting claim needs confirmation |
| MaskPR_curve.png | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/MaskPR_curve.png | mask precision-recall curves and image-derived class AP@0.5 values | best figure for segmentation class-level discussion | plastic_bag, mixed_waste, and rigid_plastic have weaker AP@0.5 than organic_waste and paper_cardboard |
| BoxPR_curve.png | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/BoxPR_curve.png | box precision-recall curves and image-derived class AP@0.5 values | compare detection versus segmentation behavior | class ranking is similar to mask curve; image-derived values need confirmation |
| confusion_matrix_normalized.png | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/confusion_matrix_normalized.png | normalized class confusion including background | supports class confusion and false-negative discussion | background row is prominent for plastic_bag, rigid_plastic, glass, and mixed_waste; visual interpretation needs confirmation |
| labels.jpg | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/diagnostics/labels.jpg | training-set class distribution and bounding-box geometry | supports class imbalance discussion | mixed_waste has more training instances than glass or plastic_bottle |
| val_batch0_pred.jpg and val_batch0_labels.jpg | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/annotated_batches/val_batch0_pred.jpg; kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/annotated_batches/val_batch0_labels.jpg | side-by-side qualitative prediction and label examples | illustrate successes, missed detections, class substitutions, and cluttered scenes | small visual sample; do not generalize without a larger qualitative audit |

## Supported Visual Observations

- The loss and mAP curves in `results.png` are consistent with learning over the installed epochs.
- `labels.jpg` and `dataset_manifest.json` show class imbalance, with mixed_waste and rigid_plastic having more instances than glass or plastic_bottle.
- `MaskPR_curve.png` and `BoxPR_curve.png` show stronger AP@0.5 for organic_waste and paper_cardboard, and weaker AP@0.5 for plastic_bag and mixed_waste.
- The normalized confusion matrix suggests background-related misses for several classes, especially plastic_bag, rigid_plastic, glass, and mixed_waste; this is image-derived and should be confirmed if a numeric confusion matrix becomes available.
