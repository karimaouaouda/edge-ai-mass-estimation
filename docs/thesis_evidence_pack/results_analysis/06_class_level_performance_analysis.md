# Class-Level Performance Analysis

Structured per-class metric CSV/JSON artifacts were not found. However, the exported PR curve images include readable class AP@0.5 values in their legends. The values below are therefore image-derived and marked as needing confirmation.

| class_name | precision | recall | mAP50 | mAP50_95 | mask_mAP50 | mask_mAP50_95 | source_file | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| organic_waste | [REAL_VALUE_REQUIRED] | [REAL_VALUE_REQUIRED] | 0.981 | [REAL_VALUE_REQUIRED] | 0.981 | [REAL_VALUE_REQUIRED] | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/BoxPR_curve.png; kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/MaskPR_curve.png | AP@0.5 values read from PR curve legends; structured per-class precision/recall and mAP@50:95 were not found, needs confirmation. |
| plastic_bottle | [REAL_VALUE_REQUIRED] | [REAL_VALUE_REQUIRED] | 0.64 | [REAL_VALUE_REQUIRED] | 0.629 | [REAL_VALUE_REQUIRED] | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/BoxPR_curve.png; kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/MaskPR_curve.png | AP@0.5 values read from PR curve legends; structured per-class precision/recall and mAP@50:95 were not found, needs confirmation. |
| plastic_bag | [REAL_VALUE_REQUIRED] | [REAL_VALUE_REQUIRED] | 0.356 | [REAL_VALUE_REQUIRED] | 0.328 | [REAL_VALUE_REQUIRED] | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/BoxPR_curve.png; kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/MaskPR_curve.png | AP@0.5 values read from PR curve legends; structured per-class precision/recall and mAP@50:95 were not found, needs confirmation. |
| rigid_plastic | [REAL_VALUE_REQUIRED] | [REAL_VALUE_REQUIRED] | 0.531 | [REAL_VALUE_REQUIRED] | 0.464 | [REAL_VALUE_REQUIRED] | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/BoxPR_curve.png; kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/MaskPR_curve.png | AP@0.5 values read from PR curve legends; structured per-class precision/recall and mAP@50:95 were not found, needs confirmation. |
| metal_can | [REAL_VALUE_REQUIRED] | [REAL_VALUE_REQUIRED] | 0.66 | [REAL_VALUE_REQUIRED] | 0.65 | [REAL_VALUE_REQUIRED] | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/BoxPR_curve.png; kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/MaskPR_curve.png | AP@0.5 values read from PR curve legends; structured per-class precision/recall and mAP@50:95 were not found, needs confirmation. |
| glass | [REAL_VALUE_REQUIRED] | [REAL_VALUE_REQUIRED] | 0.712 | [REAL_VALUE_REQUIRED] | 0.713 | [REAL_VALUE_REQUIRED] | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/BoxPR_curve.png; kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/MaskPR_curve.png | AP@0.5 values read from PR curve legends; structured per-class precision/recall and mAP@50:95 were not found, needs confirmation. |
| paper_cardboard | [REAL_VALUE_REQUIRED] | [REAL_VALUE_REQUIRED] | 0.814 | [REAL_VALUE_REQUIRED] | 0.811 | [REAL_VALUE_REQUIRED] | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/BoxPR_curve.png; kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/MaskPR_curve.png | AP@0.5 values read from PR curve legends; structured per-class precision/recall and mAP@50:95 were not found, needs confirmation. |
| mixed_waste | [REAL_VALUE_REQUIRED] | [REAL_VALUE_REQUIRED] | 0.498 | [REAL_VALUE_REQUIRED] | 0.447 | [REAL_VALUE_REQUIRED] | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/BoxPR_curve.png; kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/MaskPR_curve.png | AP@0.5 values read from PR curve legends; structured per-class precision/recall and mAP@50:95 were not found, needs confirmation. |

## Strong and Weak Classes

- Stronger mask AP@0.5 classes from `MaskPR_curve.png`: organic_waste, paper_cardboard, glass.
- Weaker mask AP@0.5 classes from `MaskPR_curve.png`: plastic_bag, mixed_waste, rigid_plastic.
- The normalized confusion matrix diagonal appears strongest for organic_waste and paper_cardboard, and weaker for plastic_bag, mixed_waste, and rigid_plastic (`confusion_matrix_normalized.png`).

## Possible Reasons

- Class imbalance is supported by `dataset_manifest.json` and `labels.jpg`.
- Plastic bags and mixed waste may be visually heterogeneous or deformable; this is a hypothesis and needs Karim confirmation.
- Background confusion and missed objects are visible in `confusion_matrix_normalized.png` and validation batch examples, but exact false-positive/false-negative counts were not found.
