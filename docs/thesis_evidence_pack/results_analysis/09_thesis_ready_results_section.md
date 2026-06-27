# Experimental Results: YOLO Segmentation Training

## Experimental Setup

A YOLO segmentation model was trained for the waste-characterization stage of the Edge AI mass-estimation pipeline. The installed artifacts identify the task as `segment`, the base checkpoint as `yolo26m-seg.pt`, the dataset configuration as `/kaggle/working/data/processed/waste_seg_yolo/dataset.yaml`, image size as 640, batch size as 32, optimizer as SGD, and training device as `0` (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/training_runs/waste-seg-yolo/args.yaml`). The run name was `waste-seg-yolo` (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/training_runs/waste-seg-yolo/args.yaml`). The configured target was 60 epochs, while the installed final checkpoint records 54 completed epochs (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/checkpoints/latest.json`); this run-finality status requires confirmation.

## Dataset Summary

The training dataset merged TACO, AquaTrash, and RealWaste into eight classes: organic_waste, plastic_bottle, plastic_bag, rigid_plastic, metal_can, glass, paper_cardboard, mixed_waste (`kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/lineage/dataset_manifest.json`). The manifest records 6460 images and 9702 object instances (`kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/lineage/dataset_manifest.json`). The split sizes were 4522 training images, 969 validation images, and 969 test images (`kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/lineage/dataset_manifest.json`). No duplicate images were recorded (`kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/lineage/dataset_manifest.json`).

## Quantitative Validation Results

| Metric | Value | Source |
| --- | --- | --- |
| Box precision | 0.803843 | `kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json` |
| Box recall | 0.590039 | `kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json` |
| Box mAP@50 | 0.649014 | `kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json` |
| Box mAP@50:95 | 0.599596 | `kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json` |
| Mask precision | 0.793789 | `kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json` |
| Mask recall | 0.582536 | `kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json` |
| Mask mAP@50 | 0.627705 | `kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json` |
| Mask mAP@50:95 | 0.530008 | `kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json` |

The final installed epoch row reports mask mAP@50 = 0.6208 and mask mAP@50:95 = 0.52755 at epoch 54 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv`). The best CSV epoch for mask mAP@50:95 was epoch 44 with value 0.53016 (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv`).

## Curve and Qualitative Analysis

The loss and metric curves show decreasing training/validation losses and increasing mAP over the installed epochs (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.png`). The Mask PR curve gives image-derived AP@0.5 values, with organic_waste and paper_cardboard visually strongest and plastic_bag/mixed_waste weaker (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/MaskPR_curve.png`). The normalized confusion matrix suggests background-related misses for several classes (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/confusion_matrix_normalized.png`). Annotated validation batches provide qualitative examples of correct detections, missed objects, and category substitutions (`kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/annotated_batches/val_batch0_pred.jpg`; `kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/annotated_batches/val_batch0_labels.jpg`).

## Limitations

No structured test-set metrics were found, so test performance remains [REAL_VALUE_REQUIRED]. No Jetson Nano FPS/latency benchmark artifacts were found, so edge-deployment performance remains [REAL_VALUE_REQUIRED]. No mass-estimation metrics were found, so these YOLO segmentation results must not be presented as full mass-estimation performance. No ONNX or TensorRT export artifact was found, and the resolved configuration records export as disabled (`kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/configuration/resolved_config.json`).

## Transition to Mass Estimation

These results support the object segmentation component of the pipeline. The next evaluation step should measure how segmentation quality affects depth-assisted volume estimation and mass prediction, using separate mass-estimation and edge-inference benchmark artifacts.
