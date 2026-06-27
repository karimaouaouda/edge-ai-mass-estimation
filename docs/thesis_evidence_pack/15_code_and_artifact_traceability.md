# 15 Code And Artifact Traceability

<!-- BEGIN TRAINING_RESULTS_ARTIFACT_ANALYSIS -->
## Training Results Artifact Analysis

| Thesis claim | Artifact path | Extracted value | Status | Notes |
| --- | --- | --- | --- | --- |
| YOLO segmentation model was trained | kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json | task=segment, model=/kaggle/working/yolo26m-seg.pt | supported | Training summary and args.yaml are present. |
| Best checkpoint was produced | kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/best.pt | best.pt exists | supported | Use for thesis model identity. |
| Last checkpoint was produced | kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/last.pt | last.pt exists | supported | Use for resume/debug traceability. |
| Training curves were generated | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.png | results.png exists | supported | Use as convergence figure. |
| Evaluation metrics were produced | kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv | final epoch=54 | supported | Validation metrics only unless confirmed otherwise. |
| Dataset manifest was generated | kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/lineage/dataset_manifest.json | 6460 images; 9702 instances | supported | Contains split/source/class distribution. |
| Export artifacts were produced | [REAL_VALUE_REQUIRED] | [REAL_VALUE_REQUIRED] | missing | No ONNX/TensorRT exports or exports_manifest.json found. |
| Jetson Nano benchmark was produced | [REAL_VALUE_REQUIRED] | [REAL_VALUE_REQUIRED] | missing | No FPS/latency artifacts found. |
| Mass-estimation results were produced | [REAL_VALUE_REQUIRED] | [REAL_VALUE_REQUIRED] | missing | No mass metrics found. |
<!-- END TRAINING_RESULTS_ARTIFACT_ANALYSIS -->
