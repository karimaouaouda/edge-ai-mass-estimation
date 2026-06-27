# Missing Values and Questions for Karim

## Prioritized Missing or Ambiguous Items

| item | status | source_or_expected_artifact | notes |
| --- | --- | --- | --- |
| final_run_confirmation | missing_or_ambiguous | kaggle-output/artifacts/training/yolo/waste-seg-yolo/checkpoints/latest.json | Is the installed epoch-54 artifact the final thesis run or an intermediate output? |
| test_set_metrics | missing_or_ambiguous | kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/lineage/dataset_manifest.json | No structured test metrics were found, despite a test split in dataset_manifest.json. |
| class_level_precision_recall | missing_or_ambiguous | [REAL_VALUE_REQUIRED] | Per-class precision and recall were not found in structured CSV/JSON artifacts. |
| class_level_mAP50_95 | missing_or_ambiguous | [REAL_VALUE_REQUIRED] | Per-class mAP@50:95 values were not found in structured artifacts. |
| evaluation_json | missing_or_ambiguous | [REAL_VALUE_REQUIRED] | No reports/evaluation.json artifact was found. |
| export_artifacts | missing_or_ambiguous | kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/configuration/resolved_config.json | No ONNX, TensorRT engine, or exports_manifest.json found; resolved config has export.enabled=false. |
| jetson_latency_fps | missing_or_ambiguous | [REAL_VALUE_REQUIRED] | No Jetson Nano FPS, latency, memory, or power benchmark artifacts were found. |
| mass_estimation_metrics | missing_or_ambiguous | [REAL_VALUE_REQUIRED] | No mass-estimation MAE/RMSE/R2 or end-to-end mass evaluation artifacts were found. |
| run_start_end_timestamps | missing_or_ambiguous | [REAL_VALUE_REQUIRED] | No explicit run start/end timestamps were found in the installed artifacts. |
| separate_inference_evaluation | missing_or_ambiguous | [REAL_VALUE_REQUIRED] | No full end-to-end inference or deployment evaluation report was found. |

## Questions

- Is this installed artifact set the final thesis training run or an intermediate run?
- Should `kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/best.pt` be treated as the thesis best model?
- Was the run trained on the intended merged TACO + AquaTrash + RealWaste dataset?
- Are the reported metrics validation metrics only, or is there a separate test evaluation artifact?
- Are there test-set metrics, COCO JSON outputs, or `reports/evaluation.json` files not included in `kaggle-output`?
- Are there Jetson Nano FPS/latency/power benchmark artifacts?
- Are there mass-estimation evaluation artifacts, such as MAE, RMSE, or R2?
- Should the thesis include image-derived class AP@0.5 values from the PR curves, or wait for a structured class metrics export?
- Which figures should be included in the final thesis: results curve, Mask PR curve, confusion matrix, labels distribution, or validation examples?
- Which weak classes should be discussed in the limitations section?
- Should ONNX or TensorRT export be performed later for deployment evidence?
