# Results Analysis Evidence Pack

This folder contains a thesis-ready extension for the installed YOLO segmentation training artifacts. It was generated from:

`kaggle-output`

The analysis uses only values found in the installed artifacts. Missing values remain explicit placeholders such as `[REAL_VALUE_REQUIRED]` or `[NOT_FOUND_IN_ARTIFACTS]`. Prism AI must not invent missing training, test, deployment, or mass-estimation results.

## Extracted Values

- Dataset size: 6460 images and 9702 instances (kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/lineage/dataset_manifest.json).
- Final installed epoch row: epoch 54 (kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv).
- Best-model validation mask mAP@50:95: 0.530008 (kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json).
- Best checkpoint: kaggle-output/artifacts/training/yolo/waste-seg-yolo/models/best.pt (kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json).
- Optuna objective: mask_map50_95 = 0.562886 (kaggle-output/artifacts/training/yolo/waste-seg-yolo/optimization/optuna_best.json).

## Remaining Gaps

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

## How Prism AI Should Use This Folder

- Use `03_training_metrics_analysis.md`, `04_validation_and_segmentation_metrics.md`, and `tables/overall_metrics.csv` for quantitative validation metrics.
- Use `02_dataset_manifest_analysis.md` and `tables/dataset_distribution.csv` for dataset counts, splits, source lineage, and class distribution.
- Use `05_curves_and_visual_evidence.md`, `figures_index/figure_index.md`, and `figures_index/selected_figures_for_thesis.md` for figure choices.
- Use `09_thesis_ready_results_section.md` as a paste-ready draft, but keep placeholders until Karim confirms missing values.
- Treat YOLO segmentation validation results separately from full end-to-end mass-estimation results.
