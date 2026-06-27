# 05 Training Pipeline And Mlops

<!-- BEGIN TRAINING_RESULTS_ARTIFACT_ANALYSIS -->
## Training Results Artifact Analysis

The installed run artifacts under `kaggle-output` show that the governed YOLO segmentation pipeline produced dataset lineage, Optuna tuning output, checkpointed training metrics, curves, and best/last model checkpoints. Key source files are `kaggle-output/mlruns/1/e5e5723dba6c44e490577a327c338ee1/artifacts/lineage/dataset_manifest.json`, `kaggle-output/artifacts/training/yolo/waste-seg-yolo/reports/training_summary.json`, `kaggle-output/artifacts/training/yolo/waste-seg-yolo/training/curves/results.csv`, and `kaggle-output/artifacts/training/yolo/waste-seg-yolo/checkpoints/latest.json`.

Use the generated extension at `docs/thesis_evidence_pack/results_analysis/` for detailed traceability. The installed artifacts do not include structured test metrics, deployment exports, Jetson benchmarks, or mass-estimation metrics.
<!-- END TRAINING_RESULTS_ARTIFACT_ANALYSIS -->
