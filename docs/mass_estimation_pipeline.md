# Mass-estimation MLOps pipeline

The mass-estimation workflow extends the existing governed training style used
by the YOLO segmentation pipeline. It has its own config object, durable
pipeline state, optional ZenML orchestration, optional MLflow tracking, and
run-scoped artifacts.

## Hybrid residual model

The model is intentionally hybrid instead of a black-box direct regressor:

```text
mass_base_g = estimated_volume_m3 * effective_density_kg_m3 * 1000
residual_g = real_mass_g - mass_base_g
predicted_mass_g = mass_base_g + predicted_residual_g
```

`real_mass_g` is used only as the target label. It is never included in the
feature matrix, and the feature schema rejects residual, correction, target, or
measured-mass columns.

## Input data

The default config is `configs/mass_estimation/residual_pipeline.yaml`. The main
input is an object-level CSV, JSON/JSONL, or Parquet file configured by
`data.measurements`.

Required canonical columns after `data.columns` mapping:

- `sample_id`: unique row/sample identifier.
- `class_name`: detector or material class.
- `real_mass_g`: measured scale mass in grams.

Recommended columns:

- `object_id`: physical object identifier for group-aware splitting.
- `estimated_volume_m3`: volume estimate from geometry/depth.
- `effective_density_kg_m3`: density prior or calibrated effective density.
- `material`: material family used for density priors.
- geometry/depth/reliability features such as mask area, aspect ratio, depth
  statistics, confidence, and valid depth ratio.

When `object_id` is absent, preprocessing uses `sample_id` as the group. When
density is absent, the configured class-to-material map and density priors are
used.

## Stages

```text
preprocess -> features -> split -> train -> evaluate -> register
```

`preprocess` validates object identifiers, classes, measured mass labels, and
available optional inputs. It writes:

```text
data/processed/mass_estimation/preprocessed_objects.csv
artifacts/mass_estimation/<run>/preprocessing/preprocessing_report.json
```

`features` builds a leakage-safe feature table and schema:

```text
data/processed/mass_estimation/features_all.csv
data/processed/mass_estimation/feature_schema.json
data/processed/mass_estimation/feature_summary.json
```

`split` performs train/validation/test splitting by `object_id` and reports row,
group, and class distributions:

```text
data/processed/mass_estimation/features_train.csv
data/processed/mass_estimation/features_val.csv
data/processed/mass_estimation/features_test.csv
data/processed/mass_estimation/split_summary.json
```

`train` fits each enabled candidate in `model.candidates`. The physics-only
baseline predicts zero residual; learned candidates predict residual correction.
The best model is selected by validation hybrid MAE by default:

```text
artifacts/mass_estimation/<run>/models/best_model.joblib
artifacts/mass_estimation/<run>/models/best_model_metadata.json
artifacts/mass_estimation/<run>/reports/training_summary.json
```

`evaluate` compares physics baseline, residual correction, and final hybrid
mass predictions on configured splits. It writes metrics, predictions,
per-class metrics, error analysis, and plots when Matplotlib is installed.

`register` packages the best residual model as an MLflow PyFunc model and
assigns the configured registry alias.

## Commands

Inspect readiness without writing artifacts:

```bash
edge-ai-mass mass run-pipeline \
  --config configs/mass_estimation/residual_pipeline.yaml \
  --dry-run
```

Run the full native pipeline without ZenML:

```bash
edge-ai-mass mass run-pipeline \
  --config configs/mass_estimation/residual_pipeline.yaml \
  --no-zenml
```

Run stages separately:

```bash
edge-ai-mass mass preprocess --config configs/mass_estimation/residual_pipeline.yaml
edge-ai-mass mass build-features --config configs/mass_estimation/residual_pipeline.yaml
edge-ai-mass mass train --config configs/mass_estimation/residual_pipeline.yaml
edge-ai-mass mass evaluate --config configs/mass_estimation/residual_pipeline.yaml
edge-ai-mass mass register --config configs/mass_estimation/residual_pipeline.yaml
```

Use overrides for notebook, Kaggle, or CI paths:

```bash
edge-ai-mass mass run-pipeline \
  --config configs/mass_estimation/residual_pipeline.yaml \
  --set data.measurements=/kaggle/input/mass-data/mass_measurements.csv \
  --set training.run_name=mass-residual-kaggle-v1 \
  --set orchestration.zenml.enabled=false
```

## Interpreting metrics

Evaluation reports three metric groups:

- `baseline`: performance of `mass_base_g` before learning.
- `correction`: performance of the predicted residual correction.
- `hybrid`: final mass performance after adding predicted correction to
  `mass_base_g`.

Primary thesis metrics are hybrid MAE/RMSE in grams, median absolute error,
R2 when available, safeguarded MAPE/sMAPE, and per-class MAE/RMSE. The baseline
comparison fields show whether the residual learner improved on the density
physics estimate.

## Durable output layout

```text
data/processed/mass_estimation/
  preprocessed_objects.csv
  features_all.csv
  features_train.csv
  features_val.csv
  features_test.csv
  feature_schema.json
  feature_summary.json
  split_summary.json

artifacts/mass_estimation/<run-name>/
  pipeline_state.json
  preprocessing/preprocessing_report.json
  features/{feature_schema,feature_summary,split_summary}.json
  training/<candidate>/{model.joblib,training_summary.json}
  models/{best_model.joblib,best_model_metadata.json}
  evaluation/
    metrics.json
    predictions.csv
    per_class_metrics.csv
    error_analysis.csv
    <split>/{metrics.json,predictions.csv,per_class_metrics.csv,plots/}
  reports/{training_summary,evaluation,model_config_resolved,registration}.json
  zenml/{run_context.json,stage_results/}
```

This structure keeps raw measurements, derived feature datasets, selected
models, and evaluation evidence easy to cite in the thesis.
