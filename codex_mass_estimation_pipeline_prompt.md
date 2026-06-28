# Codex Prompt — Build Mass-Estimation MLOps Pipeline

You are working inside an existing Edge AI waste-characterization project.  
The repository already contains a professional YOLO segmentation training pipeline with multiple stages, configuration files, artifact handling, experiment tracking, and MLOps conventions. Your task is to extend the existing project with a **mass-estimation pipeline** that follows the same architectural style instead of creating an isolated script.

## Goal

Build a complete, configurable, and thesis-ready pipeline for the **mass-estimation stage**:

```text
preprocessing → feature dataset creation → model training → evaluation → artifact logging → export/registry-ready outputs
```

The mass-estimation module must integrate naturally with the existing project structure, naming conventions, configuration style, MLflow tracking, ZenML orchestration, artifact layout, and CLI patterns already used by the YOLO training pipeline.

Do not redesign the whole project. Study the existing structure first, then extend it cleanly.

---

## Context

The project is an Edge AI system for waste characterization. The vision pipeline detects/segments waste objects and uses depth/geometric features to estimate approximate dimensions, volume, and mass.

The mass-estimation model should be implemented as a **hybrid residual regression model**:

```text
mass_base = estimated_volume × effective_density
correction = real_mass - mass_base
mass_pred = mass_base + predicted_correction
```

The model must predict the **correction/residual**, not the final mass directly as the only strategy.

The real measured mass must be treated as the target label only. It must never be used as an input feature.

---

## Important design principles

Follow these principles throughout the implementation:

1. **Respect the existing repository**
   - Inspect the current pipeline structure before writing code.
   - Reuse existing conventions for configs, logging, paths, stages, artifact storage, CLI commands, and experiment tracking.
   - Do not duplicate patterns that already exist.
   - Do not break the YOLO training pipeline.

2. **Configuration-first design**
   - The pipeline must be driven by YAML or the existing project configuration system.
   - Avoid hardcoded paths, class names, model types, thresholds, split ratios, or hyperparameters.
   - Allow switching between regression models through config.

3. **MLOps-grade traceability**
   - Log datasets, feature schemas, model parameters, metrics, plots, and final artifacts.
   - Use the existing MLflow and/or ZenML approach already present in the project.
   - Every produced artifact should be reproducible and easy to cite in the thesis.

4. **Small-data awareness**
   - The expected mass dataset is limited, approximately 70–120 measured samples per class.
   - Prefer robust tabular models and conservative model complexity.
   - Avoid large end-to-end RGBD neural networks unless implemented only as a future optional extension.

5. **No target leakage**
   - Do not include `real_mass`, `correction`, or any value derived from the real mass as an input feature.
   - Do not split data in a way that places the same physical object in both train and validation/test.
   - Prefer group-aware splitting by `object_id` if available.

---

## Required pipeline stages

Implement the mass-estimation pipeline with clear stages that mirror the existing YOLO pipeline style.

### 1. Preprocessing stage

Create a preprocessing stage that prepares object-level mass data.

It should support inputs such as:

- measured mass CSV or metadata file;
- object-level annotations or detections;
- segmentation masks;
- depth maps or depth-derived outputs;
- calibration metadata;
- density/material priors;
- class mapping.

The preprocessing stage should validate required columns and produce a clean object-level dataset.

Expected output:

```text
processed mass dataset
validated metadata
feature extraction readiness report
```

The implementation should be tolerant of missing optional data, but strict about required labels and identifiers.

---

### 2. Feature extraction stage

Create a feature-building stage that extracts or loads object-level features used for residual mass estimation.

The exact implementation should follow the existing project utilities where possible. Do not overfit to one file layout. Make the feature builder configurable.

Suggested feature groups:

- semantic features: class, material, optional superclass;
- physics features: estimated volume, effective density, mass_base;
- 2D geometry: mask area, bbox dimensions, projected area, aspect ratio;
- depth/geometric features: depth statistics, estimated thickness, volume proxies;
- shape features: solidity, extent, compactness;
- reliability features: mask confidence, depth valid ratio, detection confidence if available.

The feature builder should output a tabular artifact such as CSV or Parquet.

Expected output:

```text
features_train.parquet/csv
features_val.parquet/csv
features_test.parquet/csv
feature_schema.json
feature_summary.json
```

---

### 3. Splitting stage

Implement a configurable split stage.

Requirements:

- support train/validation/test split;
- support group split by `object_id` when available;
- preserve class distribution as much as possible;
- log split sizes and per-class distributions;
- make split seed configurable.

Do not silently perform unsafe random splits if repeated images of the same physical object exist.

---

### 4. Model training stage

Build a model-training stage for residual correction prediction.

The pipeline should support multiple candidate regressors through configuration. Use the existing dependency style of the project and avoid forcing unnecessary dependencies if they are not already used.

Recommended model candidates:

- physics-only baseline;
- Huber/Ridge-style linear baseline;
- Random Forest baseline;
- CatBoost or LightGBM if available or already compatible with the project;
- small MLP as a deployment-oriented optional model.

The main training target should be:

```text
target = real_mass_g - mass_base_g
```

The model should learn:

```text
predicted_correction_g = model(features)
final_mass_g = mass_base_g + predicted_correction_g
```

Use a clean model interface so that all regressors can be trained, evaluated, saved, and loaded consistently.

Expected output:

```text
trained model artifact
preprocessing/scaler/encoder artifact if needed
model_config_resolved.yaml/json
training_summary.json
```

---

### 5. Evaluation stage

Implement a professional evaluation stage.

Evaluate both:

```text
predicted_correction_g
final predicted mass_g
```

At minimum, report:

- MAE in grams;
- RMSE in grams;
- median absolute error;
- R²;
- MAPE or sMAPE, with safeguards for very small masses;
- per-class MAE;
- per-class RMSE;
- baseline comparison against `mass_base`.

The evaluation must clearly distinguish:

```text
physics baseline performance
direct or residual regression performance
final hybrid mass-estimation performance
```

Expected output:

```text
metrics.json
per_class_metrics.csv
predictions.csv
error_analysis.csv
plots/
```

Suggested plots:

- predicted vs real mass;
- residual error distribution;
- per-class MAE bar chart;
- baseline vs hybrid comparison;
- error versus true mass;
- feature importance if supported by the model.

Do not invent test metrics. If a required evaluation artifact is missing, fail clearly or mark the value as unavailable.

---

### 6. Experiment tracking and artifact logging

Integrate the pipeline with the existing MLOps stack.

Log to MLflow and/or ZenML according to the project’s current conventions:

- run config;
- dataset fingerprint or feature dataset hash;
- split summary;
- model type and hyperparameters;
- metrics;
- plots;
- model artifacts;
- feature schema;
- predictions;
- evaluation report.

The artifact structure should be easy to inspect and cite in thesis documents.

---

### 7. CLI integration

Add or extend CLI commands following the existing project style.

Example conceptual commands:

```text
mass preprocess
mass train
mass evaluate
mass run-pipeline
```

Use actual command names and argument style consistent with the repository.

The CLI should accept a config path and optional overrides if the existing project supports that pattern.

---

### 8. Tests and validation

Add practical tests for the new mass pipeline.

Focus on:

- feature schema validation;
- no target leakage;
- metric calculation correctness;
- split behavior;
- model save/load behavior;
- pipeline dry-run with a tiny synthetic dataset.

Keep tests lightweight and compatible with the existing test framework.

---

### 9. Documentation

Add concise documentation explaining:

- pipeline hierarchy;
- input data format;
- required columns;
- generated artifacts;
- how to run preprocessing;
- how to train;
- how to evaluate;
- how to interpret metrics;
- how the hybrid residual model works.

This documentation should be understandable for thesis explanation and future maintenance.

---

## Expected repository-level outcome

After implementation, the project should have a clean mass-estimation workflow similar in quality to the existing YOLO training workflow.

The final hierarchy should be easy to explain as:

```text
configs/
  mass_estimation/
    ...

src/
  ... existing project modules ...
  mass_estimation/
    data/
    features/
    models/
    training/
    evaluation/
    pipeline/

artifacts/
  mass_estimation/
    preprocessing/
    features/
    training/
    evaluation/
    models/
```

Adapt the exact locations to the existing repository structure. Do not force this exact layout if the project already uses a better convention.

---

## Acceptance criteria

The implementation is complete only when:

- the pipeline can be run end-to-end from config;
- preprocessing produces validated object-level feature data;
- at least one physics baseline and one learned residual model can be trained;
- evaluation compares baseline mass against hybrid predicted mass;
- metrics and plots are saved as artifacts;
- MLflow/ZenML tracking is integrated using the project’s existing style;
- the best model artifact is saved with enough metadata to reproduce it;
- the implementation avoids target leakage;
- documentation explains the hierarchy and commands;
- tests cover the core logic.

---

## Coding style expectations

Work like a senior MLOps engineer:

- modular code;
- typed functions where appropriate;
- clear interfaces;
- config-driven behavior;
- explicit errors;
- reproducible outputs;
- no hidden notebook-only logic;
- no hardcoded local paths;
- no metric overclaiming;
- no unnecessary complexity.

Prefer a clean, maintainable implementation over a clever but fragile one.

---

## Final instruction

Before coding, inspect the existing YOLO pipeline and summarize the structure you found. Then implement the mass-estimation pipeline by following the same patterns.

When uncertain, prefer consistency with the current project over introducing a new architecture.
