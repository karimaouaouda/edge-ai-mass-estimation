# Mass Model Candidate Report

This document explains the tabular models used by the current mass-estimation
pipeline and notebook, how they work internally, and which model is currently
selected as the best model.

The report is based on:

- `configs/mass_estimation/residual_pipeline.yaml`
- `mass_model/train_mass_estimation_pipeline.ipynb`
- `mass_model/artifacts/mass_estimation/mass-model-feature-residuals-notebook/reports/training_summary.json`
- `mass_model/artifacts/mass_estimation/mass-model-feature-residuals-notebook/evaluation/metrics.json`
- `mass_model/artifacts/mass_estimation/mass-model-feature-residuals-notebook/models/best_model_metadata.json`

## Pipeline Objective

The mass model is not trained to predict object mass directly from raw image
pixels. The vision pipeline first converts each segmented object into a tabular
feature row containing semantic, geometric, depth, volume, density, and quality
features. The mass model then learns a residual correction over a physics-based
baseline.

The hybrid formulation is:

```text
mass_base_g = selected_volume_cm3 * effective_density_g_cm3
residual_g = real_mass_g - mass_base_g
predicted_correction_g = model(features)
predicted_mass_g = mass_base_g + predicted_correction_g
```

This design is important for the thesis because it keeps the system physically
interpretable. Volume and density provide a first-principles baseline, while the
machine-learning model corrects systematic errors caused by imperfect masks,
relative depth, hollow objects, compression, irregular shapes, and material
priors.

## Feature Pipeline Used By The Models

The current feature schema uses six categorical feature groups:

| Feature | Meaning |
|---|---|
| `class_name` | Waste class detected by the vision pipeline. |
| `material` | Material group used for density priors. |
| `dimension_source` | Source of metric dimensions. |
| `thickness_source` | Source of object thickness. |
| `volume_source` | Source of selected volume estimate. |
| `baseline_method` | Method used to compute the physics baseline. |

The numeric features include density priors, pixel-to-centimeter calibration,
mask area, bounding-box dimensions, contour geometry, solidity, extent,
compactness, circularity, depth statistics, background/object depth relation,
estimated thickness, volume estimates, `selected_volume_cm3`, `mass_base_g`, and
`label_confidence`.

Columns that would leak the answer are explicitly excluded from the feature
matrix, including `real_mass_g`, `residual_g`, `correction_g`,
`predicted_correction_g`, `mass_pred_g`, baseline error columns, object/image
identifiers, split labels, and source paths.

## Preprocessing Under The Hood

All learned candidates are trained inside a shared sklearn-style pipeline:

```text
raw feature table
  -> numeric pipeline:
       median imputation
       standard scaling
  -> categorical pipeline:
       most-frequent imputation
       one-hot encoding with handle_unknown="ignore"
  -> regressor
```

This means every model receives a consistent numeric matrix even when runtime
or training rows contain missing values or new categorical levels. The
`handle_unknown="ignore"` behavior is especially useful for deployment because
an unseen categorical value does not crash inference.

Outliers are handled conservatively. The preprocessing stage uses class-wise
robust IQR/MAD fences over mass, base mass, residual correction, volume, area,
and thickness. The current policy is `flag_and_downweight`: suspicious samples
are retained for auditability but their training weight is reduced instead of
being silently deleted. This is suitable for a graduation project because it
keeps traceability while reducing the influence of extreme or noisy rows.

## Training Mode

The current notebook and configuration use:

```text
training.mode = compare_candidates
training.selection.split = val
training.selection.metric = hybrid.mae_g
training.selection.mode = minimize
```

So the pipeline trains every enabled candidate, computes validation metrics, and
selects the model with the lowest validation hybrid MAE in grams. The final
saved model is wrapped as a `ResidualMassModel`, which stores the estimator,
feature schema, preprocessing pipeline, and hybrid reconstruction rule.

## Models Used

### Physics Baseline

The physics baseline is the non-learning reference model. It predicts no
residual correction:

```text
predicted_correction_g = 0
predicted_mass_g = mass_base_g
```

Under the hood, it is equivalent to trusting the volume-density formula exactly.
It is useful as a scientific control because every learned model must show that
it improves over this physically interpretable baseline.

### Ridge Residual Regression

Ridge regression is a linear model with L2 regularization. It learns one
coefficient per transformed feature and penalizes large coefficient magnitudes.
Under the hood, it solves a regularized least-squares objective:

```text
minimize squared_error + alpha * sum(coefficients^2)
```

In this project, Ridge is useful as a simple, stable baseline. It can model
linear relationships between features and residual correction, but it cannot
naturally model strong nonlinear effects such as interactions between depth,
volume, class, and hollow-object priors.

### Huber Residual Regression

Huber regression is also a linear model, but it uses a robust loss. Small errors
are treated like squared error, while large errors are treated closer to
absolute error. Under the hood, this reduces sensitivity to mislabeled or
extreme samples.

This candidate is useful because mass labels and pseudo-depth-derived features
can contain noise. However, because it remains linear, it still has limited
capacity for complex object-shape and material interactions.

### Random Forest Residual Regression

Random Forest is an ensemble of decision trees. Under the hood, many trees are
trained on randomized samples and randomized feature subsets, and their
predictions are averaged.

Each tree partitions the feature space into regions such as:

```text
if material == plastic and selected_volume_cm3 < threshold:
    predict one residual range
else:
    continue splitting
```

The averaging step reduces variance compared with a single decision tree. This
model is well suited to tabular mass features because it can learn nonlinear
rules and feature interactions without requiring manual polynomial terms.

### Extra Trees Residual Regression

Extra Trees is similar to Random Forest, but it injects more randomness into
tree construction. Instead of searching only for the most optimal split
threshold at each node, it samples candidate thresholds more randomly and then
chooses among them.

This extra randomization usually lowers variance and can generalize well on
small or noisy tabular datasets. In this project, that behavior is valuable
because the dataset combines measured object mass, pseudo-depth features,
class/material priors, and geometric proxies. Extra Trees can capture nonlinear
relationships while being less prone to overfitting precise thresholds from a
limited dataset.

### Gradient Boosting Residual Regression

Gradient Boosting builds trees sequentially. Under the hood, each new tree tries
to correct the errors left by the previous ensemble. The final prediction is the
sum of many small tree contributions:

```text
prediction = tree_1 + learning_rate * tree_2 + ... + learning_rate * tree_n
```

This model is powerful for structured data because it focuses on difficult
residual patterns step by step. The learning rate, tree depth, and subsampling
control how aggressively the model fits the training data.

### Histogram Gradient Boosting Residual Regression

Histogram Gradient Boosting is a faster gradient-boosting variant. It bins
continuous features into histograms before finding splits. Under the hood, this
reduces computation and can add regularization because the model does not chase
every exact numeric threshold.

It is a strong candidate for larger tabular datasets, but in the current run it
did not beat Extra Trees on the validation selection metric.

### XGBoost Residual Regression

XGBoost is an optimized gradient-boosted tree engine. Under the hood, it builds
trees sequentially using gradient information, regularization, shrinkage,
subsampling, and column sampling. The current configuration uses the histogram
tree method with squared-error regression and MAE evaluation.

XGBoost is often strong on tabular data because it includes careful
regularization and efficient tree construction. In the current notebook run it
was enabled and trained, but it did not outperform Extra Trees on validation
hybrid MAE.

### LightGBM Residual Regression

LightGBM is another optimized gradient-boosted tree engine. It uses
histogram-based splits and leaf-wise tree growth controlled by settings such as
`num_leaves`, `min_child_samples`, sampling, and regularization.

LightGBM can be very efficient and accurate for tabular data, especially when
the dataset grows. In the current notebook run it was enabled and trained, but
it also did not outperform Extra Trees on the validation selection metric.

### MLP Residual Regression

The configuration includes an MLP candidate, but it is currently disabled. If
enabled, it would train a small feed-forward neural network over the encoded
tabular features. Under the hood, the network would learn nonlinear feature
combinations through hidden layers.

For the current dataset size and thesis need for interpretability, tree-based
models are a better first choice than an MLP. The MLP remains an experimental
candidate rather than a selected production model.

## Validation Candidate Ranking

The current notebook run trained the enabled candidates and selected the best
model using validation `hybrid.mae_g`, where lower is better.

| Rank | Candidate | Type | Validation MAE (g) | Validation RMSE (g) | Validation R2 |
|---:|---|---|---:|---:|---:|
| 1 | `extra_trees_residual` | Extra Trees | 32.9475 | 59.2560 | 0.7951 |
| 2 | `gradient_boosting_residual` | Gradient Boosting | 35.3582 | 64.6817 | 0.7558 |
| 3 | `lightgbm_residual` | LightGBM | 35.6129 | 69.2982 | 0.7197 |
| 4 | `random_forest_residual` | Random Forest | 38.0629 | 80.1144 | 0.6254 |
| 5 | `xgboost_residual` | XGBoost | 38.2540 | 73.3526 | 0.6859 |
| 6 | `hist_gradient_boosting_residual` | Histogram Gradient Boosting | 41.5768 | 90.9853 | 0.5168 |
| 7 | `huber_residual` | Huber Regression | 44.9472 | 89.6556 | 0.5308 |
| 8 | `ridge_residual` | Ridge Regression | 52.3082 | 98.0768 | 0.4386 |
| 9 | `physics_baseline` | Physics only | 63.9722 | 133.1716 | -0.0351 |

## Selected Best Model

The best model in the current mass pipeline and notebook is:

```text
extra_trees_residual
```

It is selected because it has the lowest validation hybrid MAE:

```text
validation hybrid MAE = 32.9475 g
validation hybrid RMSE = 59.2560 g
validation hybrid R2 = 0.7951
```

The saved artifact is:

```text
mass_model/artifacts/mass_estimation/mass-model-feature-residuals-notebook/models/best_model.joblib
```

The metadata confirms:

```text
selected_model = extra_trees_residual
selected_type = extra_trees
training_mode = compare_candidates
selection_metric = validation hybrid.mae_g
```

## Final Evaluation Of The Selected Model

The evaluator compares the selected hybrid model against the physics baseline.

### Validation Split

| Method | MAE (g) | RMSE (g) | Median AE (g) | R2 | sMAPE (%) |
|---|---:|---:|---:|---:|---:|
| Physics baseline | 63.9722 | 133.1716 | 18.4027 | -0.0351 | 89.5222 |
| Selected hybrid model | 32.9475 | 59.2560 | 11.7072 | 0.7951 | 59.7788 |

Validation improvement over baseline:

```text
MAE improvement = 31.0247 g
relative MAE improvement = 48.4972%
RMSE improvement = 73.9156 g
relative RMSE improvement = 55.5040%
```

### Test Split

| Method | MAE (g) | RMSE (g) | Median AE (g) | R2 | sMAPE (%) |
|---|---:|---:|---:|---:|---:|
| Physics baseline | 44.3073 | 88.6857 | 15.6717 | -0.0281 | 85.0404 |
| Selected hybrid model | 23.1773 | 50.2325 | 9.1752 | 0.6702 | 49.0401 |

Test improvement over baseline:

```text
MAE improvement = 21.1300 g
relative MAE improvement = 47.6897%
RMSE improvement = 38.4533 g
relative RMSE improvement = 43.3590%
```

## Interpretation

Extra Trees is currently the strongest model because it gives the best
validation error while preserving a practical balance between nonlinear
learning, robustness, and deployment simplicity. It can model interactions
between object class, material density, projected area, depth statistics,
estimated thickness, and base mass without requiring hand-designed interaction
terms.

The result also shows that the hybrid approach is justified. The physics
baseline alone has negative R2 on both validation and test splits, which means
the volume-density estimate is not sufficient by itself. Adding the learned
residual correction reduces test MAE from 44.3073 g to 23.1773 g.

XGBoost and LightGBM are useful candidates and should remain in the experiment
set, especially as more measured training samples are collected. However, for
the current dataset and current notebook run, neither one beats Extra Trees on
the configured validation selection metric.

## Thesis-Ready Conclusion

The current mass-estimation pipeline uses a hybrid physics-plus-ML residual
regression strategy. A physics baseline first estimates mass from object volume
and effective density. Several tabular regressors are then trained to predict
the residual correction between measured mass and the baseline estimate. The
candidate set includes linear, robust linear, bagged tree, extremely randomized
tree, gradient-boosted tree, XGBoost, and LightGBM models. The selected model is
`extra_trees_residual`, chosen by lowest validation hybrid MAE. On the held-out
test split, the selected hybrid model reduces MAE from 44.3073 g for the physics
baseline to 23.1773 g, corresponding to a 47.6897% MAE improvement. This makes
Extra Trees the current best model for the mass-estimation stage.
