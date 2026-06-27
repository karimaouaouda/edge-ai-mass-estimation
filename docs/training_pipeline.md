# YOLO training and MLOps pipeline

The training system is deliberately separate from the edge inference runtime. The stable entry points are the `edge-ai-mass train` command and `edge_ai_mass.training.TrainingPipeline`; scripts and notebooks call these entry points instead of implementing their own training logic. ZenML supplies the ordered execution DAG and run lineage, while the existing package methods remain the single implementation of every stage.

## ZenML orchestration

ZenML is enabled by default and records this ordered DAG:

```text
initialize -> preprocess -> tune -> train -> evaluate -> export -> register -> publish
```

Stages not selected by `--stage` remain visible as explicit skips. Caching is
disabled because DVC, Optuna, durable state, and managed checkpoints already
provide domain-specific reuse and resume behavior.

```bash
edge-ai-mass train \
  --stage preprocess,train,evaluate,publish \
  --config configs/training/yolo_segmentation.yaml
```

Use `--no-zenml` only as a diagnostic fallback to execute the same native stage
methods without creating a ZenML run.

## Architecture

```text
COCO sources                 Data governance             Model governance
TACO -----------+          +------------------+        +-------------------+
AquaTrash ------+-- COCO ->| validation/split |--YOLO->| Optuna study      |
RealWaste ------+          | manifest + DVC   |        | MLflow experiment |
                           +------------------+        | eval gates        |
                                                       | model registry    |
                                                       +-------------------+
```

The current implementation targets the `detection` model stage with Ultralytics YOLO and supports both `segment` and `detect` tasks. The orchestration boundary is model-stage agnostic so a mass-regression trainer can be added later without mixing it into the edge runtime.

## Execution flow

The training system is not a loose collection of scripts; it is one controlled pipeline with one shared config object. The code path is:

1. `edge-ai-mass train` parses the YAML config, applies any `--set` overrides, and builds a `TrainingConfig`.
2. `TrainingPipeline.plan()` validates the requested stages, checks required Python packages, inspects data-source availability, and resolves the model/checkpoint source without mutating anything.
3. `TrainingPipeline.run()` chooses ZenML when it is enabled, otherwise it falls back to the native runner.
4. `TrainingPipeline.run_native()` executes the selected stages in order and writes stage results into `pipeline_state.json` under the run's artifact directory.
5. `preprocess` materializes the merged YOLO dataset and dataset visualizations, then records the dataset fingerprint so later runs can detect drift.
6. `tune` launches Optuna, `train` resumes from the managed checkpoint state when available, `evaluate` reloads `models/best.pt`, `export` converts that best model into deployable formats, `register` pushes the best artifact to MLflow, and `publish` packages the final outputs for external storage.

The important consequence is that every stage reads the same resolved configuration and the same run-state record. That is what makes the pipeline resumable, reproducible, and safe to split across separate jobs or notebook cells.

## Install

```bash
pip install -e ".[mlops]"
```

The `mlops` extra installs MLflow, Optuna, DVC, ONNX tooling, and the report dependencies. Core inference users do not need to load these packages when running the edge agent.

## Commands

Inspect configuration, mounts, package readiness, and the exact execution plan without writing files:

```bash
edge-ai-mass train --config configs/training/yolo_segmentation.yaml --dry-run
```

Run the entire detection pipeline:

```bash
edge-ai-mass train --stage detection --config configs/training/yolo_segmentation.yaml
```

Run or resume individual stages:

```bash
edge-ai-mass train --stage preprocess --config configs/training/yolo_segmentation.yaml
edge-ai-mass train --stage tune --config configs/training/yolo_segmentation.yaml
edge-ai-mass train --stage train,evaluate,export,register --config configs/training/yolo_segmentation.yaml
edge-ai-mass train --stage publish --config configs/training/yolo_segmentation.yaml
```

Use any Ultralytics checkpoint and apply config overrides without cloning a YAML file:

```bash
edge-ai-mass train --stage detection \
  --config configs/training/yolo_segmentation.yaml \
  --set model.checkpoint=models/weights/custom-seg.pt \
  --set training.epochs=150 \
  --set training.device=0
```

`--skip-optuna` removes the tune stage from a full run. Preprocessing always rebuilds atomically when that stage is invoked; DVC decides when reproduction is needed. `--force` permits a changed config to replace model state under the same run name and allows registration to run again. Using a new `training.run_name` is preferred when the experiment is conceptually different.

### Periodic checkpoints and chunked training

The existing `train` stage stores a resumable checkpoint every
`training.checkpointing.interval_epochs`; no extra pipeline stage is required.

```yaml
training:
  epochs: 20
  checkpointing:
    enabled: true
    interval_epochs: 5
    save_final: true
    keep_last: 10
    resume:
      mode: auto
      checkpoint: null
      additional_epochs: 20
```

With this example, the first invocation trains 20 epochs. Every later
invocation loads `checkpoints/latest.json`, restores model, optimizer, scaler,
scheduler, and epoch state, then trains 20 additional epochs. Set
`additional_epochs: 0` when `training.epochs` is a fixed total target.

Resume modes:

- `auto`: use the latest managed checkpoint when present, otherwise load
  `model.checkpoint`.
- `never`: always start from `model.checkpoint`.
- `required`: fail unless a managed or explicit resume checkpoint exists.

Select a mounted or local checkpoint explicitly:

```bash
edge-ai-mass train \
  --stage train \
  --config configs/training/yolo_segmentation.yaml \
  --set training.checkpointing.resume.mode=required \
  --set training.checkpointing.resume.checkpoint=/path/to/weights.pt \
  --set training.checkpointing.resume.additional_epochs=20
```

The same professional resolver is available to scripts and notebooks:

```python
from edge_ai_mass.training import load_model_or_checkpoint

model, source = load_model_or_checkpoint(pipeline.config)
print(source.kind, source.path, source.resume)

# Ignore stored checkpoints and load model.checkpoint.
base_model, base_source = load_model_or_checkpoint(
    pipeline.config,
    prefer_resume=False,
)
```

## Data contract

Each configured source must provide:

- a COCO JSON file with `images`, `annotations`, and `categories`;
- an image root from which one of `file_name_fields` resolves;
- polygon segmentations for a segmentation task, or `allow_bbox_fallback: true` when rectangular masks are intentionally acceptable;
- a resolver or explicit `category_mapping` into `data.classes`.

The default config merges TACO, AquaTrash, and RealWaste into the repository's eight-class taxonomy. RealWaste is supplied as two independent inputs: the untouched `realwaste-main/RealWaste/<original-class>/...` image tree and a COCO segmentation JSON whose category names are already project labels. Original RealWaste folder classes are never used as training labels. `source_file_name` is preferred for locating nested raw images, while `category_id -> categories[].name` remains the label authority. `REALWASTE_IMAGES` may point at the Kaggle dataset mount, `realwaste-main`, or the final `RealWaste` directory; preprocessing records both configured and resolved roots in the source report.

Dataset diagnostics are enabled by default and printed as one-line JSON records prefixed with `[dataset-debug]`. They cover build/source boundaries, path-root candidates, COCO counts/categories, bounded image-resolution traces, periodic progress, and every quality-gate input. Control verbosity with `DATA_DEBUG_ENABLED`, `DATA_DEBUG_SAMPLE_LIMIT`, `DATA_DEBUG_PROGRESS_EVERY`, and `DATA_DEBUG_ROOT_ENTRY_LIMIT`.

Preprocessing provides:

- image and annotation validation with configurable quality gates;
- clipping and degenerate-polygon rejection;
- deterministic, source-aware train/validation/test splits;
- SHA-256 image fingerprints so exact duplicates cannot cross splits;
- collision-safe output names and portable YOLO YAML;
- hard-link materialization locally, with copy fallback across filesystems;
- `dataset_manifest.json` containing source annotation hashes, aggregate image hashes, class distributions, split counts, and the final dataset fingerprint.
- annotated samples and split contact sheets under `dataset_visualizations/` before tuning or training starts.

COCO RLE masks are rejected rather than silently converted incorrectly. Convert them to polygons upstream if a source uses RLE. For multipart objects, the default `largest` policy keeps the largest valid component and reports how many smaller components were reduced.

## Data versioning with DVC

`dvc.yaml` defines the merged YOLO dataset as a reproducible stage. First track raw sources that are not already under DVC:

```bash
dvc add data/raw/aquatrash
dvc add data/raw/realwaste-main/RealWaste
dvc add data/annotations/realwaste/annotations.json
git add data/raw/*.dvc data/normalized/*.dvc .gitignore
```

Then reproduce and commit lineage metadata:

```bash
dvc repro yolo_dataset
git add dvc.yaml dvc.lock configs/training/yolo_segmentation.yaml
git commit -m "version YOLO training dataset"
```

Configure an object-store remote once for team or CI use:

```bash
dvc remote add -d training-data <remote-url>
dvc push
```

DVC owns data versions and reproducibility. MLflow owns runs, metrics, model packages, and registry state; this avoids two tools claiming the same responsibility.

## Optuna tuning

The search space is fully config-driven. It covers the meaningful model-selection surface: image size, batch size, optimizer, early stopping, learning-rate and warmup schedules, loss gains, nominal batch scaling, mask behavior, backbone freezing, mosaic scheduling, multi-scale training, and color/geometric/composition augmentation. `tuning.enforce_complete_space` validates that every configured model-quality hyperparameter, augmentation, and extra training argument has an Optuna distribution, preventing a newly added parameter from silently remaining untuned. Operational parameters such as device, workers, cache, AMP, deterministic execution, epochs-per-trial, and seed remain fixed because they define resources/reproducibility rather than a fair model-quality search.

The default study uses seeded multivariate TPE, a median pruner, a persistent SQLite backend, and mask mAP50-95 as the objective. Its name is suffixed with both the dataset fingerprint and optimization-config digest, so reruns resume compatible trials without mixing trials from changed data or training logic. Because the space is high-dimensional, use at least 50 trials and preferably 100 or more for final model selection. For concurrent workers, point `tuning.storage` at a shared PostgreSQL URL and launch the tune stage from each worker with the same study name and config.

Each trial is a nested MLflow run. Failed CUDA/resource combinations are marked as failed while the study continues. The best parameters and study metadata are written to `optimization/optuna_best.json`, recorded in pipeline state, and applied to the final training run. Optimization history, parameter importance, slices, and parallel-coordinate visualizations are saved as HTML when Plotly is available.

## Curves and qualitative evaluation

The final best-parameter training run enables Ultralytics plots. Its loss/metric history, precision-recall, F1, precision, recall, and confusion-matrix curves are copied into `training/curves/`; dataset-label diagnostics and annotated train/validation batches are retained beside them. Every managed checkpoint atomically stores resumable weights, current best weights, full CSV and JSON metric history, a generated all-metric curve panel, checksums, and a manifest.

Evaluation reloads `models/best.pt`, never `last.pt`. For each configured split it saves quantitative metrics, PR/F1/confusion curves, COCO JSON when supported, and deterministic annotated predictions sampled from the actual split. Everything is logged to MLflow and referenced from `reports/evaluation.json`.

## Best-model export

Export is an explicit stage and always uses `models/best.pt`. Enable it and choose one or multiple formats in YAML, or override them from the CLI:

```bash
edge-ai-mass train --stage export \
  --config configs/training/yolo_segmentation.yaml \
  --set export.enabled=true \
  --set 'export.formats=[onnx, engine]'
```

Per-format options live under `export.options.formats`. ONNX supports settings such as opset, dynamic axes, simplification, half precision, and NMS. TensorRT `engine` supports half/INT8 precision, workspace, dynamic shapes, simplification, and NMS. INT8 automatically receives the versioned dataset YAML for calibration. TensorRT export requires a supported NVIDIA environment with TensorRT installed.

Exports are moved into `exports/<format>/`; they are checksummed and indexed by `exports/exports_manifest.json`. `continue_on_error: false` makes deployment conversion fail-fast, while `true` records a failed format and continues with the remaining selections.

## MLflow tracking and registry

Local development defaults to a SQLite MLflow backend at `artifacts/mlflow/mlflow.db`, avoiding the deprecated filesystem tracking backend. CI or a training server should set:

```bash
MLFLOW_TRACKING_URI=https://mlflow.example.com
MLFLOW_REGISTRY_URI=https://mlflow.example.com
MLFLOW_EXPERIMENT_NAME=edge-ai-mass-yolo-segmentation
```

The pipeline logs the resolved config, Git commit/dirty state, dataset manifest and fingerprint, dataset mosaics, Optuna parameters and plots, final checkpoints, training curves, annotated evaluation samples, exported deployment models, reports, and metrics. Registration packages the best YOLO checkpoint as an MLflow PyFunc model and assigns the configured `candidate` alias.

## Reusable-output dataset publication

The final `publish` step creates
`karimaouaouda/edge-ai-mass-training-outputs` when absent and creates a new
Kaggle dataset version on every later successful run.

It includes checkpoints, best/last models, deployment exports, Optuna results,
metric histories and chart curves, reports, resolved configuration, pipeline
state, Git metadata, and the dataset manifest. It excludes raw data, processed
YOLO images/labels, dataset visualizations, annotated samples/batches,
Ultralytics scratch runs, caches, and the MLflow database.

```yaml
publication:
  enabled: true
  provider: kaggle
  dataset: karimaouaouda/edge-ai-mass-training-outputs
  public: false
  require_checkpoint: true
```

Restore the output for future training:

```python
from edge_ai_mass.training import restore_training_outputs

restore_training_outputs(
    "/kaggle/input/edge-ai-mass-training-outputs/training_outputs.zip",
    "/kaggle/working/artifacts/training/yolo/waste-seg-yolo",
)
```

The helper validates archive paths, excludes metadata from extraction, removes
stale MLflow IDs, rewrites checkpoint/model paths, and makes
`resume.mode=auto` immediately usable.

Promotion should remain a separate approval action after edge benchmarking:

```python
from mlflow import MlflowClient

client = MlflowClient()
client.set_registered_model_alias(
    "edge-ai-mass-yolo-segmentation",
    "production",
    version="<approved-version>",
)
```

The edge release process should consume the approved `production` version, export it to ONNX/TensorRT, and publish it through the existing device updater rather than letting training write directly into deployed device weights.

## Jupyter and Kaggle

Open `notebooks/yolo_training_pipeline.ipynb` and clone the repository into the notebook session first. The notebook should treat GitHub as the source of truth for the code, not a Kaggle dataset upload.

Typical notebook startup flow:

1. Clone the repository into `/kaggle/working` or the local notebook workspace.
2. Install the project in editable mode when the environment permits it, or add the cloned `src` directory to `sys.path`.
3. Attach only the training data as Kaggle datasets or local mounts: AquaTrash, RealWaste, and any other required image/annotation sources.
4. Download TACO from its Hugging Face COCO JSON through the same helper used by the repository code.
5. Build `TrainingPipeline.from_config(...)` from the checked-out repo config, apply overrides for `/kaggle/working`, then run `plan("all")` and `run("preprocess")` before starting the full training stages.
6. Persist `/kaggle/working/artifacts` as notebook output, or point MLflow at a remote tracking server if you want the run history outside the notebook.

See `docs/kaggle_training_guide.md` for the notebook-oriented setup details, but treat the repository checkout itself as code cloned from GitHub rather than as a data artifact.

The notebook contains no duplicated preprocessing or training implementation, which keeps local, CI, and notebook behavior aligned.

## Durable outputs

```text
data/processed/waste_seg_yolo/
  dataset.yaml
  dataset_manifest.json
  images/{train,val,test}/<source>/
  labels/{train,val,test}/<source>/

artifacts/training/yolo/<run-name>/
  pipeline_state.json
  dataset_visualizations/{train,val,test}/
  optimization/
    optuna_best.json
    visualizations/*.html
  checkpoints/
    latest.json
    epoch_000005/{weights.pt,best.pt,results.csv,metrics.json,metrics_history.json,metrics_curves.png,checkpoint_manifest.json}
    epoch_000010/{...}
  models/{best.pt,last.pt}
  training/{curves,diagnostics,annotated_batches}/
  evaluation/<split>/{curves,annotated_samples}/
  exports/<format>/
  exports/exports_manifest.json
  reports/{training_summary,evaluation,registration}.json
  zenml/{run_context.json,stage_results/}
```

`pipeline_state.json` lets stages run in separate CLI jobs or notebook cells. A changed config cannot accidentally reuse trained state under the same run name unless `--force` is explicit.
