# YOLO training and MLOps pipeline

The training system is deliberately separate from the edge inference runtime. The stable entry points are the `edge-ai-mass train` command and `edge_ai_mass.training.TrainingPipeline`; scripts and notebooks call these entry points instead of implementing their own training logic.

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

## Data contract

Each configured source must provide:

- a COCO JSON file with `images`, `annotations`, and `categories`;
- an image root from which one of `file_name_fields` resolves;
- polygon segmentations for a segmentation task, or `allow_bbox_fallback: true` when rectangular masks are intentionally acceptable;
- a resolver or explicit `category_mapping` into `data.classes`.

The default config merges TACO, AquaTrash, and RealWaste into the repository's eight-class taxonomy. RealWaste uses `source_class_name` from the YOLO-to-SAM preparation notebook, retaining the original folder class even though its generated COCO category is one-class `trash`.

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
dvc add data/raw/RealWaste
dvc add data/normalized/realwaste
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

The final best-parameter training run enables Ultralytics plots. Its loss/metric history, precision-recall, F1, precision, recall, and confusion-matrix curves are copied into `training/curves/`; dataset-label diagnostics and annotated train/validation batches are retained beside them.

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

Open `notebooks/yolo_training_pipeline.ipynb`. It discovers a repository uploaded as a Kaggle dataset, adds its `src` folder to Python, binds mounted dataset paths through environment variables, and calls the same package API as the CLI.

Recommended Kaggle layout:

1. Upload this repository or its built wheel as a private module dataset.
2. Attach TACO, AquaTrash images, AquaTrash COCO labels, raw RealWaste, RealWaste COCO/SAM labels, and the starting checkpoint as datasets.
3. Set the path variables shown in the notebook.
4. Run `preprocess`, inspect the mosaics and manifest, then set `TRAIN_STAGES=tune,train,evaluate,export,register`.
5. Persist `/kaggle/working/artifacts` as a notebook output, or point MLflow to a remote server.

The notebook contains no duplicated preprocessing or training implementation, which keeps local, CI, and Kaggle behavior aligned.

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
  models/{best.pt,last.pt}
  training/{curves,diagnostics,annotated_batches}/
  evaluation/<split>/{curves,annotated_samples}/
  exports/<format>/
  exports/exports_manifest.json
  reports/{training_summary,evaluation,registration}.json
```

`pipeline_state.json` lets stages run in separate CLI jobs or notebook cells. A changed config cannot accidentally reuse trained state under the same run name unless `--force` is explicit.
