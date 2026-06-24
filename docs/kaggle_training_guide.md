# Running the YOLO training pipeline on Kaggle

This guide explains how to upload the project as an importable Kaggle dataset, attach the training data, start the governed pipeline, persist its outputs, and resume an Optuna study in another Kaggle session.

The recommended Kaggle setup separates code from data:

```text
Kaggle notebook
├── edge-ai-mass-module         # generated source/config archive
├── aquatrash                   # Images/ + annotations.csv
├── aquatrash-segmantations     # labels_final.json (existing slug has this typo)
├── realwaste                   # raw realwaste-main/RealWaste/<original-class>/...
└── segment-realwaste           # COCO annotations.json with project labels
```

TACO is not attached as a Kaggle dataset. The notebook downloads its COCO `annotations.json` from Hugging Face and installs the referenced images under `/kaggle/working/data/raw/taco`. Kaggle mounts the remaining attached datasets read-only under `/kaggle/input`; all downloaded and generated state stays under `/kaggle/working`.

## 1. Prepare the project as an importable dataset

There are two supported approaches. The source-tree approach is easiest while developing. A wheel is cleaner for repeatable released training jobs.

For this repository, the direct and recommended workflow is already automated:

```powershell
# Validate and build builds/kaggle/edge-ai-mass-module/edge_ai_mass_module.zip
python scripts/publish_kaggle_training.py prepare --force

# Version karimaouaouda/edge-ai-mass-module, wait until it is ready,
# then push and start karimaouaouda/train-pipeline.
python scripts/publish_kaggle_training.py publish `
  --message "Update governed YOLO training pipeline"
```

The equivalent dataset-only Bash automation is:

```bash
bash scripts/publish_kaggle_dataset.sh \
  --message "Update governed YOLO training pipeline"
```

The wrapper automatically selects `.venv/Scripts/python.exe` on Windows/Git Bash or `.venv/bin/python` on Linux, and accepts `PYTHON=/custom/python` when needed.

The helper validates every notebook code cell, excludes caches, weights, runs and invalid legacy scripts, creates a deterministic module archive with checksums, updates the existing private module dataset, and only then pushes the GPU kernel. Use `--skip-module` or `--skip-kernel` when publishing only one side.

### Option A: upload the generated module archive

`publish_kaggle_training.py prepare` creates a clean upload directory:

```text
builds/kaggle/edge-ai-mass-module/
├── dataset-metadata.json
├── edge_ai_mass-<version>-py3-none-any.whl
├── edge_ai_mass_module.zip
└── module-manifest.json
```

The wheel is the importable Python package and contains the package `__init__.py` files. The notebook installs it automatically with `pip --no-deps`. The archive retains `src/edge_ai_mass`, training configuration, documentation, and valid standalone scripts; the notebook extracts it into `/kaggle/working/edge_ai_mass_module` to access repository-level configuration. This is necessary because Kaggle dataset mounts are data paths and do not become importable packages simply because a ZIP contains `__init__.py`.

The `publish` command creates the dataset when absent or versions the existing `karimaouaouda/edge-ai-mass-module` dataset. To upload the prepared directory manually instead:

```powershell
kaggle datasets version `
  -p builds/kaggle/edge-ai-mass-module `
  -m "Update governed YOLO training pipeline"
```

### Option B: upload a wheel

Build the package locally:

```powershell
python -m pip install build
python -m build --wheel
```

Create a Kaggle dataset containing:

```text
edge-ai-mass-wheel/
├── edge_ai_mass-<version>-py3-none-any.whl
└── configs/
    └── training/
        └── yolo_segmentation.yaml
```

Keep the configuration beside the wheel because repository-level `configs/` files are not embedded in the Python package.

Install the wheel from the attached dataset without contacting PyPI:

```python
import subprocess
import sys
from pathlib import Path

wheel = next(Path("/kaggle/input/edge-ai-mass-wheel").glob("edge_ai_mass-*.whl"))
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", str(wheel)])

CONFIG_PATH = Path(
    "/kaggle/input/edge-ai-mass-wheel/configs/training/yolo_segmentation.yaml"
)
```

`--no-deps` is appropriate only when the Kaggle image already contains the required packages or you attached an offline wheelhouse. Otherwise enable internet temporarily and install the training dependencies:

```python
%pip install -q ultralytics mlflow optuna onnx onnxruntime pyyaml pillow
```

TensorRT `engine` export additionally requires a TensorRT build compatible with the Kaggle CUDA environment. ONNX export is usually the more portable Kaggle output; TensorRT can also be produced later on the target Jetson or in a matching NVIDIA container.

## 2. Import source code from the mounted dataset

The published dataset contains an installable wheel and `edge_ai_mass_module.zip`. Install the wheel, then extract the archive for configuration files:

```python
import os
import subprocess
import sys
import zipfile
from pathlib import Path

wheel = next(Path("/kaggle/input/edge-ai-mass-module").glob("edge_ai_mass-*.whl"))
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)]
)

archive = Path("/kaggle/input/edge-ai-mass-module/edge_ai_mass_module.zip")
MODULE_ROOT = Path("/kaggle/working/edge_ai_mass_module")
if not (MODULE_ROOT / "src" / "edge_ai_mass").is_dir():
    MODULE_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(MODULE_ROOT)
from edge_ai_mass.training import TrainingPipeline

CONFIG_PATH = MODULE_ROOT / "configs" / "training" / "yolo_segmentation.yaml"
os.chdir("/kaggle/working")
```

If Kaggle gives the mount a different name, inspect the attached roots:

```python
for path in sorted(Path("/kaggle/input").iterdir()):
    print(path)
```

The supplied notebook performs this extraction automatically. It also supports a directly mounted source tree containing `src/edge_ai_mass`. If discovery is ambiguous, set:

```python
os.environ["EDGE_AI_MASS_MODULE_ROOT"] = "/kaggle/working/edge_ai_mass_module"
```

Adding `src` to `sys.path` makes the project importable but does not install third-party dependencies. If they are not already present in the Kaggle image, install them with internet enabled or attach an offline wheelhouse:

```python
%pip install -q ultralytics mlflow optuna onnx onnxruntime pyyaml pillow
```

## 3. Download TACO, then bind the mounted datasets

Attach the AquaTrash inputs, raw RealWaste dataset, and RealWaste segmentation JSON through Kaggle. TACO is installed from the same Hugging Face JSON flow used by the earlier repository notebooks:

```python
import os
from pathlib import Path

from edge_ai_mass.training import TACO_ANNOTATIONS_URL, download_taco_dataset

taco_root = Path("/kaggle/working/data/raw/taco")
taco_summary = download_taco_dataset(
    taco_root,
    annotations_url=os.getenv("TACO_ANNOTATIONS_URL", TACO_ANNOTATIONS_URL),
    max_workers=int(os.getenv("TACO_DOWNLOAD_WORKERS", "8")),
    retries=int(os.getenv("TACO_DOWNLOAD_RETRIES", "3")),
    max_failures=int(os.getenv("TACO_MAX_DOWNLOAD_FAILURES", "0")),
)
os.environ["TACO_ANNOTATIONS"] = str(taco_root / "annotations.json")
os.environ["TACO_IMAGES"] = str(taco_root)

# AquaTrash raw images and its separate segmentation-label dataset.
os.environ["AQUATRASH_IMAGES"] = "/kaggle/input/aquatrash/Images"
os.environ["AQUATRASH_ANNOTATIONS"] = (
    "/kaggle/input/aquatrash-segmantations/labels_final.json"
)

# Raw images retain their original non-project class folders. They are used
# only for path resolution; labels always come from the COCO JSON categories.
os.environ["REALWASTE_IMAGES"] = (
    "/kaggle/input/realwaste/realwaste-main/RealWaste"
)
os.environ["REALWASTE_ANNOTATIONS"] = "/kaggle/input/segment-realwaste/annotations.json"
```

The downloader retrieves `https://huggingface.co/datasets/karimaouaouda/taco/resolve/main/annotations.json`, validates it as COCO JSON, validates downloaded images, writes files atomically, retries alternate URLs, and resumes existing valid images. Any missing image fails the job by default; set `TACO_MAX_DOWNLOAD_FAILURES` only when deliberately accepting a documented partial source. Internet must remain enabled in the Kaggle kernel.

The existing Kaggle source slug is `aquatrash-segmantations` with the typo. If a corrected dataset is attached, use its actual mounted path instead. Keep AquaTrash images and `labels_final.json` separate; they come from different mounted datasets.

The kernel metadata attaches `joebeachcapital/realwaste` for the raw `realwaste-main/RealWaste` tree and `karimaouaouda/segment-realwaste` for the segmentation JSON. The JSON must be COCO-shaped, contain polygon `segmentation` values, and use only the eight configured project labels in `categories[].name`. The notebook validates this contract and rejects a legacy one-class `trash` JSON. `REALWASTE_IMAGES` may point to the dataset mount, `realwaste-main`, or `RealWaste`; preprocessing normalizes it to the internal image root. When `images[].source_file_name` contains `realwaste-main/RealWaste/<class>/<image>`, preprocessing strips the redundant prefix and locates the raw file without interpreting `<class>` as a label.

For path failures, search Kaggle output for `[dataset-debug]`. `image_root.candidate` shows every checked root and sample directory entries; `image.resolve.after` shows exact attempted paths for a bounded sample; `source.progress` provides full-source counters; and `quality.failed` prints the resolved root plus representative unresolved JSON references. Increase `DATA_DEBUG_SAMPLE_LIMIT` only when the default 25 samples are insufficient.

## 4. Configure Kaggle working paths

Use overrides so every generated file goes to `/kaggle/working`:

```python
OVERRIDES = [
    "data.output_dir=/kaggle/working/data/processed/waste_seg_yolo",
    "data.materialize=copy",
    "training.artifacts_dir=/kaggle/working/artifacts/training/yolo",
    "tuning.storage=/kaggle/working/artifacts/optuna/yolo.db",
    "tracking.uri=sqlite:////kaggle/working/artifacts/mlflow/mlflow.db",
    "tracking.registry_uri=sqlite:////kaggle/working/artifacts/mlflow/mlflow.db",
    "training.device=0",
]

pipeline = TrainingPipeline.from_config(CONFIG_PATH, overrides=OVERRIDES)
```

For a remote MLflow server, set `MLFLOW_TRACKING_URI`, `MLFLOW_REGISTRY_URI`, and any authentication values from Kaggle Secrets rather than writing credentials into the notebook.

## 5. Validate before starting

Run a dry plan first. It checks source mounts, Python dependencies, selected stages, ONNX availability, and TensorRT availability when `engine` is enabled.

```python
import json

plan = pipeline.plan("all")
print(json.dumps(plan, indent=2))
assert plan["ready"], plan["blocking_issues"]
```

Do not start a long GPU run until `ready` is `true`.

## 6. Preprocess and inspect the dataset

Run preprocessing by itself first:

```python
preprocess_result = pipeline.run("preprocess")
print(json.dumps(preprocess_result, indent=2, default=str)[:20000])
```

The pipeline creates annotated samples and mosaics before tuning starts. Display them in the notebook:

```python
from IPython.display import display
from PIL import Image

visual_root = pipeline.config.artifacts_dir / "dataset_visualizations"
for mosaic in sorted(visual_root.glob("*/dataset_mosaic.jpg")):
    print(mosaic.parent.name)
    display(Image.open(mosaic))
```

Also inspect:

```text
/kaggle/working/data/processed/waste_seg_yolo/dataset_manifest.json
```

Verify class counts, missing-image counts, empty images, split sizes, and representative masks before spending GPU time.

## 7. Start tuning, training, evaluation, export, and registration

Run every remaining stage:

```python
results = pipeline.run("tune,train,evaluate,export,register")
print(json.dumps(results, indent=2, default=str)[:20000])
```

Export is disabled by default. Select one format:

```python
OVERRIDES += [
    "export.enabled=true",
    "export.formats=[onnx]",
]
```

Or multiple formats:

```python
OVERRIDES += [
    "export.enabled=true",
    "export.formats=[onnx, engine]",
]
```

Recreate the pipeline after changing `OVERRIDES`:

```python
pipeline = TrainingPipeline.from_config(CONFIG_PATH, overrides=OVERRIDES)
```

Only `models/best.pt` is exported. Outputs are organized under:

```text
/kaggle/working/artifacts/training/yolo/<run-name>/
├── dataset_visualizations/
├── optimization/
├── checkpoints/latest.json
├── checkpoints/epoch_NNNNNN/
│   ├── weights.pt
│   ├── best.pt
│   ├── results.csv
│   ├── metrics_history.json
│   └── metrics_curves.png
├── models/best.pt
├── training/curves/
├── evaluation/<split>/curves/
├── evaluation/<split>/annotated_samples/
├── exports/onnx/
├── exports/engine/
├── exports/exports_manifest.json
└── reports/
```

## 8. Start through the CLI instead of Python

When the wheel is installed, the console command is available:

```python
!edge-ai-mass train \
  --stage preprocess \
  --config /kaggle/input/edge-ai-mass-wheel/configs/training/yolo_segmentation.yaml \
  --set data.output_dir=/kaggle/working/data/processed/waste_seg_yolo \
  --set data.materialize=copy \
  --set training.artifacts_dir=/kaggle/working/artifacts/training/yolo
```

With a source-tree dataset that has not been installed, use the Python API. It is more reliable than expecting Kaggle to create a console entry point from a read-only mount.

## 9. Make other project code importable

Reusable logic should live under `src/edge_ai_mass`, not inside a notebook cell or a top-level script. For example:

```text
src/edge_ai_mass/training/custom_callback.py
```

It is then importable after mounting the module dataset:

```python
from edge_ai_mass.training.custom_callback import CustomCallback
```

Top-level `scripts/` files should remain thin CLI wrappers. If an old standalone script must be loaded without refactoring it into the package, use `importlib` explicitly:

```python
import importlib.util
from pathlib import Path

script_path = MODULE_ROOT / "scripts" / "example.py"
spec = importlib.util.spec_from_file_location("edge_ai_mass_example", script_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

# Call a function defined by the script.
module.main_function()
```

This fallback only works safely when the script does not execute training at import time. Moving reusable functions into `src/edge_ai_mass` is the preferred design.

## 10. Upload the notebook with Kaggle metadata

The repository `kernel-metadata.json` is already configured to update the existing `karimaouaouda/train-pipeline` kernel and run `notebooks/yolo_training_pipeline.ipynb` on a T4 GPU:

```json
{
  "id": "karimaouaouda/train-pipeline",
  "title": "Edge AI Mass YOLO Training Pipeline",
  "code_file": "notebooks/yolo_training_pipeline.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [
    "karimaouaouda/edge-ai-mass-module",
    "harshpanwar/aquatrash",
    "karimaouaouda/aqua_seg",
    "joebeachcapital/realwaste"
  ],
  "kernel_sources": [
    "karimaouaouda/segment-realwaste"
  ]
}
```

The notebook installs only missing dependencies including ZenML, extracts the generated module archive, downloads TACO from its Hugging Face JSON, binds the remaining input paths, writes every mutable artifact under `/kaggle/working`, and runs the full ordered ZenML pipeline by default. The Kaggle-specific defaults use 10 Optuna trials × 8 epochs followed by a 50-epoch final run; edit the notebook overrides for a larger production search.

To prepare and push both the module and kernel safely, use the helper:

```powershell
python scripts/publish_kaggle_training.py publish `
  --message "Update governed YOLO training pipeline"
```

For a kernel-only update after the module dataset is already current:

```powershell
python scripts/publish_kaggle_training.py publish --skip-module
```

## 11. Save outputs and resume later

Kaggle preserves `/kaggle/working` when a notebook version completes successfully. Save the notebook version with outputs, or create a new Kaggle dataset from the artifact directory.

Periodic final-training checkpoints are written during training. To run fixed
chunks, set `training.epochs` to the first chunk and
`training.checkpointing.resume.additional_epochs` to each resumed chunk size.
Keep the complete run artifact directory between sessions; the next invocation
with `resume.mode=auto` loads `checkpoints/latest.json`.

```python
OVERRIDES += [
    "training.epochs=20",
    "training.checkpointing.interval_epochs=5",
    "training.checkpointing.resume.mode=auto",
    "training.checkpointing.resume.additional_epochs=20",
]
```

The final ZenML `publish` step automatically creates or versions the private
Kaggle dataset selected by `TRAINING_OUTPUT_DATASET` (default:
`karimaouaouda/edge-ai-mass-training-outputs`). It uploads
`training_outputs.zip`, a checksum manifest, and Kaggle metadata. It never
uploads raw images, processed images/labels, or qualitative sample images.

When this output dataset is attached to a later notebook, the notebook
discovers `training_outputs.zip`, restores it into the run artifact directory,
sanitizes stale local paths and MLflow IDs, and enables automatic resume from
`checkpoints/latest.json`.

For a convenient downloadable archive:

```python
import shutil

shutil.make_archive(
    "/kaggle/working/edge_ai_mass_training_artifacts",
    "zip",
    "/kaggle/working/artifacts",
)
```

To resume Optuna in a later notebook session:

1. Save `/kaggle/working/artifacts` as a Kaggle dataset.
2. Attach that dataset to the new notebook.
3. Copy it back into the writable working directory before creating the pipeline.

```python
import shutil
from pathlib import Path

previous = Path("/kaggle/input/edge-ai-mass-previous-run/artifacts")
current = Path("/kaggle/working/artifacts")
if previous.exists() and not current.exists():
    shutil.copytree(previous, current)
```

Use the same dataset contents, training config, study name, and run name. The persistent Optuna database and pipeline state will resume compatible work. The study identity includes both the dataset fingerprint and optimization-config digest, preventing trials from different data or search spaces from being mixed.

## 12. Recommended execution order

For the first Kaggle run:

```text
1. Attach the module, AquaTrash data/labels, raw RealWaste, and its segmentation JSON.
2. Import the package from the module dataset.
3. Download TACO from its Hugging Face COCO JSON and bind environment paths.
4. Construct TrainingPipeline with `/kaggle/working` overrides.
5. Run plan("all") and resolve every blocking issue.
6. Run preprocess.
7. Inspect mosaics and dataset_manifest.json.
8. Run tune.
9. Run train,evaluate.
10. Inspect curves and annotated predictions.
11. Run export,register.
12. Save /kaggle/working artifacts as notebook output or a dataset.
```
