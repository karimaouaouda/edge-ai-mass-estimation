# Edge AI Mass Estimation

Real-time waste object characterization and mass estimation on edge devices (NVIDIA Jetson Nano + RGB camera).

## Architecture

```
Frame → [Detection] → [Depth Estimation] → [Mass Estimation] → Results
           │                │                     │
        YOLOv8-seg    Depth Anything V2     Regression MLP
           ↓                ↓                     ↓
        (fallback)      (fallback)           (fallback)
      MobileNetV3-SSD   MiDaS small       Density × Volume
```

Each stage has a **primary** model and an automatic **fallback** that triggers if the primary fails or exceeds its latency budget.

## Features

- **Cascade pipeline** with per-stage latency budgets and automatic fallback
- **Object detection + segmentation** via YOLOv8-nano-seg
- **Monocular depth estimation** via Depth Anything V2
- **Mass estimation** — hybrid: learned regression + physics-based density×volume
- **Camera calibration** for metric-scale depth
- **TensorRT export** for optimised Jetson inference
- **Config-driven** — swap models and tune parameters via YAML, no code changes
- **CLI** for inference, live demo, benchmarking, and calibration
- **Jetson orchestrator** for GitHub Release and MQTT-triggered model/app updates

## Quick Start

```bash
# Install (development mode with all extras)
pip install -e ".[all]"

# Run inference on an image
edge-ai-mass infer --image photo.jpg --config configs/pipeline/default.yaml

# Live camera demo
edge-ai-mass demo --config configs/pipeline/default.yaml --camera 0

# Benchmark latency
edge-ai-mass benchmark --config configs/pipeline/default.yaml

# Calibrate camera
edge-ai-mass calibrate --images data/calibration/ --output configs/calibration.json
```

## Training

```bash
# Inspect the governed YOLO pipeline and dataset mounts
edge-ai-mass train --config configs/training/yolo_segmentation.yaml --dry-run

# Preprocess, visualize, tune, train, evaluate, optionally export, and register
edge-ai-mass train --stage detection --config configs/training/yolo_segmentation.yaml

# Resume selected stages or load any Ultralytics checkpoint
edge-ai-mass train --stage train,evaluate,export,register \
    --config configs/training/yolo_segmentation.yaml \
    --set model.checkpoint=models/weights/custom-seg.pt

# Export only the selected best checkpoint to ONNX and TensorRT
edge-ai-mass train --stage export \
    --config configs/training/yolo_segmentation.yaml \
    --set export.enabled=true \
    --set 'export.formats=[onnx, engine]'

# Collect features for mass regression
python scripts/collect_mass_features.py \
    --images data/raw/mass_dataset/ \
    --labels data/raw/mass_labels.csv \
    --output data/splits/mass_features.csv

# Train mass regression head
python scripts/train_mass_regression.py --config configs/models/mass_regression_training.yaml
```

The YOLO pipeline uses DVC for data lineage, Optuna for persistent hyperparameter studies, and MLflow for experiment tracking and model registration. See `docs/training_pipeline.md`, `docs/kaggle_training_guide.md`, and `notebooks/yolo_training_pipeline.ipynb`.

ZenML orders and records the complete DAG, including a final `publish` step
that creates or versions a private Kaggle dataset containing reusable
checkpoints, models, metrics, configuration, and reports—but no raw or
processed datasets.

Prepare and directly publish the importable module dataset plus GPU training kernel:

```bash
python scripts/publish_kaggle_training.py prepare --force
python scripts/publish_kaggle_training.py publish
```

From Bash, publish only a new module-dataset version and wait until it is ready:

```bash
bash scripts/publish_kaggle_dataset.sh \
  --message "Publish updated training module"
```

Final YOLO training saves periodic resumable snapshots and automatically loads
the newest one on the next invocation. Configure the interval and resumed chunk
size with `training.checkpointing.interval_epochs` and
`training.checkpointing.resume.additional_epochs`.

## Jetson Deployment

```bash
# Export to TensorRT
python scripts/export_tensorrt.py --model models/weights/yolov8n-seg.pt --format engine --half

# Build Docker image
docker build -f deploy/Dockerfile.jetson -t edge-ai-mass:jetson .

# Run
docker run --runtime nvidia -it edge-ai-mass:jetson demo --camera 0

# Run the update orchestrator process
edge-ai-orchestrator --config configs/orchestration/jetson_nano.yaml
```

See `docs/orchestrator.md` for GitHub Release manifests, Mosquitto commands, rollback, and systemd deployment.

## Project Structure

```
├── src/edge_ai_mass/          # Python package
│   ├── modules/               # AI model wrappers
│   │   ├── detection/         # YOLOv8, MobileNet
│   │   ├── depth/             # Depth Anything, MiDaS
│   │   ├── mass/              # Density estimator, Regression
│   │   ├── segmentation/      # SAM (future)
│   │   └── material/          # Material classifier (future)
│   ├── pipeline/              # Cascade orchestrator + factory
│   ├── calibration/           # Camera calibration
│   ├── data/                  # Dataset + data prep
│   ├── evaluation/            # Metrics + benchmarking
│   ├── training/              # DVC, Optuna, MLflow training pipelines
│   ├── models/                # Custom architectures
│   ├── export/                # ONNX / TensorRT export
│   └── utils/                 # Config, logging, image helpers
├── configs/                   # YAML configurations
│   ├── pipeline/              # Pipeline configs (default, jetson)
│   ├── models/                # Training hyperparams
│   └── hardware/              # Device profiles
├── scripts/                   # Training & export scripts
├── tests/                     # Unit + integration tests
├── deploy/                    # Dockerfiles
├── data/                      # Datasets (gitignored)
├── models/weights/            # Model weights (gitignored)
├── notebooks/                 # Experiments
└── docs/                      # Technical documentation
```

## Testing

```bash
make test    # unit tests
make lint    # ruff + mypy
```

## Hardware Targets

| Device | Expected FPS | Power |
|--------|-------------|-------|
| Jetson Nano (5W) | 5–8 | 5W |
| Jetson Nano (MAXN) | 8–12 | 10W |
| Jetson Orin Nano | 15–25 | 15W |
| Desktop GPU | 30+ | — |
