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
# Fine-tune YOLOv8 on your waste dataset
python scripts/train_detector.py --config configs/models/yolov8_training.yaml

# Collect features for mass regression
python scripts/collect_mass_features.py \
    --images data/raw/mass_dataset/ \
    --labels data/raw/mass_labels.csv \
    --output data/splits/mass_features.csv

# Train mass regression head
python scripts/train_mass_regression.py --config configs/models/mass_regression_training.yaml
```

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
