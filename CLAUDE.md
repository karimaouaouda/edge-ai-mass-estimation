# Edge AI Mass Estimation — Development Guide

## Project overview
Edge AI system for waste object characterization and mass estimation, designed to run on NVIDIA Jetson Nano with an RGB camera. Uses a cascade pipeline: Detection → Depth → Mass.

## Quick start
```bash
make dev              # install with all extras
make test             # run unit tests
make lint             # ruff + mypy
edge-ai-mass --help   # CLI entry point
```

## Architecture
The system is organized around a **pipeline of stages**, each with a primary model and optional fallback:

- **Detection**: YOLOv8-seg (primary) → MobileNetV3-SSD (fallback)
- **Depth**: Depth Anything V2 (primary) → MiDaS small (fallback)
- **Mass**: Regression MLP (primary) → Density lookup (fallback)

All modules implement `BaseModule` (`src/edge_ai_mass/modules/base.py`) with `load()`, `_forward()`, and `predict()`.

## Key directories
- `src/edge_ai_mass/` — installable Python package
- `configs/` — YAML configs for pipeline, model training, hardware profiles
- `scripts/` — training, export, and data collection scripts
- `tests/` — pytest unit and integration tests
- `deploy/` — Dockerfiles for desktop and Jetson
- `docs/` — project PDFs and technical documentation

## Adding a new AI module
1. Create a new file under `src/edge_ai_mass/modules/<category>/`
2. Subclass `BaseModule` and implement `load()` and `_forward()`
3. Register it in the YAML config under the appropriate stage
4. Add unit tests in `tests/unit/`

## Testing
```bash
pytest tests/unit/ -v         # unit tests only (no GPU needed)
pytest tests/integration/ -v  # requires models downloaded
```

## Conventions
- Python 3.10+, type hints everywhere
- Ruff for linting, mypy for type checking
- Config-driven: pipeline behavior changes via YAML, not code edits
- All model weights go in `models/weights/` (gitignored, download via scripts)
