"""FastAPI host inference server for edge-stage fallback execution."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from edge_ai_mass.host.serialization import (
    decode_pipeline_request,
    decode_stage_request,
    encode_pipeline_result,
    module_result_to_payload,
)
from edge_ai_mass.pipeline.factory import build_pipeline

logger = logging.getLogger(__name__)


def create_app(
    *,
    config_path: str | Path = "configs/pipeline/default.yaml",
    preload: bool = False,
    models_dir: str | Path = "models",
    validate_primary: bool = True,
) -> Any:
    """Create the FastAPI app.

    The server exposes both explicit stage endpoints and a full-pipeline
    endpoint:

    - ``GET  /health``
    - ``GET  /v1/stages/{stage}/health?load=true``
    - ``POST /v1/stages/detection``
    - ``POST /v1/stages/depth``
    - ``POST /v1/stages/mass``
    - ``POST /v1/stages/{stage}``
    - ``POST /v1/pipeline``
    """

    FastAPI, Body, HTTPException = _require_fastapi()

    app = FastAPI(
        title="Edge AI Mass Host Inference Server",
        version="1.0.0",
        description=(
            "Host-side execution provider for edge pipeline stages. "
            "Use it when an edge primary model fails and the host primary model "
            "should be tried before the edge fallback."
        ),
    )
    models_root = prepare_host_model_environment(models_dir)
    pipeline = build_pipeline(str(config_path))
    lock = threading.RLock()
    primary_status: dict[str, dict[str, Any]] = {}

    if preload:
        _load_all_host_primaries(
            pipeline,
            primary_status,
            validate_primary=validate_primary,
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "config_path": str(config_path),
            "models_dir": str(models_root),
            "stages": {
                name: {
                    "primary_loaded": bool(stage.primary.is_loaded),
                    "primary_status": primary_status.get(name, {}),
                    "fallback_loaded": (
                        bool(stage.fallback.is_loaded)
                        if stage.fallback is not None
                        else None
                    ),
                }
                for name, stage in pipeline.stages.items()
            },
        }

    @app.get("/v1/stages/{stage_name}/health")
    def stage_health(stage_name: str, load: bool = False) -> dict[str, Any]:
        stage = _stage_or_404(pipeline, stage_name, HTTPException)
        if load:
            try:
                with lock:
                    _load_and_validate_primary(
                        stage_name,
                        stage,
                        primary_status,
                        validate_primary=validate_primary,
                    )
            except Exception as exc:
                logger.exception("Host stage primary failed to load: %s", stage_name)
                raise HTTPException(
                    status_code=503,
                    detail={
                        "status": "unavailable",
                        "stage": stage_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                ) from exc
        return {
            "status": "ready" if stage.primary.is_loaded else "ok",
            "stage": stage_name,
            "primary_loaded": bool(stage.primary.is_loaded),
            "primary_status": primary_status.get(stage_name, {}),
            "models_dir": str(models_root),
            "fallback_configured": stage.fallback is not None,
        }

    @app.post("/v1/stages/detection")
    def run_detection(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return _run_stage(
            pipeline,
            "detection",
            body,
            lock,
            HTTPException,
            primary_status,
            validate_primary=validate_primary,
        )

    @app.post("/v1/stages/depth")
    def run_depth(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return _run_stage(
            pipeline,
            "depth",
            body,
            lock,
            HTTPException,
            primary_status,
            validate_primary=validate_primary,
        )

    @app.post("/v1/stages/mass")
    def run_mass(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return _run_stage(
            pipeline,
            "mass",
            body,
            lock,
            HTTPException,
            primary_status,
            validate_primary=validate_primary,
        )

    @app.post("/v1/stages/{stage_name}")
    def run_stage(stage_name: str, body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        return _run_stage(
            pipeline,
            stage_name,
            body,
            lock,
            HTTPException,
            primary_status,
            validate_primary=validate_primary,
        )

    @app.post("/v1/pipeline")
    def run_pipeline(body: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            image, include_depth_maps = decode_pipeline_request(body)
            with lock:
                _load_all_host_primaries(
                    pipeline,
                    primary_status,
                    validate_primary=validate_primary,
                )
                result = pipeline.run(image)
        except Exception as exc:
            logger.exception("Host full-pipeline inference failed")
            raise HTTPException(
                status_code=500,
                detail={"error": f"{type(exc).__name__}: {exc}"},
            ) from exc
        return {
            "result": encode_pipeline_result(
                result,
                include_depth_maps=include_depth_maps,
            )
        }

    return app


def run_server(
    *,
    config_path: str | Path = "configs/pipeline/default.yaml",
    host: str = "0.0.0.0",
    port: int = 8090,
    preload: bool = False,
    models_dir: str | Path = "models",
    validate_primary: bool = True,
    log_level: str = "info",
) -> None:
    """Run the host inference app with Uvicorn."""

    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            'Host inference server requires uvicorn. Install with: pip install -e ".[host]"'
        ) from exc
    app = create_app(
        config_path=config_path,
        preload=preload,
        models_dir=models_dir,
        validate_primary=validate_primary,
    )
    uvicorn.run(app, host=host, port=int(port), log_level=log_level)


def prepare_host_model_environment(models_dir: str | Path = "models") -> Path:
    """Create and expose host model/cache directories under ``models/``.

    Host-side execution may download or load assets for several frameworks:
    Ultralytics detector weights, Hugging Face depth checkpoints, Torch Hub
    MiDaS fallback weights, and tabular mass artifacts.  Keeping all of those
    under a single project-local model root makes releases and cloud instances
    much easier to reason about than silently using user-level cache folders.
    """

    root = Path(models_dir).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()

    paths = {
        "weights": root / "weights",
        "huggingface": root / "huggingface",
        "huggingface_hub": root / "huggingface" / "hub",
        "torch": root / "torch",
        "torch_hub": root / "torch" / "hub",
        "ultralytics": root / "ultralytics",
        "debug": root / "debug",
    }
    for path in (root, *paths.values()):
        path.mkdir(parents=True, exist_ok=True)

    # These variables are intentionally set for the host process, not merely
    # defaulted, because the host release contract is: runtime models live under
    # the project ``models/`` directory unless the operator chooses another
    # ``--models-dir``.
    os.environ["EDGE_AI_MODELS_DIR"] = str(root)
    os.environ["HF_HOME"] = str(paths["huggingface"])
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(paths["huggingface_hub"])
    os.environ["TORCH_HOME"] = str(paths["torch"])
    os.environ["YOLO_CONFIG_DIR"] = str(paths["ultralytics"])
    os.environ["ULTRALYTICS_CACHE_DIR"] = str(paths["ultralytics"])
    logger.info("Host model environment prepared under %s", root)
    return root


def _run_stage(
    pipeline: Any,
    stage_name: str,
    body: dict[str, Any],
    lock: threading.RLock,
    HTTPException: type[Exception],
    primary_status: dict[str, dict[str, Any]],
    *,
    validate_primary: bool,
) -> dict[str, Any]:
    stage = _stage_or_404(pipeline, stage_name, HTTPException)
    try:
        image, kwargs = decode_stage_request(body)
        # Host endpoints intentionally execute the host primary for this stage.
        # They do not hide a host fallback behind the edge's fallback policy.
        with lock:
            _load_and_validate_primary(
                stage_name,
                stage,
                primary_status,
                validate_primary=validate_primary,
            )
            logger.info("Running host primary stage '%s' with model primary status: %s", stage_name, primary_status)
            result = stage.primary.predict(image, **kwargs)
    except Exception as exc:
        logger.exception("Host stage inference failed: %s", stage_name)
        raise HTTPException(
            status_code=500,
            detail={"stage": stage_name, "error": f"{type(exc).__name__}: {exc}"},
        ) from exc
    response = module_result_to_payload(result)
    response.setdefault("metadata", {})
    response["metadata"]["host_execution"] = True
    response["metadata"]["host_stage"] = stage_name
    return response


def _load_all_host_primaries(
    pipeline: Any,
    primary_status: dict[str, dict[str, Any]],
    *,
    validate_primary: bool,
) -> None:
    """Load and smoke-test every host primary stage."""

    for stage_name, stage in pipeline.stages.items():
        _load_and_validate_primary(
            stage_name,
            stage,
            primary_status,
            validate_primary=validate_primary,
        )


def _load_and_validate_primary(
    stage_name: str,
    stage: Any,
    primary_status: dict[str, dict[str, Any]],
    *,
    validate_primary: bool,
) -> dict[str, Any]:
    """Load a host primary module and run a small stage-aware smoke test."""

    cached = primary_status.get(stage_name) or {}
    if stage.primary.is_loaded and (
        not validate_primary or cached.get("validated") is True
    ):
        return cached

    started = time.perf_counter()
    status: dict[str, Any] = {
        "stage": stage_name,
        "loaded": False,
        "validated": False,
    }
    try:
        if not stage.primary.is_loaded:
            logger.info("Loading host primary stage '%s'", stage_name)
            stage.primary.load()
        status["loaded"] = bool(stage.primary.is_loaded)
        status["load_elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)

        if validate_primary:
            logger.info("Smoke-testing host primary stage '%s'", stage_name)
            test_started = time.perf_counter()
            test_result = _smoke_test_primary(stage_name, stage.primary)
            status.update(
                {
                    "validated": True,
                    "test_elapsed_ms": round((time.perf_counter() - test_started) * 1000, 3),
                    "test_latency_ms": float(getattr(test_result, "latency_ms", 0.0)),
                    "result_type": type(getattr(test_result, "data", None)).__name__,
                    "metadata": dict(getattr(test_result, "metadata", {}) or {}),
                }
            )
        primary_status[stage_name] = status
        return status
    except Exception as exc:
        status.update(
            {
                "loaded": bool(getattr(stage.primary, "is_loaded", False)),
                "validated": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        primary_status[stage_name] = status
        raise


def _smoke_test_primary(stage_name: str, module: Any) -> Any:
    """Run the smallest practical inference that proves a primary is usable."""

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    if stage_name == "mass":
        return module.predict(image, features=_smoke_test_mass_features())
    return module.predict(image)


def _smoke_test_mass_features() -> dict[str, Any]:
    """Synthetic object features for validating the residual mass primary."""

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:44, 20:44] = 1
    return {
        "bbox": [20, 20, 44, 44],
        "mask": mask,
        "class_id": 0,
        "class_name": "plastic_bottle",
        "confidence": 0.95,
        "depth_stats": {
            "mean": 0.8,
            "median": 0.8,
            "std": 0.02,
            "min": 0.75,
            "max": 0.85,
            "p10": 0.77,
            "p90": 0.83,
            "iqr": 0.04,
            "range": 0.10,
            "valid_ratio": 1.0,
        },
        "geometry": {
            "width_m": 0.024,
            "height_m": 0.024,
            "volume_m3": 1.2e-5,
            "mean_object_depth_m": 0.8,
            "mean_background_depth_m": 1.0,
            "method": "host_smoke_test",
            "calibration_id": "host-smoke-test",
            "depth_scale_id": "host-smoke-test",
            "background_id": "host-smoke-test",
        },
        "volume_m3": 1.2e-5,
        "material": "plastic",
        "warnings": ["host_smoke_test"],
    }


def _stage_or_404(pipeline: Any, stage_name: str, HTTPException: type[Exception]) -> Any:
    stage = pipeline.stages.get(stage_name)
    if stage is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "unknown_stage",
                "stage": stage_name,
                "available_stages": sorted(pipeline.stages),
            },
        )
    return stage


def _require_fastapi() -> tuple[Any, Any, Any]:
    try:
        from fastapi import Body, FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            'Host inference server requires FastAPI. Install with: pip install -e ".[host]"'
        ) from exc
    return FastAPI, Body, HTTPException
