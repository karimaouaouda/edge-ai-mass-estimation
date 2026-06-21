"""Build a Pipeline from a YAML configuration file.

The factory reads the config, instantiates the right module classes, wraps them
in ``Stage`` objects with fallbacks, and returns a ready-to-use ``Pipeline``.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any

from edge_ai_mass.modules.base import BaseModule
from edge_ai_mass.modules.geometry import GeometryEstimator
from edge_ai_mass.pipeline.pipeline import Pipeline, Stage
from edge_ai_mass.utils.config import load_config

logger = logging.getLogger(__name__)


def show_env() -> None:
    print("python =", sys.executable)
    print("cwd =", os.getcwd())
    print("USER =", os.environ.get("USER"))
    print("CONDA_PREFIX =", os.environ.get("CONDA_PREFIX"))
    print("LD_PRELOAD =", os.environ.get("LD_PRELOAD"))
    print("LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH"))
    print("PATH =", os.environ.get("PATH"))


def _instantiate_module(spec: dict[str, Any]) -> BaseModule:
    """Dynamically import and instantiate a module from its dotted class path."""

    show_env()
    class_path: str = spec["class"]
    module_path, class_name = class_path.rsplit(".", 1)
    print(f"Instantiating module: {class_path} with params: {spec.get('params', {})}")
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(config=spec.get("params", {}))


def build_pipeline(config_path: str) -> Pipeline:
    """Create a fully-wired ``Pipeline`` from a YAML config file."""
    cfg = load_config(config_path)
    pipeline = Pipeline()

    for stage_name, stage_cfg in cfg.get("stages", {}).items():
        primary = _instantiate_module(stage_cfg["primary"])

        fallback = None
        if "fallback" in stage_cfg:
            fallback = _instantiate_module(stage_cfg["fallback"])

        budget = stage_cfg.get("latency_budget_ms", float("inf"))

        pipeline.add_stage(
            stage_name,
            Stage(
                name=stage_name,
                primary=primary,
                fallback=fallback,
                latency_budget_ms=budget,
                fallback_on_latency_exceeded=bool(
                    stage_cfg.get("fallback_on_latency_exceeded", True)
                ),
            ),
        )
        logger.info(
            "Registered stage '%s' — primary=%s  fallback=%s  budget=%.0f ms",
            stage_name,
            stage_cfg["primary"]["class"],
            stage_cfg.get("fallback", {}).get("class", "none"),
            budget,
        )

    if _geometry_enabled(cfg):
        pipeline.set_geometry_estimator(GeometryEstimator.from_config(cfg))
        logger.info("Registered geometry estimator for calibration-aware volume estimation")

    return pipeline


def _geometry_enabled(cfg: dict[str, Any]) -> bool:
    geometry_cfg = cfg.get("geometry") or {}
    if "enabled" in geometry_cfg:
        return bool(geometry_cfg["enabled"])
    return any(key in cfg for key in ("calibration", "depth_scale", "background"))
