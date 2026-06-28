"""Capture camera frames and record depth metrics using the configured depth model.

Examples:
    python scripts/mass/capture_depth_metrics.py \
        --config scripts/mass/depth_capture.yaml \
        --profile depth_anything

    python scripts/mass/capture_depth_metrics.py \
        --config scripts/mass/depth_capture.yaml \
        --profile midas
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from edge_ai_mass.calibration.depth_scale import DepthScaleConfig, raw_depth_to_metric
from edge_ai_mass.utils.config import load_config
from edge_ai_mass.utils.logging import setup_logging
from scripts.mass.utils import (
    append_csv_row,
    depth_statistics,
    load_profiled_config,
    save_depth_preview,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture frames from a camera and save depth metrics from YAML config."
    )
    parser.add_argument(
        "--config",
        default="scripts/mass/depth_capture.yaml",
        help="YAML script config with camera, output, and model profile settings.",
    )
    parser.add_argument("--profile", default=None, help="Profile name inside the YAML config")
    args = parser.parse_args()

    script_cfg, selected_profile = load_profiled_config(args.config, args.profile)
    setup_logging(str(script_cfg.get("log_level", "INFO")))

    pipeline_config = str(script_cfg.get("pipeline_config", "configs/pipeline/default.yaml"))
    cfg = load_config(pipeline_config)
    depth_source = str(script_cfg.get("depth_source", "primary"))
    depth_module = _load_depth_module(cfg, source=depth_source)
    depth_scale = DepthScaleConfig.from_config(cfg.get("depth_scale", {}))

    camera = str(script_cfg.get("camera", "0"))
    camera_source: int | str = int(camera) if camera.isdigit() else camera
    capture = cv2.VideoCapture(camera_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera source: {camera}")

    output = Path(str(script_cfg["output"]))
    save_depth_dir = _optional_path(script_cfg.get("save_depth_dir"))
    preview_dir = _optional_path(script_cfg.get("preview_dir"))
    frames = int(script_cfg.get("frames", 1))
    interval_seconds = float(script_cfg.get("interval_seconds", 0.0))
    display = bool(script_cfg.get("display", False))
    preview_source = str(script_cfg.get("preview_source", "metric"))
    if save_depth_dir:
        save_depth_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    try:
        for frame_index in range(frames):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read frame {frame_index} from camera")

            timestamp = datetime.now(timezone.utc).isoformat()
            result = depth_module.predict(frame)
            raw_depth = np.asarray(result.data, dtype=np.float32)
            metric_depth, valid_mask = raw_depth_to_metric(raw_depth, depth_scale)
            metrics = depth_statistics(metric_depth, valid_mask)
            stem = f"frame_{frame_index:06d}"
            row = {
                "frame_index": frame_index,
                "timestamp_utc": timestamp,
                "camera": camera,
                "depth_source": depth_source,
                "script_profile": selected_profile,
                "depth_module": depth_module.__class__.__name__,
                "latency_ms": float(result.latency_ms),
                "height_px": int(frame.shape[0]),
                "width_px": int(frame.shape[1]),
                "depth_scale_id": depth_scale.calibration_id,
                **metrics,
            }
            if save_depth_dir:
                raw_path = save_depth_dir / f"{stem}_raw_depth.npy"
                metric_path = save_depth_dir / f"{stem}_metric_depth.npy"
                np.save(raw_path, raw_depth)
                np.save(metric_path, metric_depth)
                row["raw_depth_path"] = str(raw_path)
                row["metric_depth_path"] = str(metric_path)
            if preview_dir:
                preview_depth = raw_depth if preview_source == "raw" else metric_depth
                preview_path = save_depth_preview(preview_depth, preview_dir / f"{stem}.png")
                row["depth_preview_path"] = str(preview_path)
            append_csv_row(output, row)

            if display:
                cv2.imshow("depth-metrics-camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if interval_seconds > 0 and frame_index < frames - 1:
                time.sleep(interval_seconds)
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()

    write_json(
        output.with_suffix(".metadata.json"),
        {
            "config": str(Path(args.config).resolve()),
            "pipeline_config": str(Path(pipeline_config).resolve()),
            "profile": selected_profile,
            "camera": camera,
            "depth_source": depth_source,
            "frames_requested": frames,
            "output": str(output),
            "effective_config": script_cfg,
        },
    )
    print(f"Wrote depth metrics to {output}")


def _load_depth_module(config: dict[str, Any], *, source: str):
    stage = config.get("stages", {}).get("depth")
    if not isinstance(stage, dict):
        raise ValueError("Config must contain stages.depth")
    spec = stage.get(source)
    if not isinstance(spec, dict):
        raise ValueError(f"Config does not define stages.depth.{source}")
    class_path = str(spec["class"])
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    instance = cls(config=dict(spec.get("params", {})))
    instance.load()
    return instance


def _optional_path(value: Any) -> Path | None:
    if value in (None, "", False):
        return None
    return Path(str(value))


if __name__ == "__main__":
    main()
