"""Shared utilities for mass-estimation data-collection scripts."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class MaskGeometry:
    """Geometry features computed from a binary object mask."""

    mask_area_px: int
    bbox_x_px: int
    bbox_y_px: int
    bbox_width_px: int
    bbox_height_px: int
    bbox_area_px: int
    contour_area_px: float
    perimeter_px: float
    convex_hull_area_px: float
    area_cm2: float
    bbox_width_cm: float
    bbox_height_cm: float
    bbox_area_cm2: float
    aspect_ratio: float
    solidity: float
    extent: float
    compactness: float


def load_profiled_config(
    path: str | Path,
    profile: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Load a script YAML config and optionally merge one named profile.

    Config files can keep shared defaults at the top level and put switchable
    variants under `profiles`. The selected profile overrides those defaults.
    """

    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Script config must contain a YAML mapping: {config_path}")

    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        return payload, None

    selected = profile or str(payload.get("active_profile", "default"))
    if selected not in profiles:
        raise ValueError(
            f"Profile '{selected}' was not found in {config_path}. "
            f"Available profiles: {sorted(profiles)}"
        )
    base = {key: value for key, value in payload.items() if key != "profiles"}
    profile_payload = profiles[selected]
    if not isinstance(profile_payload, dict):
        raise ValueError(f"Profile '{selected}' must contain a YAML mapping")
    return deep_merge(base, profile_payload), selected


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries without mutating the inputs."""

    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_mask(mask_path: str | Path) -> np.ndarray:
    """Load a binary mask from an image or `.npy` file."""

    path = Path(mask_path)
    if not path.is_file():
        raise FileNotFoundError(f"Mask file not found: {path}")
    if path.suffix.lower() == ".npy":
        mask = np.load(path)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask image: {path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        raise ValueError(f"Mask contains no foreground pixels: {path}")
    return binary


def mask_geometry(mask: np.ndarray, *, pixel_per_cm: float) -> MaskGeometry:
    """Compute physical 2D shape features from a binary mask."""

    if pixel_per_cm <= 0:
        raise ValueError("pixel_per_cm must be greater than zero")
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if xs.size == 0:
        raise ValueError("Mask contains no foreground pixels")

    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    bbox_width = int(x_max - x_min + 1)
    bbox_height = int(y_max - y_min + 1)
    bbox_area = int(bbox_width * bbox_height)
    mask_area = int(binary.sum())

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    contour_area = float(cv2.contourArea(contour)) if contour is not None else float(mask_area)
    perimeter = float(cv2.arcLength(contour, True)) if contour is not None else 0.0
    if contour is not None and len(contour) >= 3:
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
    else:
        hull_area = contour_area

    area_cm2 = mask_area / (pixel_per_cm**2)
    bbox_width_cm = bbox_width / pixel_per_cm
    bbox_height_cm = bbox_height / pixel_per_cm
    bbox_area_cm2 = bbox_area / (pixel_per_cm**2)
    aspect_ratio = bbox_width / max(float(bbox_height), 1.0)
    solidity = contour_area / hull_area if hull_area > 0 else 0.0
    extent = mask_area / bbox_area if bbox_area > 0 else 0.0
    compactness = (
        (4.0 * math.pi * contour_area) / (perimeter**2)
        if perimeter > 0 and contour_area > 0
        else 0.0
    )

    return MaskGeometry(
        mask_area_px=mask_area,
        bbox_x_px=x_min,
        bbox_y_px=y_min,
        bbox_width_px=bbox_width,
        bbox_height_px=bbox_height,
        bbox_area_px=bbox_area,
        contour_area_px=contour_area,
        perimeter_px=perimeter,
        convex_hull_area_px=hull_area,
        area_cm2=area_cm2,
        bbox_width_cm=bbox_width_cm,
        bbox_height_cm=bbox_height_cm,
        bbox_area_cm2=bbox_area_cm2,
        aspect_ratio=aspect_ratio,
        solidity=solidity,
        extent=extent,
        compactness=compactness,
    )


def normalize_depth_value(value: float, *, unit: str) -> float:
    """Return a depth or height value in centimetres."""

    normalized = unit.strip().lower()
    if normalized in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
        return float(value)
    if normalized in {"m", "meter", "meters", "metre", "metres"}:
        return float(value) * 100.0
    if normalized in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
        return float(value) / 10.0
    raise ValueError(f"Unsupported depth unit: {unit}")


def estimate_thickness_cm(
    *,
    camera_height_cm: float,
    background_default_depth_cm: float,
    mode: str = "camera_minus_background",
    min_thickness_cm: float = 0.0,
) -> float:
    """Estimate object thickness from the two collection-time depth references.

    The default mode treats `camera_height_cm` as the camera-to-support-plane
    distance and `background_default_depth_cm` as the measured/assumed object
    surface distance. If your setup stores the values in the opposite direction,
    use `background_minus_camera`.
    """

    if mode == "camera_minus_background":
        value = camera_height_cm - background_default_depth_cm
    elif mode == "background_minus_camera":
        value = background_default_depth_cm - camera_height_cm
    elif mode == "absolute_difference":
        value = abs(camera_height_cm - background_default_depth_cm)
    else:
        raise ValueError(
            "thickness mode must be camera_minus_background, "
            "background_minus_camera, or absolute_difference"
        )
    return max(float(value), float(min_thickness_cm), 0.0)


def depth_statistics(depth_map: np.ndarray, valid_mask: np.ndarray | None = None) -> dict[str, float]:
    """Compute robust scalar depth metrics for a full frame or ROI."""

    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask).astype(bool)
    values = depth[valid]
    if values.size == 0:
        return {
            "depth_mean": 0.0,
            "depth_median": 0.0,
            "depth_std": 0.0,
            "depth_min": 0.0,
            "depth_max": 0.0,
            "depth_p05": 0.0,
            "depth_p95": 0.0,
            "depth_valid_ratio": 0.0,
        }
    return {
        "depth_mean": float(np.mean(values)),
        "depth_median": float(np.median(values)),
        "depth_std": float(np.std(values)),
        "depth_min": float(np.min(values)),
        "depth_max": float(np.max(values)),
        "depth_p05": float(np.percentile(values, 5)),
        "depth_p95": float(np.percentile(values, 95)),
        "depth_valid_ratio": float(values.size / max(depth.size, 1)),
    }


def load_density_priors(
    *,
    priors_path: str | Path | None = None,
    inline_pairs: list[str] | None = None,
) -> dict[str, float]:
    """Load class density priors in g/cm3 from JSON/YAML and CLI pairs."""

    priors: dict[str, float] = {}
    if priors_path:
        path = Path(priors_path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Density priors file must contain a mapping: {path}")
        payload = payload.get("density_priors_g_cm3", payload)
        if not isinstance(payload, dict):
            raise ValueError("density_priors_g_cm3 must be a mapping")
        priors.update({str(key): float(value) for key, value in payload.items()})

    for pair in inline_pairs or []:
        if "=" not in pair:
            raise ValueError(f"Class density prior must use class=value syntax: {pair}")
        name, raw_value = pair.split("=", 1)
        priors[name.strip()] = float(raw_value)
    return priors


def density_for_class(
    class_name: str,
    *,
    priors_g_cm3: dict[str, float],
    fallback_g_cm3: float | None = None,
) -> float:
    """Resolve a density prior for one class, with case-insensitive lookup."""

    if class_name in priors_g_cm3:
        return float(priors_g_cm3[class_name])
    lowered = {key.lower(): value for key, value in priors_g_cm3.items()}
    if class_name.lower() in lowered:
        return float(lowered[class_name.lower()])
    if fallback_g_cm3 is not None:
        return float(fallback_g_cm3)
    raise KeyError(
        f"No density prior found for class '{class_name}'. Provide --density-prior-g-cm3 "
        "or --class-density-prior class=value."
    )


def append_csv_row(path: str | Path, row: dict[str, Any]) -> Path:
    """Append one row to a CSV file, creating a header when needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exists = output_path.is_file() and output_path.stat().st_size > 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return output_path


def write_json(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return output_path


def save_depth_preview(depth_map: np.ndarray, path: str | Path) -> Path:
    """Write a colorized depth preview that remains visible for constant maps."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    depth = np.asarray(depth_map, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros(depth.shape[:2], dtype=np.uint8)
    if np.any(valid):
        values = depth[valid]
        low = float(np.percentile(values, 2))
        high = float(np.percentile(values, 98))
        if high <= low:
            normalized[valid] = 160
        else:
            scaled = (depth[valid] - low) / (high - low)
            normalized[valid] = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    color[~valid] = (32, 32, 32)
    if not cv2.imwrite(str(output_path), color):
        raise RuntimeError(f"OpenCV could not write depth preview: {output_path}")
    return output_path
