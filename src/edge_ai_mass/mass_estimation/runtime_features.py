"""Runtime feature construction for the trained residual mass model.

The training pipeline learns a residual correction on top of a physics base:

    mass_base_g = selected_volume_cm3 * effective_density_g_cm3

This module builds the same object-level tabular feature contract during
dashboard-triggered inference, using outputs from detection, depth, and
calibration-aware geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from edge_ai_mass.modules.material import ClassMaterialMapper


DEFAULT_DENSITIES_KG_M3: dict[str, float] = {
    "plastic": 40.0,
    "glass": 300.0,
    "metal": 160.0,
    "paper": 90.0,
    "cardboard": 60.0,
    "organic": 500.0,
    "textile": 120.0,
    "wood": 250.0,
    "other": 100.0,
}

DEFAULT_HOLLOW_PRIORS: dict[str, float] = {
    "plastic_battle": 1.0,
    "plastic_bottle": 1.0,
    "bottle": 1.0,
    "metal_can": 1.0,
    "can": 1.0,
    "glass": 0.7,
    "glass_bottle": 0.7,
    "glass_jar": 0.7,
    "paper_cardboard": 0.2,
    "cardboard": 0.2,
    "plastic_bag": 0.4,
    "rigid_plastic": 0.4,
    "mixed_waste": 0.5,
    "textile_trash": 0.5,
    "textile": 0.5,
    "organic_waste": 0.0,
    "organic": 0.0,
    "vegetation": 0.0,
}


@dataclass(frozen=True, slots=True)
class RuntimeMassFeatureRow:
    """A model-ready feature row plus non-fatal feature-quality warnings."""

    row: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def build_runtime_mass_feature_row(
    features: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> RuntimeMassFeatureRow:
    """Build one residual-mass-model feature row from pipeline stage outputs.

    Parameters
    ----------
    features:
        Object-level outputs assembled by the inference pipeline. Expected keys
        include `bbox`, `mask`, `class_name`, `confidence`, `depth_stats`,
        `geometry`, and `volume_m3`.
    config:
        Mass estimator configuration containing density priors, class/material
        mapping, and conservative fallback geometry defaults.
    """

    cfg = config or {}
    warnings = [str(item) for item in features.get("warnings", [])]

    class_name = str(features.get("class_name") or "other")
    material = str(
        features.get("material")
        or ClassMaterialMapper.from_config(cfg).material_for(class_name)
        or "other"
    )
    bbox = _bbox_array(features.get("bbox"))
    mask = _binary_mask(features.get("mask"))
    if bbox is None:
        bbox = _bbox_from_mask(mask)
        warnings.append("mass_features_bbox_derived_from_mask")
    if bbox is None:
        bbox = np.zeros(4, dtype=np.float32)
        warnings.append("mass_features_missing_bbox")

    shape = _shape_features(mask, bbox)
    geometry = features.get("geometry") if isinstance(features.get("geometry"), dict) else {}
    depth_stats = (
        features.get("depth_stats") if isinstance(features.get("depth_stats"), dict) else {}
    )
    scales = _physical_scale_features(shape, geometry, cfg)
    depths = _depth_features(depth_stats, geometry, cfg)
    density_g_cm3 = _resolve_density_g_cm3(class_name, material, cfg)
    hollow_prior = _resolve_hollow_prior(class_name, material, cfg)

    thickness_prior_cm = _positive_float(
        cfg.get("thickness_prior_cm"),
        cfg.get("default_thickness_cm"),
        _positive_float(cfg.get("default_thickness_m"), default=0.01) * 100.0,
        default=1.0,
    )
    estimated_thickness_cm = _positive_float(
        depths.get("geometry_height_cm"),
        cfg.get("estimated_thickness_cm"),
        thickness_prior_cm,
        default=thickness_prior_cm,
    )
    thickness_source = (
        str(geometry.get("method") or "geometry_depth")
        if depths.get("geometry_height_cm", 0.0) > 0
        else str(cfg.get("thickness_source", "config_default"))
    )

    mask_area_cm2 = shape["mask_area_px"] * scales["pixel_area_cm2"]
    bbox_area_cm2 = shape["bbox_area_px"] * scales["pixel_area_cm2"]
    bbox_width_cm = shape["bbox_width_px"] * scales["cm_per_pixel_x"]
    bbox_height_cm = shape["bbox_height_px"] * scales["cm_per_pixel_y"]
    perimeter_cm = shape["perimeter_px"] * math.sqrt(max(scales["pixel_area_cm2"], 0.0))
    convex_hull_area_cm2 = shape["convex_hull_area_px"] * scales["pixel_area_cm2"]

    volume_box_cm3 = bbox_area_cm2 * estimated_thickness_cm
    volume_mask_cm3 = mask_area_cm2 * estimated_thickness_cm
    geometry_volume_m3 = _positive_float(features.get("volume_m3"), geometry.get("volume_m3"))
    if geometry_volume_m3 > 0:
        volume_depth_cm3 = geometry_volume_m3 * 1_000_000.0
        selected_volume_cm3 = volume_depth_cm3
        volume_source = str(geometry.get("method") or cfg.get("volume_source", "geometry_depth"))
    else:
        volume_depth_cm3 = volume_mask_cm3
        selected_volume_cm3 = volume_mask_cm3
        volume_source = str(cfg.get("volume_source", "mask_thickness"))
        warnings.append("mass_features_used_mask_thickness_volume")

    effective_density_kg_m3 = density_g_cm3 * 1000.0
    mass_base_g = selected_volume_cm3 * density_g_cm3
    label_confidence = float(cfg.get("runtime_label_confidence", cfg.get("label_confidence", 1.0)))

    row: dict[str, Any] = {
        "class_name": class_name,
        "material": material,
        "dimension_source": scales["dimension_source"],
        "thickness_source": thickness_source,
        "volume_source": volume_source,
        "baseline_method": str(cfg.get("baseline_method", "volume_density")),
        "hollow_prior": hollow_prior,
        "effective_density_g_cm3": density_g_cm3,
        "effective_density_kg_m3": effective_density_kg_m3,
        "cm_per_pixel_x": scales["cm_per_pixel_x"],
        "cm_per_pixel_y": scales["cm_per_pixel_y"],
        "pixel_area_cm2": scales["pixel_area_cm2"],
        "mask_area_px": float(shape["mask_area_px"]),
        "mask_area_cm2": mask_area_cm2,
        "bbox_width_px": shape["bbox_width_px"],
        "bbox_height_px": shape["bbox_height_px"],
        "bbox_width_cm": bbox_width_cm,
        "bbox_height_cm": bbox_height_cm,
        "bbox_area_px": shape["bbox_area_px"],
        "bbox_area_cm2": bbox_area_cm2,
        "aspect_ratio": shape["aspect_ratio"],
        "perimeter_px": shape["perimeter_px"],
        "perimeter_cm": perimeter_cm,
        "convex_hull_area_px": shape["convex_hull_area_px"],
        "convex_hull_area_cm2": convex_hull_area_cm2,
        "solidity": shape["solidity"],
        "extent": shape["extent"],
        "compactness": shape["compactness"],
        "circularity": shape["circularity"],
        "depth_mean": depths["depth_mean"],
        "depth_median": depths["depth_median"],
        "depth_std": depths["depth_std"],
        "depth_min": depths["depth_min"],
        "depth_max": depths["depth_max"],
        "depth_range": depths["depth_range"],
        "depth_p10": depths["depth_p10"],
        "depth_p90": depths["depth_p90"],
        "depth_iqr": depths["depth_iqr"],
        "depth_valid_ratio": depths["depth_valid_ratio"],
        "background_depth_raw": depths["background_depth_raw"],
        "object_to_background_depth_ratio": depths["object_to_background_depth_ratio"],
        "estimated_thickness_cm": estimated_thickness_cm,
        "table_depth_cm": depths["table_depth_cm"],
        "object_depth_cm": depths["object_depth_cm"],
        "thickness_prior_cm": thickness_prior_cm,
        "volume_box_cm3": volume_box_cm3,
        "volume_mask_cm3": volume_mask_cm3,
        "volume_depth_cm3": volume_depth_cm3,
        "selected_volume_cm3": selected_volume_cm3,
        "estimated_volume_m3": selected_volume_cm3 * 1e-6,
        "mass_base_g": mass_base_g,
        "label_confidence": label_confidence,
    }

    row.update(
        {
            "class_id": int(features.get("class_id", -1)),
            "detection_confidence": float(features.get("confidence", 0.0)),
            "calibration_id": geometry.get("calibration_id") or features.get("calibration_id"),
            "depth_scale_id": geometry.get("depth_scale_id") or features.get("depth_scale_id"),
            "background_id": geometry.get("background_id") or features.get("background_id"),
        }
    )
    return RuntimeMassFeatureRow(row=row, warnings=warnings)


def _bbox_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 4:
        return None
    return arr[:4]


def _binary_mask(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    mask = np.asarray(value)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if mask.ndim != 2:
        return None
    binary = (mask > 0).astype(np.uint8)
    return binary if int(binary.sum()) > 0 else None


def _bbox_from_mask(mask: np.ndarray | None) -> np.ndarray | None:
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return np.asarray([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _shape_features(mask: np.ndarray | None, bbox: np.ndarray) -> dict[str, float]:
    x1, y1, x2, y2 = bbox.astype(float)
    bbox_width = max(float(x2 - x1), 0.0)
    bbox_height = max(float(y2 - y1), 0.0)
    bbox_area = bbox_width * bbox_height

    if mask is None:
        mask_area = bbox_area
        contour_area = bbox_area
        perimeter = 2.0 * (bbox_width + bbox_height) if bbox_area > 0 else 0.0
        hull_area = bbox_area
    else:
        mask_area = float(np.sum(mask > 0))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea) if contours else None
        contour_area = float(cv2.contourArea(contour)) if contour is not None else mask_area
        perimeter = float(cv2.arcLength(contour, True)) if contour is not None else 0.0
        if contour is not None and len(contour) >= 3:
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        else:
            hull_area = max(contour_area, mask_area)

    if bbox_area <= 0 and mask is not None:
        derived = _bbox_from_mask(mask)
        if derived is not None:
            return _shape_features(mask, derived)

    solidity = _safe_div(contour_area, hull_area, default=0.0)
    extent = _safe_div(mask_area, bbox_area, default=0.0)
    compactness = _safe_div(perimeter**2, 4.0 * math.pi * contour_area, default=0.0)
    circularity = _safe_div(1.0, compactness, default=0.0) if compactness > 0 else 0.0
    return {
        "mask_area_px": float(mask_area),
        "bbox_width_px": bbox_width,
        "bbox_height_px": bbox_height,
        "bbox_area_px": bbox_area,
        "aspect_ratio": _safe_div(bbox_width, bbox_height, default=0.0),
        "perimeter_px": perimeter,
        "convex_hull_area_px": hull_area,
        "solidity": solidity,
        "extent": extent,
        "compactness": compactness,
        "circularity": circularity,
    }


def _physical_scale_features(
    shape: dict[str, float],
    geometry: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, float | str]:
    pixel_to_m = _positive_float(
        config.get("pixel_to_m"),
        config.get("default_pixel_to_m"),
        default=0.001,
    )
    width_m = _positive_float(geometry.get("width_m"))
    height_m = _positive_float(geometry.get("height_m"))
    dimension_source = "geometry"

    if width_m <= 0:
        width_m = shape["bbox_width_px"] * pixel_to_m
        dimension_source = str(config.get("dimension_source", "pixel_to_m_config"))
    if height_m <= 0:
        height_m = shape["bbox_height_px"] * pixel_to_m
        dimension_source = str(config.get("dimension_source", "pixel_to_m_config"))

    cm_per_pixel_x = _safe_div(width_m * 100.0, shape["bbox_width_px"], default=pixel_to_m * 100.0)
    cm_per_pixel_y = _safe_div(
        height_m * 100.0,
        shape["bbox_height_px"],
        default=pixel_to_m * 100.0,
    )
    return {
        "dimension_source": dimension_source,
        "cm_per_pixel_x": cm_per_pixel_x,
        "cm_per_pixel_y": cm_per_pixel_y,
        "pixel_area_cm2": cm_per_pixel_x * cm_per_pixel_y,
    }


def _depth_features(
    depth_stats: dict[str, Any],
    geometry: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, float]:
    depth_mean = _stat(depth_stats, "mean", "depth_mean")
    depth_median = _stat(depth_stats, "median", "depth_median", default=depth_mean)
    depth_std = _stat(depth_stats, "std", "depth_std")
    depth_min = _stat(depth_stats, "min", "depth_min", default=depth_median)
    depth_max = _stat(depth_stats, "max", "depth_max", default=depth_median)
    depth_p10 = _stat(depth_stats, "p10", "depth_p10", default=depth_min)
    depth_p90 = _stat(depth_stats, "p90", "depth_p90", default=depth_max)
    depth_iqr = _stat(
        depth_stats,
        "iqr",
        "depth_iqr",
        default=max(depth_p90 - depth_p10, 0.0),
    )
    depth_range = _stat(
        depth_stats,
        "range",
        "depth_range",
        default=max(depth_max - depth_min, 0.0),
    )
    depth_valid_ratio = _stat(depth_stats, "valid_ratio", "depth_valid_ratio", default=1.0)

    background_raw = _positive_float(
        geometry.get("mean_background_depth_m"),
        config.get("background_depth_raw"),
        config.get("default_background_depth_m"),
        config.get("background_depth_m"),
        default=depth_median,
    )
    object_depth_m = _positive_float(geometry.get("mean_object_depth_m"), depth_median)
    geometry_height_cm = _positive_float(geometry.get("mean_height_m")) * 100.0
    table_depth_cm = _positive_float(
        config.get("table_depth_cm"),
        background_raw * 100.0,
        default=background_raw * 100.0,
    )
    object_depth_cm = _positive_float(
        config.get("object_depth_cm"),
        object_depth_m * 100.0,
        default=object_depth_m * 100.0,
    )

    return {
        "depth_mean": depth_mean,
        "depth_median": depth_median,
        "depth_std": depth_std,
        "depth_min": depth_min,
        "depth_max": depth_max,
        "depth_range": depth_range,
        "depth_p10": depth_p10,
        "depth_p90": depth_p90,
        "depth_iqr": depth_iqr,
        "depth_valid_ratio": depth_valid_ratio,
        "background_depth_raw": background_raw,
        "object_to_background_depth_ratio": _safe_div(depth_median, background_raw, default=0.0),
        "table_depth_cm": table_depth_cm,
        "object_depth_cm": object_depth_cm,
        "geometry_height_cm": geometry_height_cm,
    }


def _resolve_density_g_cm3(class_name: str, material: str, config: dict[str, Any]) -> float:
    direct_g = config.get("density_priors_g_cm3")
    if isinstance(direct_g, dict):
        value = _lookup_mapping(direct_g, class_name, material)
        if value is not None:
            return float(value)

    kg_sources = []
    if isinstance(config.get("density_priors_kg_m3"), dict):
        kg_sources.append(config["density_priors_kg_m3"])
    if isinstance(config.get("densities"), dict):
        kg_sources.append(config["densities"])
    kg_sources.append(DEFAULT_DENSITIES_KG_M3)
    for mapping in kg_sources:
        value = _lookup_mapping(mapping, class_name, material)
        if value is not None:
            return float(value) / 1000.0
    return float(config.get("default_density_g_cm3", 0.1))


def _resolve_hollow_prior(class_name: str, material: str, config: dict[str, Any]) -> float:
    configured = config.get("hollow_priors")
    if isinstance(configured, dict):
        value = _lookup_mapping(configured, class_name, material)
        if value is not None:
            return float(value)
    value = _lookup_mapping(DEFAULT_HOLLOW_PRIORS, class_name, material)
    return float(value) if value is not None else float(config.get("default_hollow_prior", 0.5))


def _lookup_mapping(mapping: dict[Any, Any], *keys: str) -> Any | None:
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for key in keys:
        normalized = str(key).strip().lower()
        if normalized in lowered:
            return lowered[normalized]
    for key in keys:
        normalized = str(key).strip().lower()
        for token, value in lowered.items():
            if token and token in normalized:
                return value
    return None


def _stat(
    stats: dict[str, Any],
    short_key: str,
    long_key: str,
    *,
    default: float = 0.0,
) -> float:
    return _finite_float(stats.get(long_key), stats.get(short_key), default=default)


def _positive_float(*values: Any, default: float = 0.0) -> float:
    for value in values:
        number = _finite_float(value, default=float("nan"))
        if number > 0:
            return number
    return float(default)


def _finite_float(*values: Any, default: float = 0.0) -> float:
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return float(default)


def _safe_div(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    if denominator == 0 or not math.isfinite(float(denominator)):
        return float(default)
    value = float(numerator) / float(denominator)
    return value if math.isfinite(value) else float(default)
