"""Create one object-level mass-feature row from a mask and collection metadata.

This script writes a CSV row compatible with the governed mass-estimation
pipeline in `configs/mass_estimation/residual_pipeline.yaml`.

Examples:
    python scripts/mass/create_mass_feature_row.py \
        --config scripts/mass/feature_rows.yaml \
        --profile single_example
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.mass.utils import (
    append_csv_row,
    deep_merge,
    density_for_class,
    estimate_thickness_cm,
    load_density_priors,
    load_mask,
    load_profiled_config,
    mask_geometry,
    normalize_depth_value,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create tabular mass-estimation feature rows from YAML config."
    )
    parser.add_argument(
        "--config",
        default="scripts/mass/feature_rows.yaml",
        help="YAML script config with output, defaults, priors, and row data.",
    )
    parser.add_argument("--profile", default=None, help="Profile name inside the YAML config")
    args = parser.parse_args()

    config, selected_profile = load_profiled_config(args.config, args.profile)
    output, rows = write_feature_rows_from_config(config)
    profile_note = f" profile={selected_profile}" if selected_profile else ""
    print(f"Appended {len(rows)} mass feature row(s) to {output}{profile_note}")


def write_feature_rows_from_config(config: dict[str, object]) -> tuple[Path, list[dict[str, object]]]:
    """Build and append all rows described by a feature-row config."""

    if "output" not in config:
        raise ValueError("Feature-row config requires output")
    output = Path(str(config["output"]))
    rows = build_feature_rows_from_config(config)
    for row in rows:
        append_csv_row(output, row)
    return output, rows


def build_feature_rows_from_config(config: dict[str, object]) -> list[dict[str, object]]:
    """Build feature rows from config without writing them to disk."""

    defaults = config.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("defaults must be a mapping")
    raw_rows = config.get("rows")
    if raw_rows is None and config.get("row") is not None:
        raw_rows = [config["row"]]
    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("Feature-row config requires row or non-empty rows")

    density_priors_path = config.get("density_priors")
    class_priors = config.get("class_density_priors_g_cm3", {})
    if not isinstance(class_priors, dict):
        raise ValueError("class_density_priors_g_cm3 must be a mapping")
    class_density_prior = [
        f"{class_name}={density}" for class_name, density in class_priors.items()
    ]

    result = []
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            raise ValueError("Each configured row must be a mapping")
        item = deep_merge(defaults, raw_row)
        result.append(
            build_feature_row(
                image_id=str(_required(item, "image_id")),
                object_id=_optional_string(item.get("object_id")),
                sample_id=_optional_string(item.get("sample_id")),
                class_name=str(_required(item, "class_name")),
                material=_optional_string(item.get("material")),
                mask_path=str(_required(item, "mask")),
                pixel_per_cm=float(_required(item, "pixel_per_cm")),
                background_default_depth=float(_required(item, "background_default_depth")),
                camera_height=float(_required(item, "camera_height")),
                depth_unit=str(item.get("depth_unit", "cm")),
                thickness_mode=str(item.get("thickness_mode", "camera_minus_background")),
                min_thickness_cm=float(item.get("min_thickness_cm", 0.0)),
                density_priors_path=(
                    str(density_priors_path) if density_priors_path not in (None, "") else None
                ),
                class_density_prior=class_density_prior,
                density_prior_g_cm3=(
                    float(item["density_prior_g_cm3"])
                    if item.get("density_prior_g_cm3") is not None
                    else None
                ),
                real_mass_g=float(_required(item, "real_mass_g")),
            )
        )
    return result


def build_feature_row(
    *,
    image_id: str,
    object_id: str | None,
    sample_id: str | None,
    class_name: str,
    material: str | None,
    mask_path: str | Path,
    pixel_per_cm: float,
    background_default_depth: float,
    camera_height: float,
    depth_unit: str,
    thickness_mode: str,
    min_thickness_cm: float,
    density_priors_path: str | Path | None,
    class_density_prior: list[str],
    density_prior_g_cm3: float | None,
    real_mass_g: float,
) -> dict[str, float | int | str]:
    mask = load_mask(mask_path)
    geometry = mask_geometry(mask, pixel_per_cm=pixel_per_cm)
    priors = load_density_priors(
        priors_path=density_priors_path,
        inline_pairs=class_density_prior,
    )
    density_g_cm3 = density_for_class(
        class_name,
        priors_g_cm3=priors,
        fallback_g_cm3=density_prior_g_cm3,
    )
    background_cm = normalize_depth_value(background_default_depth, unit=depth_unit)
    camera_height_cm = normalize_depth_value(camera_height, unit=depth_unit)
    thickness_cm = estimate_thickness_cm(
        camera_height_cm=camera_height_cm,
        background_default_depth_cm=background_cm,
        mode=thickness_mode,
        min_thickness_cm=min_thickness_cm,
    )
    volume_cm3 = geometry.area_cm2 * thickness_cm
    mass_base_g = volume_cm3 * density_g_cm3
    correction_g = float(real_mass_g) - mass_base_g

    sample = sample_id or image_id
    object_identifier = object_id or sample
    estimated_volume_m3 = volume_cm3 * 1e-6
    projected_area_m2 = geometry.area_cm2 * 1e-4
    effective_density_kg_m3 = density_g_cm3 * 1000.0

    return {
        "sample_id": sample,
        "image_id": image_id,
        "object_id": object_identifier,
        "source_image": image_id,
        "class_name": class_name,
        "material": material or class_name,
        "mask_path": str(mask_path),
        "pixel_per_cm": float(pixel_per_cm),
        "background_default_depth_cm": background_cm,
        "camera_height_cm": camera_height_cm,
        "estimated_thickness_cm": thickness_cm,
        "estimated_thickness_m": thickness_cm / 100.0,
        "area_cm2": geometry.area_cm2,
        "bbox_width_cm": geometry.bbox_width_cm,
        "bbox_height_cm": geometry.bbox_height_cm,
        "bbox_area_cm2": geometry.bbox_area_cm2,
        "volume_cm3": volume_cm3,
        "estimated_volume_m3": estimated_volume_m3,
        "density_prior_g_cm3": density_g_cm3,
        "effective_density_kg_m3": effective_density_kg_m3,
        "base_mass_g": mass_base_g,
        "mass_base_g": mass_base_g,
        "real_mass_g": float(real_mass_g),
        "correction_g": correction_g,
        "residual_target_g": correction_g,
        "bbox_x_px": geometry.bbox_x_px,
        "bbox_y_px": geometry.bbox_y_px,
        "bbox_width_px": geometry.bbox_width_px,
        "bbox_height_px": geometry.bbox_height_px,
        "bbox_area_px": geometry.bbox_area_px,
        "mask_area_px": geometry.mask_area_px,
        "contour_area_px": geometry.contour_area_px,
        "perimeter_px": geometry.perimeter_px,
        "convex_hull_area_px": geometry.convex_hull_area_px,
        "projected_area_m2": projected_area_m2,
        "aspect_ratio": geometry.aspect_ratio,
        "solidity": geometry.solidity,
        "extent": geometry.extent,
        "compactness": geometry.compactness,
        "depth_mean_m": background_cm / 100.0,
        "depth_median_m": background_cm / 100.0,
        "depth_std_m": 0.0,
        "depth_min_m": background_cm / 100.0,
        "depth_max_m": background_cm / 100.0,
        "depth_valid_ratio": 1.0,
    }


def _required(mapping: dict[str, object], key: str) -> object:
    value = mapping.get(key)
    if value in (None, ""):
        raise ValueError(f"Configured feature row requires {key}")
    return value


def _optional_string(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


if __name__ == "__main__":
    main()
