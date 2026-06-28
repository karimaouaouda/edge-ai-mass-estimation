"""Preprocess measured object-level data for residual mass estimation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from edge_ai_mass.mass_estimation.config import (
    MassEstimationConfig,
    MassEstimationConfigError,
)
from edge_ai_mass.mass_estimation.io import (
    dataframe_fingerprint,
    file_sha256,
    read_table,
    write_json,
    write_table,
)
from edge_ai_mass.modules.mass.density_estimator import MATERIAL_DENSITIES
from edge_ai_mass.modules.material.class_mapping import ClassMaterialMapper


def preprocess_mass_dataset(config: MassEstimationConfig) -> dict[str, Any]:
    """Validate measured samples and produce a clean object-level table.

    The stage keeps measured mass as a target-only column. Physics quantities
    such as base mass are computed from volume and density priors, never from
    the measured label.
    """

    data_cfg = config.payload["data"]
    source_path = config.path(data_cfg["measurements"])
    if not source_path.is_file():
        raise FileNotFoundError(f"Mass measurement file not found: {source_path}")

    df = read_table(source_path)
    original_columns = list(df.columns)
    df = _standardize_columns(df, data_cfg.get("columns", {}))
    required = ["sample_id", "class_name", "real_mass_g"]
    _require_columns(df, required, context="mass measurement input")

    if df["sample_id"].isna().any():
        raise MassEstimationConfigError("sample_id contains missing values")
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    if (df["sample_id"] == "").any():
        raise MassEstimationConfigError("sample_id contains empty identifiers")
    if df["sample_id"].duplicated().any():
        duplicate = df.loc[df["sample_id"].duplicated(), "sample_id"].iloc[0]
        raise MassEstimationConfigError(f"Duplicate sample_id in mass data: {duplicate}")

    if "object_id" not in df.columns:
        df["object_id"] = df["sample_id"]
        generated_object_ids = True
    else:
        df["object_id"] = df["object_id"].fillna(df["sample_id"]).astype(str).str.strip()
        generated_object_ids = False

    df["class_name"] = df["class_name"].astype(str).str.strip()
    if (df["class_name"] == "").any():
        raise MassEstimationConfigError("class_name contains empty values")

    pd = __import__("pandas")
    df["real_mass_g"] = pd.to_numeric(df["real_mass_g"], errors="coerce")
    bad_mass = df["real_mass_g"].isna() | (df["real_mass_g"] <= 0)
    if bool(bad_mass.any()):
        bad_ids = df.loc[bad_mass, "sample_id"].head(5).tolist()
        raise MassEstimationConfigError(
            f"real_mass_g must be positive numeric grams; invalid sample_id(s): {bad_ids}"
        )

    material_cfg = data_cfg.get("material", {})
    mapper = ClassMaterialMapper.from_config(material_cfg)
    if "material" not in df.columns:
        df["material"] = df["class_name"].map(mapper.material_for)
    else:
        df["material"] = df["material"].fillna("").astype(str).str.strip()
        missing = df["material"] == ""
        if bool(missing.any()):
            df.loc[missing, "material"] = df.loc[missing, "class_name"].map(
                mapper.material_for
            )

    densities = {
        **MATERIAL_DENSITIES,
        **{
            str(key): float(value)
            for key, value in material_cfg.get("density_priors_kg_m3", {}).items()
        },
    }
    if "effective_density_kg_m3" not in df.columns:
        df["effective_density_kg_m3"] = df["material"].map(
            lambda value: densities.get(str(value), densities["other"])
        )
    else:
        df["effective_density_kg_m3"] = pd.to_numeric(
            df["effective_density_kg_m3"], errors="coerce"
        )
        missing_density = df["effective_density_kg_m3"].isna()
        if bool(missing_density.any()):
            df.loc[missing_density, "effective_density_kg_m3"] = df.loc[
                missing_density, "material"
            ].map(lambda value: densities.get(str(value), densities["other"]))

    volume_column = data_cfg.get("columns", {}).get("volume", "estimated_volume_m3")
    if volume_column in df.columns and "estimated_volume_m3" not in df.columns:
        df["estimated_volume_m3"] = df[volume_column]
    if "estimated_volume_m3" in df.columns:
        df["estimated_volume_m3"] = pd.to_numeric(
            df["estimated_volume_m3"], errors="coerce"
        )
        df["estimated_volume_m3"] = _convert_volume_to_m3(
            df["estimated_volume_m3"],
            unit=str(data_cfg.get("columns", {}).get("volume_unit", "m3")),
        )

    if {"estimated_volume_m3", "effective_density_kg_m3"} <= set(df.columns):
        df["mass_base_g"] = (
            df["estimated_volume_m3"] * df["effective_density_kg_m3"] * 1000.0
        )

    optional_inputs = _inspect_optional_inputs(config, data_cfg.get("optional_inputs", {}))
    output_path = config.processed_dir / "preprocessed_objects.csv"
    write_table(df, output_path)

    readiness = {
        "has_volume": "estimated_volume_m3" in df.columns
        and not bool(df["estimated_volume_m3"].isna().all()),
        "has_density": "effective_density_kg_m3" in df.columns
        and not bool(df["effective_density_kg_m3"].isna().all()),
        "has_mass_base": "mass_base_g" in df.columns
        and not bool(df["mass_base_g"].isna().all()),
        "has_group_ids": "object_id" in df.columns,
    }
    report = {
        "source": str(source_path),
        "source_sha256": file_sha256(source_path),
        "output": str(output_path),
        "rows": int(len(df)),
        "classes": df["class_name"].value_counts().sort_index().to_dict(),
        "materials": df["material"].value_counts().sort_index().to_dict(),
        "original_columns": original_columns,
        "processed_columns": list(df.columns),
        "generated_object_ids": generated_object_ids,
        "optional_inputs": optional_inputs,
        "feature_readiness": readiness,
        "dataset_fingerprint": dataframe_fingerprint(df),
        "data_config_digest": config.data_digest,
    }
    write_json(config.artifacts_dir / "preprocessing" / "preprocessing_report.json", report)
    write_json(config.processed_dir / "preprocessing_report.json", report)
    return report


def _standardize_columns(df: Any, mapping: dict[str, str]) -> Any:
    rename: dict[str, str] = {}
    for canonical, source in mapping.items():
        if canonical == "volume_unit":
            continue
        target = {
            "volume": "estimated_volume_m3",
            "density": "effective_density_kg_m3",
        }.get(canonical, canonical)
        source_name = str(source)
        if source_name in df.columns and target not in df.columns:
            rename[source_name] = target
    return df.rename(columns=rename)


def _require_columns(df: Any, columns: list[str], *, context: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise MassEstimationConfigError(f"{context} missing required columns: {missing}")


def _convert_volume_to_m3(series: Any, *, unit: str) -> Any:
    normalized = unit.strip().lower()
    if normalized in {"m3", "m^3", "cubic_meter", "cubic_metre"}:
        return series
    if normalized in {"cm3", "cc", "ml", "milliliter", "millilitre"}:
        return series * 1e-6
    if normalized in {"l", "liter", "litre"}:
        return series * 1e-3
    raise MassEstimationConfigError(f"Unsupported volume unit: {unit}")


def _inspect_optional_inputs(
    config: MassEstimationConfig,
    optional_inputs: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for name, value in optional_inputs.items():
        if value in (None, ""):
            report[str(name)] = {"configured": False, "exists": False, "path": None}
            continue
        path = config.path(str(value))
        report[str(name)] = {
            "configured": True,
            "exists": path.exists(),
            "path": str(path),
            "kind": "directory" if path.is_dir() else "file" if path.is_file() else "missing",
        }
    return report
