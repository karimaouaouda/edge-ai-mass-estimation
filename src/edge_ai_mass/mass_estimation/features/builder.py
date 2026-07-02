"""Build leakage-safe tabular features for residual mass estimation."""

from __future__ import annotations

from typing import Any

from edge_ai_mass.mass_estimation.config import (
    MassEstimationConfig,
    MassEstimationConfigError,
)
from edge_ai_mass.mass_estimation.features.schema import validate_no_target_leakage
from edge_ai_mass.mass_estimation.io import (
    dataframe_fingerprint,
    read_table,
    write_json,
    write_table,
)


IDENTIFIER_COLUMNS = {
    "sample_id",
    "object_id",
    "image_id",
    "annotation_id",
    "source_image",
    "image_path",
    "mask_path",
    "depth_path",
}
TRACE_COLUMNS = {
    "source_dataset",
    "split",
    "calibration_id",
    "feature_version",
    "mass_label_type",
    "sample_weight",
    "is_outlier",
    "outlier_reason",
    "mass_metadata_category",
    "mass_metadata_state",
    "mass_metadata_notes",
}
TARGET_COLUMNS = {"real_mass_g"}
DERIVED_TARGET_COLUMNS = {
    "correction_g",
    "residual_g",
    "residual_target_g",
    "predicted_correction_g",
    "mass_pred_g",
    "predicted_mass_g",
    "baseline_error_g",
    "abs_baseline_error_g",
    "baseline_relative_error",
}


def build_feature_dataset(config: MassEstimationConfig) -> dict[str, Any]:
    """Create the full feature table and durable feature schema artifacts."""

    features_cfg = config.payload["features"]
    input_path = config.processed_dir / "preprocessed_objects.csv"
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Preprocessed mass dataset not found: {input_path}. Run stage preprocess first."
        )
    df = read_table(input_path)
    _require_columns(df, ["sample_id", "object_id", "class_name", "real_mass_g"])

    mass_base_column = str(features_cfg.get("mass_base_column", "mass_base_g"))
    if bool(features_cfg.get("require_mass_base", True)):
        _require_columns(df, [mass_base_column])
        if bool(df[mass_base_column].isna().any()):
            bad_ids = df.loc[df[mass_base_column].isna(), "sample_id"].head(5).tolist()
            raise MassEstimationConfigError(
                f"{mass_base_column} contains missing values for sample_id(s): {bad_ids}"
            )

    numeric_columns, ignored_numeric = _resolve_numeric_columns(df, features_cfg)
    categorical_columns, ignored_categorical = _resolve_categorical_columns(df, features_cfg)
    feature_columns = _unique([*categorical_columns, *numeric_columns])
    validate_no_target_leakage(feature_columns)

    keep_columns = _unique(
        [
            "sample_id",
            "object_id",
            "class_name",
            "material",
            "real_mass_g",
            mass_base_column,
            *sorted(TRACE_COLUMNS),
            *feature_columns,
        ]
    )
    existing_keep = [column for column in keep_columns if column in df.columns]
    feature_df = df[existing_keep].copy()

    output_format = str(features_cfg.get("output_format", "csv")).lower()
    suffix = ".parquet" if output_format == "parquet" else ".csv"
    output_path = config.processed_dir / f"features_all{suffix}"
    write_table(feature_df, output_path)

    schema = {
        "schema_version": 1,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "target_column": "real_mass_g",
        "mass_base_column": mass_base_column,
        "identifier_columns": [column for column in IDENTIFIER_COLUMNS if column in feature_df],
        "trace_columns": [column for column in TRACE_COLUMNS if column in feature_df],
        "excluded_columns": sorted(_excluded_columns(features_cfg)),
        "ignored_configured_columns": {
            "numeric": ignored_numeric,
            "categorical": ignored_categorical,
        },
        "dtypes": {column: str(dtype) for column, dtype in feature_df.dtypes.items()},
        "data_config_digest": config.data_digest,
        "dataset_fingerprint": dataframe_fingerprint(feature_df),
    }
    summary = {
        "rows": int(len(feature_df)),
        "columns": list(feature_df.columns),
        "feature_count": len(feature_columns),
        "missing_values": {
            column: int(value)
            for column, value in feature_df.isna().sum().sort_index().items()
            if int(value) > 0
        },
        "class_distribution": feature_df["class_name"].value_counts().sort_index().to_dict(),
        "mass_g": _describe(feature_df["real_mass_g"]),
        "mass_base_g": (
            _describe(feature_df[mass_base_column])
            if mass_base_column in feature_df
            else None
        ),
    }

    for root in (config.processed_dir, config.artifacts_dir / "features"):
        write_json(root / "feature_schema.json", schema)
        write_json(root / "feature_summary.json", summary)

    return {
        "features": str(output_path),
        "schema": str(config.processed_dir / "feature_schema.json"),
        "summary": str(config.processed_dir / "feature_summary.json"),
        "dataset_fingerprint": schema["dataset_fingerprint"],
        "feature_columns": feature_columns,
    }


def _resolve_numeric_columns(df: Any, features_cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    configured = [str(column) for column in features_cfg.get("numeric_columns", [])]
    strict = bool(features_cfg.get("strict_columns", False))
    ignored = [column for column in configured if column not in df.columns]
    if strict and ignored:
        raise MassEstimationConfigError(f"Configured numeric columns missing: {ignored}")
    selected = [column for column in configured if column in df.columns]

    if bool(features_cfg.get("auto_include_numeric", True)):
        pd = __import__("pandas")
        excluded = _excluded_columns(features_cfg)
        for column in df.columns:
            if column in selected or column in excluded:
                continue
            if pd.api.types.is_numeric_dtype(df[column]):
                selected.append(column)
    return selected, ignored


def _resolve_categorical_columns(
    df: Any, features_cfg: dict[str, Any]
) -> tuple[list[str], list[str]]:
    configured = [str(column) for column in features_cfg.get("categorical_columns", [])]
    strict = bool(features_cfg.get("strict_columns", False))
    ignored = [column for column in configured if column not in df.columns]
    if strict and ignored:
        raise MassEstimationConfigError(f"Configured categorical columns missing: {ignored}")
    return [column for column in configured if column in df.columns], ignored


def _excluded_columns(features_cfg: dict[str, Any]) -> set[str]:
    configured = {str(column) for column in features_cfg.get("exclude_columns", [])}
    return (
        IDENTIFIER_COLUMNS
        | TRACE_COLUMNS
        | TARGET_COLUMNS
        | DERIVED_TARGET_COLUMNS
        | configured
    )


def _require_columns(df: Any, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise MassEstimationConfigError(f"Feature dataset missing required columns: {missing}")


def _describe(series: Any) -> dict[str, float]:
    description = series.describe()
    return {
        key: float(value)
        for key, value in description.items()
        if key in {"count", "mean", "std", "min", "25%", "50%", "75%", "max"}
    }


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
