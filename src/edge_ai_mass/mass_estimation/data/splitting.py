"""Group-aware train/validation/test splitting for small mass datasets."""

from __future__ import annotations

from collections import Counter
from typing import Any

from edge_ai_mass.mass_estimation.config import (
    MassEstimationConfig,
    MassEstimationConfigError,
)
from edge_ai_mass.mass_estimation.io import read_table, write_json, write_table


def split_feature_dataset(config: MassEstimationConfig) -> dict[str, Any]:
    """Split feature rows while preventing the same object crossing splits."""

    split_cfg = config.payload["split"]
    schema_path = config.processed_dir / "feature_schema.json"
    if not schema_path.is_file():
        raise FileNotFoundError(
            f"Feature schema not found: {schema_path}. Run stage features first."
        )
    feature_path = _feature_path(config)
    df = read_table(feature_path)

    group_column = str(split_cfg.get("group_column", "object_id"))
    if group_column not in df.columns:
        if bool(split_cfg.get("unsafe_random_fallback", False)):
            group_column = "sample_id"
        else:
            raise MassEstimationConfigError(
                f"Group column '{group_column}' is missing. Set split.group_column to an "
                "available identifier or explicitly enable unsafe_random_fallback."
            )
    if "class_name" not in df.columns:
        raise MassEstimationConfigError("class_name is required for split reporting")

    split_column = str(split_cfg.get("split_column", "split"))
    if bool(split_cfg.get("use_existing", False)):
        if split_column not in df.columns:
            raise MassEstimationConfigError(
                f"split.use_existing=true but column '{split_column}' is missing"
            )
        split_df = df.copy()
        split_df["split"] = split_df[split_column].map(_normalize_split_name)
        invalid = sorted(set(split_df["split"].dropna()) - {"train", "val", "test"})
        if invalid:
            raise MassEstimationConfigError(
                f"Existing split column contains unsupported values: {invalid}"
            )
        if split_df["split"].isna().any():
            raise MassEstimationConfigError("Existing split column contains missing values")
        strategy = {"existing_split": True, "split_column": split_column, "group_aware": True}
    else:
        assignments, strategy = _split_groups(
            df, group_column=group_column, split_cfg=split_cfg
        )
        split_df = df.copy()
        split_df["split"] = split_df[group_column].map(assignments)
        if split_df["split"].isna().any():
            raise RuntimeError("Internal split error: some rows were not assigned a split")

    outputs: dict[str, str] = {}
    suffix = feature_path.suffix
    for split_name in ("train", "val", "test"):
        subset = split_df[split_df["split"] == split_name].drop(columns=["split"])
        if subset.empty and float(split_cfg.get(split_name, 0.0)) > 0:
            raise MassEstimationConfigError(
                f"Split '{split_name}' is empty. Add more measured samples or adjust ratios."
            )
        output_path = config.processed_dir / f"features_{split_name}{suffix}"
        write_table(subset, output_path)
        outputs[split_name] = str(output_path)

    summary = {
        "group_column": group_column,
        "seed": int(split_cfg.get("seed", 42)),
        "ratios": {
            name: float(split_cfg.get(name, 0.0)) for name in ("train", "val", "test")
        },
        "strategy": strategy,
        "outputs": outputs,
        "rows": {
            name: int((split_df["split"] == name).sum()) for name in ("train", "val", "test")
        },
        "groups": {
            name: int(split_df.loc[split_df["split"] == name, group_column].nunique())
            for name in ("train", "val", "test")
        },
        "class_distribution": {
            name: split_df.loc[split_df["split"] == name, "class_name"]
            .value_counts()
            .sort_index()
            .to_dict()
            for name in ("train", "val", "test")
        },
        "group_overlap": _group_overlap_report(split_df, group_column),
    }
    write_json(config.processed_dir / "split_summary.json", summary)
    write_json(config.artifacts_dir / "features" / "split_summary.json", summary)
    return summary


def _feature_path(config: MassEstimationConfig):
    for name in ("features_all.parquet", "features_all.csv"):
        path = config.processed_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"features_all.csv/parquet not found under {config.processed_dir}. "
        "Run stage features first."
    )


def _split_groups(df: Any, *, group_column: str, split_cfg: dict[str, Any]):
    from sklearn.model_selection import train_test_split

    seed = int(split_cfg.get("seed", 42))
    ratios = {name: float(split_cfg.get(name, 0.0)) for name in ("train", "val", "test")}
    groups = (
        df.groupby(group_column, dropna=False)["class_name"]
        .agg(lambda values: Counter(values).most_common(1)[0][0])
        .reset_index()
        .rename(columns={"class_name": "group_class"})
    )
    if len(groups) < 2 and (ratios["val"] > 0 or ratios["test"] > 0):
        raise MassEstimationConfigError("At least two groups are required for non-train splits")

    group_ids = groups[group_column].tolist()
    labels = groups["group_class"].tolist()
    temp_ratio = ratios["val"] + ratios["test"]
    strategy = {"group_aware": True, "stratified_train_temp": False, "stratified_val_test": False}

    if temp_ratio == 0:
        return {group_id: "train" for group_id in group_ids}, strategy

    stratify = _stratify_or_none(labels)
    try:
        train_groups, temp_groups, _, temp_labels = train_test_split(
            group_ids,
            labels,
            test_size=temp_ratio,
            random_state=seed,
            stratify=stratify,
        )
        strategy["stratified_train_temp"] = stratify is not None
    except ValueError:
        train_groups, temp_groups, _, temp_labels = train_test_split(
            group_ids,
            labels,
            test_size=temp_ratio,
            random_state=seed,
            stratify=None,
        )

    assignments = {group_id: "train" for group_id in train_groups}
    if ratios["val"] == 0:
        assignments.update({group_id: "test" for group_id in temp_groups})
        return assignments, strategy
    if ratios["test"] == 0:
        assignments.update({group_id: "val" for group_id in temp_groups})
        return assignments, strategy

    val_ratio_within_temp = ratios["val"] / temp_ratio
    stratify_temp = _stratify_or_none(temp_labels)
    try:
        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=1.0 - val_ratio_within_temp,
            random_state=seed + 1,
            stratify=stratify_temp,
        )
        strategy["stratified_val_test"] = stratify_temp is not None
    except ValueError:
        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=1.0 - val_ratio_within_temp,
            random_state=seed + 1,
            stratify=None,
        )
    assignments.update({group_id: "val" for group_id in val_groups})
    assignments.update({group_id: "test" for group_id in test_groups})
    return assignments, strategy


def _stratify_or_none(labels: list[Any]) -> list[Any] | None:
    counts = Counter(labels)
    if len(counts) < 2 or min(counts.values()) < 2:
        return None
    return labels


def _normalize_split_name(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    aliases = {"validation": "val", "valid": "val", "dev": "val", "testing": "test"}
    return aliases.get(normalized, normalized)


def _group_overlap_report(df: Any, group_column: str) -> dict[str, list[str]]:
    split_groups = {
        name: set(df.loc[df["split"] == name, group_column].astype(str))
        for name in ("train", "val", "test")
    }
    report = {
        "train_val": sorted(split_groups["train"] & split_groups["val"]),
        "train_test": sorted(split_groups["train"] & split_groups["test"]),
        "val_test": sorted(split_groups["val"] & split_groups["test"]),
    }
    overlaps = [value for values in report.values() for value in values]
    if overlaps:
        raise RuntimeError(f"Group leakage detected across splits: {sorted(set(overlaps))}")
    return report
