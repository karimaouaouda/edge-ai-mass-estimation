"""Feature schema validation for mass-estimation models."""

from __future__ import annotations

import re


LEAKAGE_COLUMN_PATTERNS = (
    re.compile(r"(^|_)real_?mass(_|$)", re.IGNORECASE),
    re.compile(r"(^|_)measured_?mass(_|$)", re.IGNORECASE),
    re.compile(r"(^|_)ground_?truth_?mass(_|$)", re.IGNORECASE),
    re.compile(r"(^|_)target(_|$)", re.IGNORECASE),
    re.compile(r"(^|_)label(_|$)", re.IGNORECASE),
    re.compile(r"(^|_)correction(_|$)", re.IGNORECASE),
    re.compile(r"(^|_)residual(_|$)", re.IGNORECASE),
)


ALLOWED_PHYSICS_COLUMNS = {"mass_base_g", "base_mass_g"}


def validate_no_target_leakage(feature_columns: list[str]) -> None:
    """Reject target, residual, or measured-mass columns from model inputs."""

    leaked = []
    for column in feature_columns:
        normalized = column.strip().lower()
        if normalized in ALLOWED_PHYSICS_COLUMNS:
            continue
        if any(pattern.search(normalized) for pattern in LEAKAGE_COLUMN_PATTERNS):
            leaked.append(column)
    if leaked:
        raise ValueError(
            "Target leakage detected in feature columns. Remove these inputs: "
            + ", ".join(sorted(leaked))
        )
