"""COCO segmentation ingestion and deterministic YOLO dataset materialization."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from edge_ai_mass.training.config import TrainingConfig
from edge_ai_mass.training.taxonomy import resolve_category

logger = logging.getLogger(__name__)


@dataclass
class Annotation:
    class_id: int
    polygon: list[float] | None
    bbox: list[float] | None


@dataclass
class ImageRecord:
    source: str
    source_id: int
    source_name: str
    path: Path
    width: int
    height: int
    annotations: list[Annotation] = field(default_factory=list)
    digest: str = ""
    split: str = ""


class DatasetBuildError(RuntimeError):
    """Raised when dataset quality or source contracts are violated."""


def _emit_debug(debug_cfg: dict[str, Any], event: str, **details: Any) -> None:
    """Print one structured diagnostic line that remains readable on Kaggle."""
    if not debug_cfg.get("enabled", False):
        return
    payload = {"event": event, **details}
    print(f"[dataset-debug] {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def build_yolo_dataset(config: TrainingConfig) -> dict[str, Any]:
    """Merge configured COCO sources into one deterministic YOLO dataset."""
    data_cfg = config.payload["data"]
    debug_cfg = data_cfg.get("debug", {})
    # Rebuild whenever the stage is invoked. DVC decides whether invocation is
    # necessary; silently short-circuiting on config alone would miss changed
    # images or annotations and break data lineage.

    classes = [str(label) for label in data_cfg["classes"]]
    class_to_id = {label: index for index, label in enumerate(classes)}
    records: list[ImageRecord] = []
    source_reports: dict[str, Any] = {}
    input_versions: dict[str, Any] = {}
    _emit_debug(
        debug_cfg,
        "build.before",
        project_root=config.project_root,
        output_dir=config.dataset_dir,
        classes=classes,
        configured_sources=[item.get("name") for item in data_cfg["sources"]],
    )

    for source_cfg in data_cfg["sources"]:
        if not source_cfg.get("enabled", True):
            _emit_debug(debug_cfg, "source.skipped", source=source_cfg.get("name"))
            continue
        _emit_debug(debug_cfg, "source.before", source=source_cfg.get("name"))
        source_records, report, version = _read_coco_source(config, source_cfg, class_to_id)
        records.extend(source_records)
        source_reports[str(source_cfg["name"])] = report
        input_versions[str(source_cfg["name"])] = version
        _emit_debug(
            debug_cfg,
            "source.after",
            source=source_cfg.get("name"),
            report=report,
            version=version,
        )

    quality_cfg = data_cfg.get("quality", {})
    if len(records) < int(quality_cfg.get("min_images", 1)):
        raise DatasetBuildError(
            f"Only {len(records)} valid images found; expected at least "
            f"{quality_cfg.get('min_images', 1)}"
        )
    instance_count = sum(len(record.annotations) for record in records)
    if instance_count < int(quality_cfg.get("min_instances", 1)):
        raise DatasetBuildError(
            f"Only {instance_count} valid instances found; expected at least "
            f"{quality_cfg.get('min_instances', 1)}"
        )

    _emit_debug(
        debug_cfg,
        "merge.after",
        total_images=len(records),
        total_instances=instance_count,
    )
    _assign_splits(records, data_cfg.get("split", {}))
    manifest = _materialize(config, records, classes, source_reports, input_versions)
    _emit_debug(
        debug_cfg,
        "build.after",
        dataset_fingerprint=manifest.get("dataset_fingerprint"),
        total_images=manifest.get("total_images"),
        total_instances=manifest.get("total_instances"),
    )
    return manifest


def inspect_sources(config: TrainingConfig) -> list[dict[str, Any]]:
    """Return a no-write source readiness report for CLI and notebooks."""
    result = []
    for source in config.payload["data"]["sources"]:
        annotations = config.path(source["annotations"])
        configured_images = config.path(source["images"])
        images = _resolve_source_images_root(
            config,
            source,
            debug_cfg=config.payload["data"].get("debug", {}),
        )
        result.append(
            {
                "name": source["name"],
                "enabled": bool(source.get("enabled", True)),
                "required": bool(source.get("required", True)),
                "annotations": str(annotations),
                "annotations_exist": annotations.is_file(),
                "images_configured": str(configured_images),
                "images": str(images),
                "images_exist": images.is_dir(),
            }
        )
    return result


def _read_coco_source(
    config: TrainingConfig,
    source_cfg: dict[str, Any],
    class_to_id: dict[str, int],
) -> tuple[list[ImageRecord], dict[str, Any], dict[str, Any]]:
    name = str(source_cfg["name"]).strip().lower()
    debug_cfg = config.payload["data"].get("debug", {})
    annotations_path = config.path(source_cfg["annotations"])
    configured_images_root = config.path(source_cfg["images"])
    _emit_debug(
        debug_cfg,
        "source.paths.before",
        source=name,
        annotations=annotations_path,
        annotations_exists=annotations_path.is_file(),
        images_configured=configured_images_root,
        images_configured_exists=configured_images_root.is_dir(),
    )
    images_root = _resolve_source_images_root(config, source_cfg, debug_cfg=debug_cfg)
    required = bool(source_cfg.get("required", True))
    _emit_debug(
        debug_cfg,
        "source.paths.after",
        source=name,
        images_resolved=images_root,
        images_resolved_exists=images_root.is_dir(),
    )
    if not annotations_path.is_file() or not images_root.is_dir():
        message = (
            f"Dataset source '{name}' is unavailable: annotations={annotations_path} "
            f"images={images_root}"
        )
        if required:
            raise FileNotFoundError(message)
        logger.warning("%s; optional source skipped", message)
        return [], {"skipped": True, "reason": message}, {}

    _emit_debug(debug_cfg, "coco.load.before", source=name, path=annotations_path)
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("images"), list) or not isinstance(
        payload.get("annotations"), list
    ):
        raise DatasetBuildError(f"COCO file has no images/annotations arrays: {annotations_path}")
    categories = {int(item["id"]): item for item in payload.get("categories", [])}
    _emit_debug(
        debug_cfg,
        "coco.load.after",
        source=name,
        images=len(payload["images"]),
        annotations=len(payload["annotations"]),
        categories=list(categories.values()),
        first_images=payload["images"][:3],
    )
    if source_cfg.get("require_category_names_in_data_classes", False):
        # RealWaste raw directory names describe the source dataset taxonomy,
        # not this model's targets. Requiring canonical COCO category names
        # prevents an old one-class `trash` file from being trained silently.
        category_names = {str(item.get("name", "")).strip() for item in categories.values()}
        invalid_names = sorted(name for name in category_names if name not in class_to_id)
        _emit_debug(
            debug_cfg,
            "categories.validate.before",
            source=name,
            category_names=sorted(category_names),
            allowed_names=sorted(class_to_id),
            invalid_names=invalid_names,
        )
        if not category_names or invalid_names:
            raise DatasetBuildError(
                f"Source '{name}' COCO categories must be names from data.classes; "
                f"invalid={invalid_names or ['<no categories>']}"
            )
        _emit_debug(debug_cfg, "categories.validate.after", source=name, status="passed")
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    resolver = str(source_cfg.get("resolver", "direct"))
    explicit_mapping = source_cfg.get("category_mapping", {})
    image_label_field = source_cfg.get("category_from_image_field")
    unknown_policy = str(source_cfg.get("unknown_category", "error"))
    allow_bbox = bool(source_cfg.get("allow_bbox_fallback", False))
    task = config.payload["model"].get("task", "segment")
    multipart_policy = str(config.payload["data"].get("multipart_policy", "largest"))
    file_name_fields = source_cfg.get("file_name_fields", ["file_name"])
    strip_path_prefixes = source_cfg.get("strip_path_prefixes", [])
    # A recursive index is built once per source, not once per annotation. It
    # lets a COCO JSON reference a flattened filename while raw RealWaste stays
    # organized under its original class directories.
    _emit_debug(
        debug_cfg,
        "image_index.before",
        source=name,
        root=images_root,
        enabled=bool(source_cfg.get("recursive_basename_fallback", False)),
    )
    basename_index = (
        _build_unique_basename_index(images_root)
        if source_cfg.get("recursive_basename_fallback", False)
        else {}
    )
    _emit_debug(
        debug_cfg,
        "image_index.after",
        source=name,
        root=images_root,
        indexed_names=len(basename_index),
        unique_names=sum(path is not None for path in basename_index.values()),
        ambiguous_names=sum(path is None for path in basename_index.values()),
    )
    report: dict[str, Any] = {
        "images_declared": len(payload["images"]),
        "images_root_configured": str(configured_images_root),
        "images_root_resolved": str(images_root),
        "images": 0,
        "instances": 0,
        "missing_images": 0,
        "missing_image_examples": [],
        "empty_images": 0,
        "invalid_annotations": 0,
        "unmapped_annotations": 0,
        "bbox_fallbacks": 0,
        "multipart_reduced": 0,
        "dimension_mismatches": 0,
        "image_resolution": Counter(),
        "classes": Counter(),
    }
    records: list[ImageRecord] = []

    sample_limit = max(0, int(debug_cfg.get("sample_limit", 25)))
    progress_every = max(0, int(debug_cfg.get("progress_every", 250)))
    total_declared = len(payload["images"])
    for image_index, image_data in enumerate(payload["images"], start=1):
        if image_index <= sample_limit:
            _emit_debug(
                debug_cfg,
                "image.resolve.before",
                source=name,
                index=image_index,
                image_id=image_data.get("id"),
                references={
                    field_name: image_data.get(field_name)
                    for field_name in file_name_fields
                    if image_data.get(field_name)
                },
                images_root=images_root,
            )
        image_path, source_name, resolution = _resolve_image(
            images_root,
            image_data,
            file_name_fields,
            strip_path_prefixes=strip_path_prefixes,
            basename_index=basename_index,
        )
        should_trace = image_index <= sample_limit
        if image_path is None:
            report["missing_images"] += 1
            report["image_resolution"][resolution] += 1
            if len(report["missing_image_examples"]) < 10:
                report["missing_image_examples"].append(
                    {
                        "image_id": image_data.get("id"),
                        "references": {
                            field_name: image_data.get(field_name)
                            for field_name in file_name_fields
                            if image_data.get(field_name)
                        },
                    }
                )
            should_trace = should_trace or report["missing_images"] <= sample_limit
        if should_trace:
            _emit_debug(
                debug_cfg,
                "image.resolve.after",
                source=name,
                index=image_index,
                image_id=image_data.get("id"),
                resolved=image_path,
                resolution=resolution,
                trace=_trace_image_resolution(
                    images_root,
                    image_data,
                    file_name_fields,
                    strip_path_prefixes,
                    basename_index,
                ),
            )
        if progress_every and (
            image_index % progress_every == 0 or image_index == total_declared
        ):
            _emit_debug(
                debug_cfg,
                "source.progress",
                source=name,
                processed=image_index,
                declared=total_declared,
                resolved=image_index - report["missing_images"],
                missing=report["missing_images"],
            )
        if image_path is None:
            continue
        report["image_resolution"][resolution] += 1
        with Image.open(image_path) as image:
            actual_width, actual_height = image.size
        declared_width = int(image_data.get("width") or actual_width)
        declared_height = int(image_data.get("height") or actual_height)
        if (declared_width, declared_height) != (actual_width, actual_height):
            report["dimension_mismatches"] += 1

        record = ImageRecord(
            source=name,
            source_id=int(image_data["id"]),
            source_name=source_name,
            path=image_path,
            width=actual_width,
            height=actual_height,
        )
        for annotation_data in annotations_by_image.get(record.source_id, []):
            category = categories.get(int(annotation_data.get("category_id", -1)), {})
            # COCO category names are authoritative by default. An image-level
            # override exists only for legacy sources and is not used by the new
            # RealWaste contract, whose folder classes are unrelated labels.
            label_override = image_data.get(image_label_field) if image_label_field else None
            target_label = resolve_category(
                category,
                resolver=resolver,
                explicit_mapping=explicit_mapping,
                label_override=None if label_override is None else str(label_override),
            )
            if target_label not in class_to_id:
                report["unmapped_annotations"] += 1
                if unknown_policy == "skip":
                    continue
                if unknown_policy == "mixed_waste" and "mixed_waste" in class_to_id:
                    target_label = "mixed_waste"
                else:
                    source_label = label_override or category.get(
                        "name", annotation_data.get("category_id")
                    )
                    raise DatasetBuildError(
                        f"Source '{name}' label '{source_label}' is not mapped to data.classes"
                    )

            bbox = _valid_bbox(annotation_data.get("bbox"), actual_width, actual_height)
            polygon = None
            if task == "segment":
                polygons = _valid_polygons(
                    annotation_data.get("segmentation"), actual_width, actual_height
                )
                if polygons:
                    polygon = _select_polygon(polygons, multipart_policy)
                    if len(polygons) > 1:
                        report["multipart_reduced"] += len(polygons) - 1
                elif allow_bbox and bbox:
                    polygon = _bbox_polygon(bbox)
                    report["bbox_fallbacks"] += 1
                else:
                    report["invalid_annotations"] += 1
                    continue
            elif bbox is None:
                report["invalid_annotations"] += 1
                continue

            record.annotations.append(
                Annotation(class_id=class_to_id[target_label], polygon=polygon, bbox=bbox)
            )
            report["instances"] += 1
            report["classes"][target_label] += 1

        if not record.annotations:
            report["empty_images"] += 1
            if not config.payload["data"].get("keep_empty_images", True):
                continue
        record.digest = _sha256_file(record.path)
        records.append(record)
        report["images"] += 1

    report["classes"] = dict(sorted(report["classes"].items()))
    report["image_resolution"] = dict(sorted(report["image_resolution"].items()))
    _emit_debug(debug_cfg, "source.read.after", source=name, report=report)
    version = {
        "annotations_sha256": _sha256_file(annotations_path),
        "images_digest": _aggregate_image_digest(records),
    }
    return records, report, version


def _resolve_source_images_root(
    config: TrainingConfig,
    source_cfg: dict[str, Any],
    *,
    debug_cfg: dict[str, Any] | None = None,
) -> Path:
    """Resolve an image root that may point above a dataset's internal tree."""
    debug_cfg = debug_cfg or {}
    configured = config.path(source_cfg["images"])
    suffixes = source_cfg.get("images_root_candidates", ["."])
    _emit_debug(
        debug_cfg,
        "image_root.before",
        source=source_cfg.get("name"),
        configured=configured,
        configured_exists=configured.is_dir(),
        candidates=suffixes,
    )
    for raw_suffix in suffixes:
        suffix = Path(str(raw_suffix).replace("\\", "/"))
        # Candidate suffixes are trusted configuration, but keeping them
        # relative prevents accidental traversal outside the selected mount.
        if suffix.is_absolute() or ".." in suffix.parts:
            raise DatasetBuildError(
                f"Source '{source_cfg['name']}' has unsafe images_root_candidates entry: "
                f"{raw_suffix!r}"
            )
        candidate = configured if str(raw_suffix) in {"", "."} else configured / suffix
        exists = candidate.is_dir()
        entries: list[str] = []
        if exists:
            limit = max(0, int(debug_cfg.get("root_entry_limit", 20)))
            try:
                entries = sorted(path.name for path in candidate.iterdir())[:limit]
            except OSError as exc:
                entries = [f"<list failed: {exc}>"]
        _emit_debug(
            debug_cfg,
            "image_root.candidate",
            source=source_cfg.get("name"),
            suffix=raw_suffix,
            path=candidate,
            exists=exists,
            entries=entries,
        )
        if exists:
            resolved = candidate.resolve()
            _emit_debug(
                debug_cfg,
                "image_root.after",
                source=source_cfg.get("name"),
                selected=resolved,
            )
            return resolved
    _emit_debug(
        debug_cfg,
        "image_root.after",
        source=source_cfg.get("name"),
        selected=configured,
        warning="no candidate directory exists",
    )
    return configured


def _resolve_image(
    images_root: Path,
    image_data: dict[str, Any],
    fields: Iterable[str],
    *,
    strip_path_prefixes: Iterable[str] = (),
    basename_index: dict[str, Path | None] | None = None,
) -> tuple[Path | None, str, str]:
    """Resolve a COCO image without deriving labels from its folder name."""
    names: list[tuple[str, str]] = []
    for field_name in fields:
        value = image_data.get(field_name)
        if value:
            names.append((str(field_name), str(value).replace("\\", "/").strip("/")))

    # Try the JSON path exactly, then remove known dataset-root prefixes such
    # as realwaste-main/RealWaste when images_root already points at RealWaste.
    for field_name, name in names:
        for relative_name, method in _relative_image_candidates(name, strip_path_prefixes):
            relative = Path(relative_name)
            if relative.is_absolute() or ".." in relative.parts:
                continue
            candidate = images_root / relative
            if candidate.is_file():
                return candidate.resolve(), name, f"{method}:{field_name}"

    # A basename fallback is safe only for names that occur exactly once in the
    # raw tree. Duplicate names across original classes remain unresolved.
    for field_name, name in names:
        indexed = (basename_index or {}).get(Path(name).name.casefold())
        if indexed is not None:
            return indexed.resolve(), name, f"unique_basename:{field_name}"

    source_name = names[0][1] if names else str(image_data.get("id", "unknown"))
    return None, source_name, "unresolved"


def _trace_image_resolution(
    images_root: Path,
    image_data: dict[str, Any],
    fields: Iterable[str],
    strip_path_prefixes: Iterable[str],
    basename_index: dict[str, Path | None],
) -> dict[str, Any]:
    """Describe every attempted image path for a bounded debug sample."""
    references: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    basename_lookups: list[dict[str, Any]] = []
    for raw_field in fields:
        field_name = str(raw_field)
        value = image_data.get(field_name)
        if not value:
            continue
        name = str(value).replace("\\", "/").strip("/")
        references[field_name] = name
        for relative_name, method in _relative_image_candidates(name, strip_path_prefixes):
            relative = Path(relative_name)
            safe = not relative.is_absolute() and ".." not in relative.parts
            candidate = images_root / relative if safe else None
            candidates.append(
                {
                    "field": field_name,
                    "method": method,
                    "relative": relative_name,
                    "path": candidate,
                    "safe": safe,
                    "is_file": bool(candidate and candidate.is_file()),
                }
            )
        basename = Path(name).name.casefold()
        indexed = basename_index.get(basename)
        basename_lookups.append(
            {
                "field": field_name,
                "basename": basename,
                "indexed_path": indexed,
                "status": (
                    "unique"
                    if indexed is not None
                    else "ambiguous_or_missing"
                ),
            }
        )
    return {
        "images_root": images_root,
        "references": references,
        "path_candidates": candidates,
        "basename_lookups": basename_lookups,
    }


def _relative_image_candidates(
    name: str,
    strip_path_prefixes: Iterable[str],
) -> list[tuple[str, str]]:
    """Return exact and configured prefix-stripped relative path candidates."""
    result = [(name, "exact")]
    folded_name = name.casefold()
    for raw_prefix in strip_path_prefixes:
        prefix = str(raw_prefix).replace("\\", "/").strip("/")
        marker = f"{prefix}/"
        if folded_name.startswith(marker.casefold()):
            stripped = name[len(marker) :]
            if stripped and stripped not in {item[0] for item in result}:
                result.append((stripped, "prefix_stripped"))
    return result


def _build_unique_basename_index(images_root: Path) -> dict[str, Path | None]:
    """Index unique image basenames and mark duplicates as ambiguous."""
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    result: dict[str, Path | None] = {}
    for path in images_root.rglob("*"):
        if not path.is_file() or path.suffix.casefold() not in supported:
            continue
        key = path.name.casefold()
        result[key] = None if key in result else path
    return result


def _valid_bbox(value: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    x, y, w, h = (float(item) for item in value)
    x1 = min(max(x, 0.0), float(width))
    y1 = min(max(y, 0.0), float(height))
    x2 = min(max(x + w, 0.0), float(width))
    y2 = min(max(y + h, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def _valid_polygons(value: Any, width: int, height: int) -> list[list[float]]:
    if not isinstance(value, list):  # COCO RLE dictionaries are intentionally rejected.
        return []
    candidates = value if value and isinstance(value[0], list) else [value]
    result: list[list[float]] = []
    for candidate in candidates:
        if not isinstance(candidate, list) or len(candidate) < 6 or len(candidate) % 2:
            continue
        points: list[float] = []
        for index in range(0, len(candidate), 2):
            points.extend(
                [
                    min(max(float(candidate[index]), 0.0), float(width)),
                    min(max(float(candidate[index + 1]), 0.0), float(height)),
                ]
            )
        if abs(_polygon_area(points)) > 1.0:
            result.append(points)
    return result


def _polygon_area(points: list[float]) -> float:
    pairs = list(zip(points[0::2], points[1::2]))
    return 0.5 * sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(pairs, pairs[1:] + pairs[:1])
    )


def _select_polygon(polygons: list[list[float]], policy: str) -> list[float]:
    if policy != "largest":
        raise DatasetBuildError("Only data.multipart_policy='largest' is supported")
    return max(polygons, key=lambda points: abs(_polygon_area(points)))


def _bbox_polygon(bbox: list[float]) -> list[float]:
    x, y, width, height = bbox
    return [x, y, x + width, y, x + width, y + height, x, y + height]


def _assign_splits(records: list[ImageRecord], split_cfg: dict[str, Any]) -> None:
    ratios = {
        "train": float(split_cfg.get("train", 0.7)),
        "val": float(split_cfg.get("val", 0.15)),
        "test": float(split_cfg.get("test", 0.15)),
    }
    seed = int(split_cfg.get("seed", 42))
    groups_by_source: dict[str, dict[str, list[ImageRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    digest_assignment: dict[str, str] = {}
    for record in records:
        groups_by_source[record.source][record.digest].append(record)

    for source, digest_groups in sorted(groups_by_source.items()):
        groups = list(digest_groups.values())
        random.Random(f"{seed}:{source}").shuffle(groups)
        counts = _split_counts(len(groups), ratios)
        cursor = 0
        for split_name in ("train", "val", "test"):
            for group in groups[cursor : cursor + counts[split_name]]:
                existing = digest_assignment.get(group[0].digest)
                assigned = existing or split_name
                digest_assignment[group[0].digest] = assigned
                for record in group:
                    record.split = assigned
            cursor += counts[split_name]


def _split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(raw, key=lambda name: (raw[name] - counts[name], ratios[name]), reverse=True)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def _materialize(
    config: TrainingConfig,
    records: list[ImageRecord],
    classes: list[str],
    source_reports: dict[str, Any],
    input_versions: dict[str, Any],
) -> dict[str, Any]:
    output_dir = config.dataset_dir
    temp_dir = output_dir.parent / f".{output_dir.name}.building"
    _safe_remove(temp_dir, config.project_root)
    temp_dir.mkdir(parents=True, exist_ok=True)
    mode = str(config.payload["data"].get("materialize", "hardlink"))
    split_reports: dict[str, Any] = {}

    try:
        for split in ("train", "val", "test"):
            (temp_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (temp_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        for record in records:
            safe_name = _safe_output_name(record)
            image_out = temp_dir / "images" / record.split / record.source / safe_name
            label_out = (
                temp_dir
                / "labels"
                / record.split
                / record.source
                / f"{Path(safe_name).stem}.txt"
            )
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(record.path, image_out, mode)
            lines = [
                _to_yolo(annotation, record, config.payload["model"]["task"])
                for annotation in record.annotations
            ]
            label_out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        for split in ("train", "val", "test"):
            split_records = [record for record in records if record.split == split]
            split_reports[split] = {
                "images": len(split_records),
                "instances": sum(len(record.annotations) for record in split_records),
                "images_by_source": dict(
                    sorted(Counter(record.source for record in split_records).items())
                ),
                "instances_by_class": _class_counts(split_records, classes),
            }

        yaml_lines = [
            "# Generated by edge-ai-mass. Paths are relative to this file.",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            f"nc: {len(classes)}",
            "names:",
            *(f"  {index}: {name}" for index, name in enumerate(classes)),
        ]
        (temp_dir / "dataset.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "data_config_digest": config.data_digest,
            "task": config.payload["model"]["task"],
            "classes": classes,
            "dataset_yaml": "dataset.yaml",
            "input_versions": input_versions,
            "sources": source_reports,
            "splits": split_reports,
            "total_images": len(records),
            "total_instances": sum(len(record.annotations) for record in records),
            "duplicate_images": len(records) - len({record.digest for record in records}),
        }
        # The identity excludes machine-specific mount paths so the same DVC or
        # Kaggle dataset receives the same fingerprint on every worker.
        fingerprint_basis = {
            "schema_version": manifest["schema_version"],
            "task": manifest["task"],
            "classes": manifest["classes"],
            "input_versions": {
                name: {
                    "annotations_sha256": version.get("annotations_sha256"),
                    "images_digest": version.get("images_digest"),
                }
                for name, version in sorted(input_versions.items())
            },
            "sources": {
                name: {
                    key: value
                    for key, value in report.items()
                    if key not in {"images_root_configured", "images_root_resolved"}
                }
                for name, report in sorted(source_reports.items())
            },
            "splits": split_reports,
        }
        fingerprint_payload = json.dumps(
            fingerprint_basis, sort_keys=True, separators=(",", ":")
        ).encode()
        manifest["dataset_fingerprint"] = hashlib.sha256(fingerprint_payload).hexdigest()
        _apply_quality_gates(
            manifest,
            config.payload["data"].get("quality", {}),
            debug_cfg=config.payload["data"].get("debug", {}),
        )
        (temp_dir / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (temp_dir / ".edge_ai_mass_dataset").write_text("managed\n", encoding="utf-8")

        _safe_remove(output_dir, config.project_root, require_marker=True)
        temp_dir.replace(output_dir)
        return manifest
    except Exception:
        _safe_remove(temp_dir, config.project_root)
        raise


def _to_yolo(annotation: Annotation, record: ImageRecord, task: str) -> str:
    if task == "segment":
        assert annotation.polygon is not None
        normalized = []
        for index in range(0, len(annotation.polygon), 2):
            normalized.extend(
                [
                    annotation.polygon[index] / record.width,
                    annotation.polygon[index + 1] / record.height,
                ]
            )
        return f"{annotation.class_id} " + " ".join(f"{value:.6f}" for value in normalized)
    assert annotation.bbox is not None
    x, y, width, height = annotation.bbox
    values = [
        (x + width / 2.0) / record.width,
        (y + height / 2.0) / record.height,
        width / record.width,
        height / record.height,
    ]
    return f"{annotation.class_id} " + " ".join(f"{value:.6f}" for value in values)


def _class_counts(records: list[ImageRecord], classes: list[str]) -> dict[str, int]:
    counts = Counter(annotation.class_id for record in records for annotation in record.annotations)
    return {label: int(counts.get(index, 0)) for index, label in enumerate(classes)}


def _safe_output_name(record: ImageRecord) -> str:
    suffix = record.path.suffix.lower() or ".jpg"
    stem = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in record.path.stem
    )
    return f"{record.source_id:08d}_{stem}{suffix}"


def _link_or_copy(source: Path, destination: Path, mode: str) -> None:
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    if mode == "symlink":
        destination.symlink_to(source)
        return
    if mode != "hardlink":
        raise DatasetBuildError("data.materialize must be hardlink, copy, or symlink")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _safe_remove(path: Path, project_root: Path, *, require_marker: bool = False) -> None:
    if not path.exists():
        return
    resolved = path.resolve()
    root = project_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise DatasetBuildError(f"Refusing to remove path outside project root: {resolved}")
    if require_marker and not (resolved / ".edge_ai_mass_dataset").exists():
        raise DatasetBuildError(
            f"Refusing to replace unmanaged dataset directory: {resolved}. "
            "Move it or add --set data.output_dir=..."
        )
    shutil.rmtree(resolved)


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_image_digest(records: list[ImageRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: (item.source_name, item.source_id)):
        digest.update(f"{record.source_name}\0{record.digest}\n".encode())
    return digest.hexdigest()


def _apply_quality_gates(
    manifest: dict[str, Any],
    quality_cfg: dict[str, Any],
    *,
    debug_cfg: dict[str, Any] | None = None,
) -> None:
    debug_cfg = debug_cfg or {}
    _emit_debug(
        debug_cfg,
        "quality.before",
        quality_config=quality_cfg,
        total_images=manifest.get("total_images"),
        total_instances=manifest.get("total_instances"),
    )
    if quality_cfg.get("require_all_classes", True):
        total_counts = Counter()
        for split in manifest["splits"].values():
            total_counts.update(split["instances_by_class"])
        missing = [label for label in manifest["classes"] if total_counts[label] == 0]
        if missing:
            _emit_debug(debug_cfg, "quality.failed", gate="all_classes", missing=missing)
            raise DatasetBuildError(
                f"Dataset quality gate failed; classes without instances: {missing}"
            )
    max_missing = float(quality_cfg.get("max_missing_image_fraction", 0.0))
    for name, report in manifest["sources"].items():
        declared = int(report.get("images_declared", 0))
        fraction = (int(report.get("missing_images", 0)) / declared) if declared else 0.0
        _emit_debug(
            debug_cfg,
            "quality.source",
            source=name,
            declared=declared,
            resolved=report.get("images"),
            missing=report.get("missing_images"),
            missing_fraction=fraction,
            allowed_fraction=max_missing,
            configured_root=report.get("images_root_configured"),
            resolved_root=report.get("images_root_resolved"),
            resolution_methods=report.get("image_resolution"),
            missing_examples=report.get("missing_image_examples"),
        )
        if fraction > max_missing:
            examples = report.get("missing_image_examples", [])
            first_missing = examples[0] if examples else None
            _emit_debug(
                debug_cfg,
                "quality.failed",
                gate="missing_image_fraction",
                source=name,
                fraction=fraction,
                allowed=max_missing,
                resolved_root=report.get("images_root_resolved"),
                first_missing=first_missing,
            )
            raise DatasetBuildError(
                f"Dataset quality gate failed; source '{name}' missing image fraction "
                f"{fraction:.3%} exceeds {max_missing:.3%}; "
                f"resolved_root={report.get('images_root_resolved')}; "
                f"first_missing={first_missing}"
            )
    _emit_debug(debug_cfg, "quality.after", status="passed")
