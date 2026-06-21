"""Dataset QA visualizations shared by CLI and notebook training flows."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from edge_ai_mass.training.config import TrainingConfig


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_COLORS = (
    (46, 204, 113),
    (52, 152, 219),
    (155, 89, 182),
    (241, 196, 15),
    (230, 126, 34),
    (26, 188, 156),
    (231, 76, 60),
    (149, 165, 166),
)


def create_dataset_visualizations(
    config: TrainingConfig,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Render deterministic annotated samples and contact sheets for every split."""
    visualization = config.payload["data"].get("visualization", {})
    if not visualization.get("enabled", True):
        return {"enabled": False, "reason": "data.visualization.enabled is false"}

    manifest_path = config.dataset_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = config.artifacts_dir / "dataset_visualizations"
    report_path = output_dir / "visualization_report.json"
    if report_path.is_file() and not force:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("dataset_fingerprint") == manifest.get("dataset_fingerprint"):
            return report

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [str(value) for value in config.payload["data"]["classes"]]
    samples_per_split = int(visualization.get("samples_per_split", 12))
    columns = max(1, int(visualization.get("columns", 3)))
    cell_width = int(visualization.get("cell_width", 480))
    cell_height = int(visualization.get("cell_height", 360))
    max_side = int(visualization.get("sample_max_side", 1280))
    seed = int(visualization.get("seed", config.payload["data"].get("split", {}).get("seed", 42)))
    requested_splits = visualization.get("splits", ["train", "val", "test"])
    split_reports: dict[str, Any] = {}

    for split in requested_splits:
        images_root = config.dataset_dir / "images" / str(split)
        if not images_root.is_dir():
            continue
        image_paths = sorted(
            path for path in images_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
        )
        rng = random.Random(f"{seed}:{split}")
        selected = rng.sample(image_paths, min(samples_per_split, len(image_paths)))
        samples_dir = output_dir / str(split) / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        rendered: list[tuple[Path, str]] = []

        for index, image_path in enumerate(selected):
            relative = image_path.relative_to(images_root)
            label_path = (config.dataset_dir / "labels" / str(split) / relative).with_suffix(
                ".txt"
            )
            annotated, instance_count = _annotate_image(
                image_path,
                label_path,
                classes,
                task=str(config.payload["model"]["task"]),
            )
            annotated.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            safe_source = "_".join(relative.parts[:-1]) or "root"
            output_path = samples_dir / f"{index:03d}_{safe_source}_{image_path.stem}.jpg"
            annotated.convert("RGB").save(output_path, quality=92)
            caption = f"{relative.as_posix()} | {instance_count} instances"
            rendered.append((output_path, caption))

        mosaic_path = output_dir / str(split) / "dataset_mosaic.jpg"
        if rendered:
            _write_contact_sheet(
                rendered,
                mosaic_path,
                columns=columns,
                cell_width=cell_width,
                cell_height=cell_height,
            )
        split_reports[str(split)] = {
            "available_images": len(image_paths),
            "rendered_samples": len(rendered),
            "mosaic": str(mosaic_path) if mosaic_path.is_file() else None,
            "samples_dir": str(samples_dir),
        }

    report = {
        "enabled": True,
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "output_dir": str(output_dir),
        "splits": split_reports,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _annotate_image(
    image_path: Path,
    label_path: Path,
    classes: list[str],
    *,
    task: str,
) -> tuple[Image.Image, int]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    lines = label_path.read_text(encoding="utf-8").splitlines() if label_path.is_file() else []
    instance_count = 0

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        if class_id < 0 or class_id >= len(classes):
            continue
        values = [float(value) for value in parts[1:]]
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        label = classes[class_id]
        anchor = (2, 2)
        if task == "segment" and len(values) >= 6 and len(values) % 2 == 0:
            points = [
                (int(values[index] * width), int(values[index + 1] * height))
                for index in range(0, len(values), 2)
            ]
            draw.polygon(points, fill=(*color, 65), outline=(*color, 255), width=3)
            anchor = points[0]
        elif len(values) == 4:
            center_x, center_y, box_width, box_height = values
            x1 = int((center_x - box_width / 2) * width)
            y1 = int((center_y - box_height / 2) * height)
            x2 = int((center_x + box_width / 2) * width)
            y2 = int((center_y + box_height / 2) * height)
            draw.rectangle((x1, y1, x2, y2), outline=(*color, 255), width=3)
            anchor = (x1, y1)
        else:
            continue
        _draw_label(draw, anchor, label, color, font)
        instance_count += 1

    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"), instance_count


def _draw_label(
    draw: ImageDraw.ImageDraw,
    anchor: tuple[int, int],
    label: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    x = max(0, anchor[0])
    y = max(0, anchor[1] - 13)
    box = draw.textbbox((x, y), label, font=font)
    draw.rectangle(box, fill=(*color, 230))
    draw.text((x, y), label, fill=(255, 255, 255, 255), font=font)


def _write_contact_sheet(
    rendered: list[tuple[Path, str]],
    output_path: Path,
    *,
    columns: int,
    cell_width: int,
    cell_height: int,
) -> None:
    caption_height = 28
    rows = (len(rendered) + columns - 1) // columns
    sheet_size = (columns * cell_width, rows * (cell_height + caption_height))
    sheet = Image.new("RGB", sheet_size, "#111827")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (image_path, caption) in enumerate(rendered):
        row, column = divmod(index, columns)
        image = Image.open(image_path).convert("RGB")
        contained = ImageOps.contain(image, (cell_width, cell_height), Image.Resampling.LANCZOS)
        x = column * cell_width + (cell_width - contained.width) // 2
        y = row * (cell_height + caption_height) + (cell_height - contained.height) // 2
        sheet.paste(contained, (x, y))
        draw.text(
            (column * cell_width + 6, row * (cell_height + caption_height) + cell_height + 7),
            caption[:80],
            fill="white",
            font=font,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)
