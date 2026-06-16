from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from prepare_merged_waste_seg_dataset import (
    AQUATRASH_LABEL_MAP,
    TARGET_LABELS,
    resolve_taco_label,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REALWASTE_LABEL_MAP = {
    "cardboard": "paper_cardboard",
    "food organics": "organic_waste",
    "glass": "glass",
    "metal": "metal_can",
    "miscellaneous trash": "mixed_waste",
    "paper": "paper_cardboard",
    "plastic": "rigid_plastic",
    "textile trash": "mixed_waste",
    "vegetation": "organic_waste",
}

DATASET_ALIASES = {
    "taco": "taco",
    "aquatrash": "aquatrash",
    "aqua_trash": "aquatrash",
    "aqua-trash": "aquatrash",
    "realwaste": "realwaste",
    "real_waste": "realwaste",
    "real-waste": "realwaste",
}


@dataclass(frozen=True)
class Sample:
    path: Path
    true_labels: tuple[str, ...]
    name: str


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ordered_labels(labels: Iterable[str]) -> tuple[str, ...]:
    label_order = {label: index for index, label in enumerate(TARGET_LABELS)}
    return tuple(sorted(set(labels), key=lambda label: label_order.get(label, len(label_order))))


def get_taco_images(data_root: Path) -> list[Sample]:
    taco_root = data_root / "taco"
    payload = load_json(taco_root / "annotations.json")
    categories = {int(category["id"]): category for category in payload.get("categories", [])}
    labels_by_image: dict[int, set[str]] = defaultdict(set)

    for annotation in payload.get("annotations", []):
        category = categories.get(int(annotation["category_id"]))
        if category is not None:
            labels_by_image[int(annotation["image_id"])].add(resolve_taco_label(category))

    samples: list[Sample] = []
    for image in payload.get("images", []):
        image_id = int(image["id"])
        true_labels = ordered_labels(labels_by_image.get(image_id, set()))
        image_path = taco_root / image["file_name"]
        if true_labels and image_path.exists():
            samples.append(Sample(path=image_path, true_labels=true_labels, name=image["file_name"]))
    return samples


def get_aquatrash_images(data_root: Path) -> list[Sample]:
    annotation_path = next(
        (
            path
            for path in [
                data_root / "aquatrash" / "labels_final.json",
                data_root / "aquatrash" / "annotations.json",
                data_root / "labels_final.json",
            ]
            if path.exists()
        ),
        None,
    )
    if annotation_path is None:
        raise FileNotFoundError("Could not find AquaTrash labels_final.json or annotations.json")

    images_root = data_root / "aquatrash" / "Images"
    payload = load_json(annotation_path)
    categories = {
        int(category["id"]): str(category.get("name", "mixed_waste")).strip()
        for category in payload.get("categories", [])
    }
    labels_by_image: dict[int, set[str]] = defaultdict(set)

    for annotation in payload.get("annotations", []):
        source_label = categories.get(int(annotation["category_id"]), "mixed_waste")
        labels_by_image[int(annotation["image_id"])].add(
            AQUATRASH_LABEL_MAP.get(source_label, "mixed_waste")
        )

    samples: list[Sample] = []
    for image in payload.get("images", []):
        image_id = int(image["id"])
        true_labels = ordered_labels(labels_by_image.get(image_id, set()))
        image_path = images_root / image["file_name"]
        if true_labels and image_path.exists():
            samples.append(Sample(path=image_path, true_labels=true_labels, name=image["file_name"]))
    return samples


def get_realwaste_images(data_root: Path) -> list[Sample]:
    realwaste_root = data_root / "RealWaste"
    samples: list[Sample] = []
    for class_dir in sorted(path for path in realwaste_root.iterdir() if path.is_dir()):
        label = REALWASTE_LABEL_MAP.get(class_dir.name.lower(), "mixed_waste")
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                samples.append(
                    Sample(
                        path=image_path,
                        true_labels=(label,),
                        name=f"{class_dir.name}/{image_path.name}",
                    )
                )
    return samples


def normalize_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().lower().replace(" ", "_")
    try:
        return DATASET_ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(set(DATASET_ALIASES.values())))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose one of: {choices}") from exc


def get_image(dataset_name: str, data_root: Path, sample_count: int, seed: int) -> list[Sample]:
    dataset_key = normalize_dataset_name(dataset_name)
    loaders: dict[str, Callable[[Path], list[Sample]]] = {
        "taco": get_taco_images,
        "aquatrash": get_aquatrash_images,
        "realwaste": get_realwaste_images,
    }
    samples = loaders[dataset_key](data_root)
    if not samples:
        raise RuntimeError(f"No labeled images found for dataset '{dataset_name}' in {data_root}")

    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples if sample_count <= 0 else samples[:sample_count]


def safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "item"


def discover_models(artifacts_dir: Path, model_filter: str | None) -> list[Path]:
    if model_filter:
        direct_path = Path(model_filter)
        if direct_path.exists():
            return [direct_path]

    models = sorted(artifacts_dir.rglob("*.pt"))
    if model_filter:
        model_filter_lower = model_filter.lower()
        models = [model for model in models if model_filter_lower in str(model).lower()]

    if not models:
        message = f"No .pt models found in {artifacts_dir}"
        if model_filter:
            message += f" matching '{model_filter}'"
        raise FileNotFoundError(message)
    return models


def predicted_labels(result) -> tuple[str, ...]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return ()

    names = result.names or {}
    labels: list[str] = []
    seen: set[str] = set()
    for class_id in boxes.cls.detach().cpu().numpy().astype(int).tolist():
        label = names[class_id] if isinstance(names, dict) else names[class_id]
        label = str(label)
        if label not in seen:
            labels.append(label)
            seen.add(label)
    return ordered_labels(labels)


def format_labels(labels: Iterable[str]) -> str:
    return ", ".join(labels) if labels else "none"


def save_labeled_image(result, sample: Sample, labels: tuple[str, ...], output_path: Path) -> None:
    plotted = result.plot()
    image = Image.fromarray(plotted[..., ::-1]).convert("RGB")

    if max(image.size) > 1280:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image.thumbnail((1280, 1280), resampling)

    font = ImageFont.load_default()
    lines = [
        f"TRUE: {format_labels(sample.true_labels)}",
        f"PRED: {format_labels(labels)}",
    ]
    line_heights = [
        ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), line, font=font)[3]
        for line in lines
    ]
    header_height = sum(line_heights) + 18 + (len(lines) - 1) * 4
    canvas = Image.new("RGB", (image.width, image.height + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    y = 6
    for line in lines:
        draw.text((8, y), line, fill="black", font=font)
        y += draw.textbbox((8, y), line, font=font)[3] + 4
    canvas.paste(image, (0, header_height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def test_model(model_path: Path, samples: list[Sample], args: argparse.Namespace) -> None:
    model = YOLO(str(model_path))
    predict_args = {
        "conf": args.conf,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device:
        predict_args["device"] = args.device

    results = model([str(sample.path) for sample in samples], **predict_args)
    model_slug = safe_slug(f"{model_path.parent.name}_{model_path.stem}")
    output_dir = args.output_dir / normalize_dataset_name(args.dataset) / model_slug

    hits = 0
    exact_hits = 0
    print(f"\nModel: {model_path}")
    for index, (sample, result) in enumerate(zip(samples, results), start=1):
        labels = predicted_labels(result)
        true_set = set(sample.true_labels)
        predicted_set = set(labels)
        hit = bool(true_set & predicted_set)
        exact = true_set == predicted_set
        hits += int(hit)
        exact_hits += int(exact)

        status = "OK" if hit else "MISS"
        annotated_path = output_dir / f"{index:03d}_{safe_slug(sample.path.stem)}.jpg"
        if not args.no_save_images:
            save_labeled_image(result, sample, labels, annotated_path)

        image_note = f" annotated={annotated_path}" if not args.no_save_images else ""
        print(
            f"[{status}] {sample.name} "
            f"true={format_labels(sample.true_labels)} "
            f"pred={format_labels(labels)}{image_note}"
        )

    total = len(samples)
    print(f"Accuracy: {hits}/{total} ({hits / total:.2%})")
    print(f"Exact label-set accuracy: {exact_hits}/{total} ({exact_hits / total:.2%})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test YOLO models from artifacts on labeled samples from a raw dataset."
    )
    parser.add_argument("dataset", help="Dataset to sample: taco, aquatrash, or realwaste")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--model", help="Optional model path or substring filter")
    parser.add_argument("--samples", type=int, default=8, help="Number of samples. Use 0 for all.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", help="Optional Ultralytics device, for example cpu or 0")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/test-models"))
    parser.add_argument("--no-save-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = get_image(args.dataset, args.data_root, args.samples, args.seed)
    models = discover_models(args.artifacts_dir, args.model)

    print(f"Dataset: {normalize_dataset_name(args.dataset)}")
    print(f"Samples: {len(samples)}")
    print(f"Models: {len(models)}")

    for model_path in models:
        test_model(model_path, samples, args)


if __name__ == "__main__":
    main()
