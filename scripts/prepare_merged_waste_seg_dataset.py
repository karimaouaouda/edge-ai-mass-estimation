"""Build a merged TACO + AquaTrash YOLO segmentation dataset.

The output dataset uses the same 8-class taxonomy as the project notebook:

0 organic_waste
1 plastic_bottle
2 plastic_bag
3 rigid_plastic
4 metal_can
5 glass
6 paper_cardboard
7 mixed_waste

TACO is read from its COCO annotations.json file. AquaTrash is read from
labels_final.json for segmentation polygons and annotations.csv for box fallback
annotations only when explicitly requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


TARGET_LABELS = [
    "organic_waste",
    "plastic_bottle",
    "plastic_bag",
    "rigid_plastic",
    "metal_can",
    "glass",
    "paper_cardboard",
    "mixed_waste",
]

OLD_TO_NEW_LABEL = {
    "Food waste": "organic_waste",
    "Other plastic bottle": "plastic_bottle",
    "Clear plastic bottle": "plastic_bottle",
    "Plastic film": "plastic_bag",
    "Six pack rings": "plastic_bag",
    "Garbage bag": "plastic_bag",
    "Other plastic wrapper": "plastic_bag",
    "Single-use carrier bag": "plastic_bag",
    "Polypropylene bag": "plastic_bag",
    "Crisp packet": "plastic_bag",
    "Plastic bottle cap": "rigid_plastic",
    "Plastic lid": "rigid_plastic",
    "Other plastic": "rigid_plastic",
    "Disposable plastic cup": "rigid_plastic",
    "Foam cup": "rigid_plastic",
    "Other plastic cup": "rigid_plastic",
    "Spread tub": "rigid_plastic",
    "Tupperware": "rigid_plastic",
    "Disposable food container": "rigid_plastic",
    "Foam food container": "rigid_plastic",
    "Other plastic container": "rigid_plastic",
    "Plastic glooves": "rigid_plastic",
    "Plastic utensils": "rigid_plastic",
    "Squeezable tube": "rigid_plastic",
    "Plastic straw": "rigid_plastic",
    "Styrofoam piece": "mixed_waste",
    "Aluminium foil": "metal_can",
    "Aluminium blister pack": "metal_can",
    "Metal bottle cap": "metal_can",
    "Food Can": "metal_can",
    "Aerosol": "metal_can",
    "Drink can": "metal_can",
    "Metal lid": "metal_can",
    "Pop tab": "metal_can",
    "Scrap metal": "metal_can",
    "Glass bottle": "glass",
    "Broken glass": "glass",
    "Glass cup": "glass",
    "Glass jar": "glass",
    "Toilet tube": "paper_cardboard",
    "Other carton": "paper_cardboard",
    "Egg carton": "paper_cardboard",
    "Drink carton": "paper_cardboard",
    "Corrugated carton": "paper_cardboard",
    "Meal carton": "paper_cardboard",
    "Pizza box": "paper_cardboard",
    "Paper cup": "paper_cardboard",
    "Magazine paper": "paper_cardboard",
    "Tissues": "paper_cardboard",
    "Wrapping paper": "paper_cardboard",
    "Normal paper": "paper_cardboard",
    "Paper bag": "paper_cardboard",
    "Plastified paper bag": "paper_cardboard",
    "Paper straw": "paper_cardboard",
    "Battery": "mixed_waste",
    "Carded blister pack": "mixed_waste",
    "Rope & strings": "mixed_waste",
    "Shoe": "mixed_waste",
    "Unlabeled litter": "mixed_waste",
    "Cigarette": "mixed_waste",
}

AQUATRASH_LABEL_MAP = {
    "glass": "glass",
    "metal": "metal_can",
    "metal_can": "metal_can",
    "paper": "paper_cardboard",
    "paper_cardboard": "paper_cardboard",
    "plastic": "rigid_plastic",
    "plastic_battle": "plastic_bottle",
    "plastic_bottle": "plastic_bottle",
    "plastic_bag": "plastic_bag",
    "rigid_plastic": "rigid_plastic",
    "organic_waste": "organic_waste",
    "mixed_waste": "mixed_waste",
}


def resolve_taco_label(category: dict[str, Any]) -> str:
    name = str(category.get("name", "")).strip()
    supercategory = str(category.get("supercategory", "")).strip()
    lower_name = name.lower()
    lower_supercategory = supercategory.lower()

    if name in OLD_TO_NEW_LABEL:
        return OLD_TO_NEW_LABEL[name]

    if lower_supercategory == "bottle":
        return "glass" if "glass" in lower_name else "plastic_bottle"
    if lower_supercategory == "bottle cap":
        return "metal_can" if "metal" in lower_name else "rigid_plastic"
    if lower_supercategory in {"paper", "carton", "paper bag"}:
        return "paper_cardboard"
    if lower_supercategory == "plastic bag & wrapper":
        return "plastic_bag"
    if lower_supercategory == "plastic container":
        return "rigid_plastic"
    if lower_supercategory == "cup":
        if lower_name.startswith("glass"):
            return "glass"
        if lower_name.startswith("paper"):
            return "paper_cardboard"
        return "rigid_plastic"
    if lower_supercategory == "lid":
        return "metal_can" if "metal" in lower_name else "rigid_plastic"
    if lower_supercategory == "other plastic":
        return "rigid_plastic"
    if lower_supercategory == "straw":
        return "paper_cardboard" if lower_name.startswith("paper") else "rigid_plastic"
    if lower_supercategory == "food waste":
        return "organic_waste"
    if lower_supercategory in {"battery", "unlabeled litter", "rope & strings", "shoe", "cigarette"}:
        return "mixed_waste"
    if "glass" in lower_name:
        return "glass"
    if "can" in lower_name or "foil" in lower_name or "metal" in lower_name:
        return "metal_can"
    if "paper" in lower_name or "carton" in lower_name or "tissue" in lower_name or "box" in lower_name:
        return "paper_cardboard"
    if (
        "plastic" in lower_name
        or "foam" in lower_name
        or "tupperware" in lower_name
        or "tube" in lower_name
        or "glove" in lower_name
        or "utensil" in lower_name
        or "straw" in lower_name
    ):
        return "rigid_plastic"
    return "mixed_waste"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize_polygon(seg: list[float], width: int, height: int) -> list[str]:
    points: list[str] = []
    for index in range(0, len(seg), 2):
        if index + 1 >= len(seg):
            break
        x = max(0.0, min(1.0, float(seg[index]) / float(width)))
        y = max(0.0, min(1.0, float(seg[index + 1]) / float(height)))
        points.extend([f"{x:.6f}", f"{y:.6f}"])
    return points


def xywh_to_xyxy(box: list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = [float(value) for value in box]
    return x, y, x + w, y + h


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def rect_polygon_from_xyxy(box: tuple[float, float, float, float]) -> list[float]:
    x1, y1, x2, y2 = box
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def split_image_ids(image_ids: list[int], train_ratio: float, val_ratio: float, seed: int) -> dict[str, set[int]]:
    shuffled = list(image_ids)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": set(shuffled[:train_end]),
        "val": set(shuffled[train_end:val_end]),
        "test": set(shuffled[val_end:]),
    }


def split_by_source(records: list[dict[str, Any]], train_ratio: float, val_ratio: float, seed: int) -> dict[int, str]:
    by_source: dict[str, list[int]] = defaultdict(list)
    for record in records:
        by_source[record["source"]].append(int(record["id"]))

    image_to_split: dict[int, str] = {}
    for source, image_ids in sorted(by_source.items()):
        splits = split_image_ids(image_ids, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
        for split, ids in splits.items():
            for image_id in ids:
                image_to_split[image_id] = split
    return image_to_split


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def build_records(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]], dict[str, Any]]:
    label_to_id = {label: index for index, label in enumerate(TARGET_LABELS)}
    images: list[dict[str, Any]] = []
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    summary: dict[str, Any] = {
        "classes": TARGET_LABELS,
        "sources": {},
        "box_fallback_annotations": 0,
        "skipped_annotations": 0,
    }

    next_image_id = 1
    next_annotation_id = 1

    taco_payload = load_json(args.taco_annotations)
    taco_categories = {int(category["id"]): category for category in taco_payload.get("categories", [])}
    taco_counts = Counter()
    taco_missing_images = 0

    for image in taco_payload.get("images", []):
        source_path = args.taco_images_root / image["file_name"]
        if not source_path.exists():
            taco_missing_images += 1
            continue

        merged_id = next_image_id
        next_image_id += 1
        output_rel = Path("taco") / image["file_name"]
        images.append(
            {
                "id": merged_id,
                "source": "taco",
                "source_image_id": image["id"],
                "source_file_name": image["file_name"],
                "file_name": output_rel.as_posix(),
                "source_path": source_path,
                "width": int(image["width"]),
                "height": int(image["height"]),
            }
        )
        image["_merged_id"] = merged_id

    taco_id_to_merged = {int(image["source_image_id"]): int(image["id"]) for image in images if image["source"] == "taco"}
    for annotation in taco_payload.get("annotations", []):
        merged_image_id = taco_id_to_merged.get(int(annotation["image_id"]))
        if merged_image_id is None:
            summary["skipped_annotations"] += 1
            continue
        category = taco_categories.get(int(annotation["category_id"]))
        if category is None:
            summary["skipped_annotations"] += 1
            continue
        label = resolve_taco_label(category)
        class_id = label_to_id[label]
        annotations_by_image[merged_image_id].append(
            {
                "id": next_annotation_id,
                "source": "taco",
                "category_id": class_id,
                "segmentation": annotation.get("segmentation", []),
                "bbox": annotation.get("bbox"),
            }
        )
        next_annotation_id += 1
        taco_counts[label] += 1

    summary["sources"]["taco"] = {
        "images": sum(1 for image in images if image["source"] == "taco"),
        "annotations": int(sum(taco_counts.values())),
        "missing_images": taco_missing_images,
        "counts_by_class": dict(taco_counts),
    }

    aqua_payload = load_json(args.aquatrash_segmentations)
    aqua_categories = {int(category["id"]): category["name"] for category in aqua_payload.get("categories", [])}
    aqua_counts = Counter()
    aqua_missing_images = 0
    aqua_file_to_merged: dict[str, int] = {}
    aqua_json_boxes_by_file: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

    for image in aqua_payload.get("images", []):
        file_name = image["file_name"]
        source_path = args.aquatrash_images_root / file_name
        if not source_path.exists():
            aqua_missing_images += 1
            continue

        merged_id = next_image_id
        next_image_id += 1
        output_rel = Path("aquatrash") / file_name
        images.append(
            {
                "id": merged_id,
                "source": "aquatrash",
                "source_image_id": image["id"],
                "source_file_name": file_name,
                "file_name": output_rel.as_posix(),
                "source_path": source_path,
                "width": int(image["width"]),
                "height": int(image["height"]),
            }
        )
        aqua_file_to_merged[file_name] = merged_id

    aqua_id_to_file = {int(image["id"]): image["file_name"] for image in aqua_payload.get("images", [])}
    for annotation in aqua_payload.get("annotations", []):
        file_name = aqua_id_to_file.get(int(annotation["image_id"]))
        if not file_name:
            summary["skipped_annotations"] += 1
            continue
        merged_image_id = aqua_file_to_merged.get(file_name)
        if merged_image_id is None:
            summary["skipped_annotations"] += 1
            continue
        source_label = str(aqua_categories.get(int(annotation["category_id"]), "mixed_waste")).strip()
        label = AQUATRASH_LABEL_MAP.get(source_label, "mixed_waste")
        class_id = label_to_id[label]
        bbox = annotation.get("bbox")
        if bbox:
            aqua_json_boxes_by_file[file_name].append(xywh_to_xyxy(bbox))
        annotations_by_image[merged_image_id].append(
            {
                "id": next_annotation_id,
                "source": "aquatrash",
                "category_id": class_id,
                "segmentation": annotation.get("segmentation", []),
                "bbox": bbox,
            }
        )
        next_annotation_id += 1
        aqua_counts[label] += 1

    if args.include_aquatrash_box_fallbacks and args.aquatrash_boxes.exists():
        with args.aquatrash_boxes.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                file_name = row["image_name"]
                merged_image_id = aqua_file_to_merged.get(file_name)
                if merged_image_id is None:
                    continue
                box = (
                    float(row["x_min"]),
                    float(row["y_min"]),
                    float(row["x_max"]),
                    float(row["y_max"]),
                )
                has_matching_polygon = any(
                    box_iou(box, existing_box) >= args.box_fallback_iou
                    for existing_box in aqua_json_boxes_by_file.get(file_name, [])
                )
                if has_matching_polygon:
                    continue

                image_record = next(image for image in images if int(image["id"]) == merged_image_id)
                width = int(image_record["width"])
                height = int(image_record["height"])
                clipped_box = (
                    max(0.0, min(float(width), box[0])),
                    max(0.0, min(float(height), box[1])),
                    max(0.0, min(float(width), box[2])),
                    max(0.0, min(float(height), box[3])),
                )
                if clipped_box[2] <= clipped_box[0] or clipped_box[3] <= clipped_box[1]:
                    continue

                source_label = str(row["class_name"]).strip()
                label = AQUATRASH_LABEL_MAP.get(source_label, "mixed_waste")
                class_id = label_to_id[label]
                annotations_by_image[merged_image_id].append(
                    {
                        "id": next_annotation_id,
                        "source": "aquatrash_box_fallback",
                        "category_id": class_id,
                        "segmentation": [rect_polygon_from_xyxy(clipped_box)],
                        "bbox": [
                            clipped_box[0],
                            clipped_box[1],
                            clipped_box[2] - clipped_box[0],
                            clipped_box[3] - clipped_box[1],
                        ],
                    }
                )
                next_annotation_id += 1
                summary["box_fallback_annotations"] += 1
                aqua_counts[label] += 1

    summary["sources"]["aquatrash"] = {
        "images": sum(1 for image in images if image["source"] == "aquatrash"),
        "annotations": int(sum(aqua_counts.values())),
        "missing_images": aqua_missing_images,
        "counts_by_class": dict(aqua_counts),
    }
    return images, annotations_by_image, summary


def write_dataset(
    images: list[dict[str, Any]],
    annotations_by_image: dict[int, list[dict[str, Any]]],
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, Any]:
    ensure_clean_dir(output_dir)
    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_to_split = split_by_source(images, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    split_summary: dict[str, Any] = {}
    annotations_payload_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    images_payload_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for image in images:
        split = image_to_split[int(image["id"])]
        dst_image = output_dir / "images" / split / image["file_name"]
        dst_label = (output_dir / "labels" / split / image["file_name"]).with_suffix(".txt")
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image["source_path"], dst_image)

        width = int(image["width"])
        height = int(image["height"])
        label_lines: list[str] = []
        for annotation in annotations_by_image.get(int(image["id"]), []):
            segs = annotation.get("segmentation", [])
            if not isinstance(segs, list):
                continue
            for seg in segs:
                if not isinstance(seg, list) or len(seg) < 6:
                    continue
                points = normalize_polygon(seg, width=width, height=height)
                if len(points) >= 6:
                    label_lines.append(f"{annotation['category_id']} " + " ".join(points))

            payload_annotation = {
                "id": annotation["id"],
                "image_id": image["id"],
                "category_id": annotation["category_id"],
                "segmentation": segs,
                "bbox": annotation.get("bbox"),
                "source": annotation.get("source"),
            }
            annotations_payload_by_split[split].append(payload_annotation)

        dst_label.write_text("\n".join(label_lines), encoding="utf-8")
        images_payload_by_split[split].append(
            {
                "id": image["id"],
                "width": width,
                "height": height,
                "file_name": image["file_name"],
                "source": image["source"],
                "source_file_name": image["source_file_name"],
            }
        )

    categories = [
        {"id": index, "name": label, "supercategory": label}
        for index, label in enumerate(TARGET_LABELS)
    ]
    for split in ["train", "val", "test"]:
        split_images = images_payload_by_split[split]
        split_annotations = annotations_payload_by_split[split]
        split_summary[split] = {
            "images": len(split_images),
            "annotations": len(split_annotations),
            "label_files": len(list((output_dir / "labels" / split).rglob("*.txt"))),
            "instances": sum(
                1
                for label_path in (output_dir / "labels" / split).rglob("*.txt")
                for line in label_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ),
            "images_by_source": dict(Counter(image["source"] for image in split_images)),
        }
        payload = {
            "images": split_images,
            "annotations": split_annotations,
            "categories": categories,
        }
        (output_dir / f"annotations_{split}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    yaml_lines = [
        f"path: {output_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(TARGET_LABELS)}",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {label}" for index, label in enumerate(TARGET_LABELS))
    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    return {
        "dataset_yaml": str(dataset_yaml.resolve()),
        "classes": TARGET_LABELS,
        "splits": split_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taco-annotations", type=Path, default=Path("data/raw/taco/annotations.json"))
    parser.add_argument("--taco-images-root", type=Path, default=Path("data/raw/taco"))
    parser.add_argument("--aquatrash-segmentations", type=Path, default=Path("data/raw/aquatrash/labels_final.json"))
    parser.add_argument("--aquatrash-boxes", type=Path, default=Path("data/raw/aquatrash/annotations.csv"))
    parser.add_argument("--aquatrash-images-root", type=Path, default=Path("data/raw/aquatrash/Images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/taco_aquatrash_yolo_seg"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--box-fallback-iou", type=float, default=0.8)
    parser.add_argument(
        "--include-aquatrash-box-fallbacks",
        action="store_true",
        help=(
            "Add rectangular segmentation labels from AquaTrash annotations.csv when no matching "
            "polygon bbox is found. Disabled by default to avoid duplicate labels."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be below 1.0")

    images, annotations_by_image, source_summary = build_records(args)
    dataset_summary = write_dataset(
        images=images,
        annotations_by_image=annotations_by_image,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    summary = {
        **dataset_summary,
        **source_summary,
        "total_images": len(images),
        "total_annotations": sum(len(value) for value in annotations_by_image.values()),
        "output_dir": str(args.output_dir.resolve()),
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1.0 - args.train_ratio - args.val_ratio,
        },
    }
    summary_path = args.output_dir / "merge_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
