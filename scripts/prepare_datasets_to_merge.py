("""Prepare datasets for merging utilities.

This script currently provides an argument parser and a minimal CLI
entrypoint. The actual merging logic should be implemented where
indicated in `main()`.
""")

from __future__ import annotations

import argparse
from multiprocessing.spawn import prepare
from typing import Sequence

import os
import json

import shutil

datasets_list = ["taco", "aquatrash", "realwaste"]


OUTPUT_DIR = "data/normalized"
AQUATRASH_ANNOTATIONS_PATH = "data/raw/aquatrash/annotations.json"
REALWASTE_SAM_MODEL = os.getenv("REALWASTE_SAM_MODEL", "mobile_sam.pt")
REALWASTE_SAM_CACHE_NAME = "sam_segmentations_cache.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


TACO_TO_AQUATRASH_LABEL = {
    "Food waste": "organic_waste",
    "Other plastic bottle": "plastic_battle",
    "Clear plastic bottle": "plastic_battle",
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


REALWASTE_TO_AQUATRASH_LABEL = {
    "Cardboard": "paper_cardboard",
    "Food Organics": "organic_waste",
    "Glass": "glass",
    "Metal": "metal_can",
    "Miscellaneous Trash": "mixed_waste",
    "Paper": "paper_cardboard",
    "Plastic": "rigid_plastic",
    "Textile Trash": "mixed_waste",
    "Vegetation": "organic_waste",
}


def load_aquatrash_categories(annotations_path: str = AQUATRASH_ANNOTATIONS_PATH) -> list[dict]:
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(
            f"AquaTrash annotations file '{annotations_path}' not found. It is required to map TACO classes into the AquaTrash 8-class label space."
        )

    with open(annotations_path, "r") as f:
        annotations = json.load(f)
    return annotations["categories"]


def resolve_taco_label(category: dict) -> str:
    name = str(category.get("name", "")).strip()
    supercategory = str(category.get("supercategory", "")).strip()
    lower_name = name.lower()
    lower_supercategory = supercategory.lower()

    if name in TACO_TO_AQUATRASH_LABEL:
        return TACO_TO_AQUATRASH_LABEL[name]

    if lower_supercategory == "bottle":
        return "glass" if "glass" in lower_name else "plastic_battle"
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


def resolve_realwaste_label(class_name: str) -> str:
    label = REALWASTE_TO_AQUATRASH_LABEL.get(class_name)
    if label is None:
        raise KeyError(f"RealWaste class '{class_name}' does not have an AquaTrash label mapping.")
    return label


def resolve_existing_dir(root_dir: str) -> str:
    if os.path.isdir(root_dir):
        return root_dir

    parent_dir = os.path.dirname(root_dir) or "."
    target_name = os.path.basename(root_dir).lower()
    if os.path.isdir(parent_dir):
        for name in os.listdir(parent_dir):
            candidate = os.path.join(parent_dir, name)
            if name.lower() == target_name and os.path.isdir(candidate):
                return candidate

    raise FileNotFoundError(f"Directory '{root_dir}' was not found.")


def safe_file_component(value: str) -> str:
    return "_".join(value.replace("-", "_").split())


def polygon_area(segmentation: list[float]) -> float:
    if len(segmentation) < 6:
        return 0.0

    area = 0.0
    points = list(zip(segmentation[0::2], segmentation[1::2]))
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += (x1 * y2) - (x2 * y1)
    return abs(area) / 2.0


def bbox_from_segmentation(segmentation: list[float]) -> list[float]:
    xs = segmentation[0::2]
    ys = segmentation[1::2]
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def rectangle_segmentation(width: int, height: int) -> list[float]:
    max_x = float(max(width - 1, 1))
    max_y = float(max(height - 1, 1))
    return [0.0, 0.0, max_x, 0.0, max_x, max_y, 0.0, max_y]


def clamp_polygon(segmentation: list[float], width: int, height: int) -> list[float]:
    max_x = float(max(width - 1, 1))
    max_y = float(max(height - 1, 1))
    clamped = []
    for index in range(0, len(segmentation) - 1, 2):
        x = max(0.0, min(max_x, float(segmentation[index])))
        y = max(0.0, min(max_y, float(segmentation[index + 1])))
        clamped.extend([x, y])
    return clamped


def load_sam_model():
    try:
        from ultralytics import SAM
    except ImportError as exc:
        raise ImportError(
            "Preparing RealWaste requires Ultralytics SAM. Install project dependencies and run with the project virtual environment."
        ) from exc

    return SAM(REALWASTE_SAM_MODEL)


def generate_sam_segmentation(sam_model, image_path: str, width: int, height: int) -> tuple[list[float], bool]:
    point = [[width / 2.0, height / 2.0]]
    results = sam_model.predict(
        source=image_path,
        points=point,
        labels=[1],
        verbose=False,
        save=False,
        retina_masks=True,
    )

    polygons = []
    if results and getattr(results[0], "masks", None) is not None:
        for polygon in results[0].masks.xy:
            segmentation = []
            for x, y in polygon:
                segmentation.extend([float(x), float(y)])
            segmentation = clamp_polygon(segmentation, width, height)
            area = polygon_area(segmentation)
            if len(segmentation) >= 6 and area > 0:
                polygons.append((area, segmentation))

    if not polygons:
        return rectangle_segmentation(width, height), False

    polygons.sort(key=lambda item: item[0], reverse=True)
    return polygons[0][1], True


def read_image_size(image_path: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(image_path) as image:
        return image.size


def load_sam_cache(cache_path: str) -> dict:
    if not os.path.exists(cache_path):
        return {}
    with open(cache_path, "r") as f:
        return json.load(f)


def save_sam_cache(cache_path: str, cache: dict) -> None:
    with open(cache_path, "w") as f:
        json.dump(cache, f)


def validate_dataset(dataset_name: str) -> str:
    """Validate that the provided dataset name is in the allowed list."""
    normalized_dataset_name = dataset_name.lower()
    if normalized_dataset_name not in datasets_list:
        raise argparse.ArgumentTypeError(
            f"Invalid dataset '{dataset_name}'. Allowed values are: {', '.join(datasets_list)}."
        )
    return normalized_dataset_name


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	"""Parse command-line arguments for dataset preparation.

	Args:
		argv: Optional list of arguments (for testing). If ``None``, uses
			``sys.argv`` via ``argparse``'s default behavior.

	Returns:
		The populated ``argparse.Namespace`` object.
	"""
	parser = argparse.ArgumentParser(description="Prepare datasets to merge")
	parser.add_argument(
		"--dataset",
		required=True,
		help="dataset name",
	)
	return parser.parse_args(argv)



def prepare_taco(root_dir: str, json_file: str = "annotations.json") -> None:
    """Prepare the TACO dataset for merging.

    This function should implement the logic to prepare the TACO dataset
    for merging, such as downloading, extracting, and reformatting the
    data as needed.

    Args:
        root_dir: The root directory where the TACO dataset should be prepared.
        json_file: The name of the JSON file containing the dataset information.
    """

    annotations_path = os.path.join(root_dir, json_file)
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations file '{annotations_path}' not found. Please ensure the TACO dataset is downloaded and extracted correctly.")
        return
    

    dist_dir = os.path.join(OUTPUT_DIR, "taco")

    os.makedirs(dist_dir, exist_ok=True)
    
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    aquatrash_categories = load_aquatrash_categories()
    aquatrash_label_to_id = {
        category["name"]: category["id"]
        for category in aquatrash_categories
    }
    taco_categories = {
        category["id"]: category
        for category in annotations["categories"]
    }

    print(f"Loaded {len(annotations['images'])} images from '{annotations_path}'.")


    # rename images from image_name to batch_xx_image_name
    for image in annotations["images"]:
        file_name = image["file_name"] # filename = "batch_xxx/img_name.jpg"
        batch_name = file_name.split("/")[0] # batch_xxx
        img_name = file_name.split("/")[1] # img_name.jpg
        new_file_name = f"{batch_name}_{img_name}" # batch_xxx_img_name.jpg
        image["file_name"] = new_file_name

        # move the image to the new location
        old_image_path = os.path.join(root_dir, file_name)
        new_image_path = os.path.join(dist_dir, new_file_name)
        if not os.path.exists(old_image_path):
            raise FileNotFoundError(f"Image file '{old_image_path}' not found. Please ensure the TACO dataset is downloaded and extracted correctly.")
            return
        
        shutil.copy2(old_image_path, new_image_path)

    for annotation in annotations["annotations"]:
        taco_category = taco_categories.get(annotation["category_id"])
        if taco_category is None:
            raise KeyError(f"TACO category id '{annotation['category_id']}' was not found in annotations categories.")

        aquatrash_label = resolve_taco_label(taco_category)
        if aquatrash_label not in aquatrash_label_to_id:
            raise KeyError(f"Mapped TACO label '{aquatrash_label}' was not found in AquaTrash categories.")

        annotation["category_id"] = aquatrash_label_to_id[aquatrash_label]

    annotations["categories"] = aquatrash_categories

    dist_annotations_path = os.path.join(dist_dir, json_file)
    with open(dist_annotations_path, "w") as f:
        json.dump(annotations, f)



def prepare_aquatrash(root_dir: str, annotations_path: str) -> None:
    """Prepare the Aquatrash dataset for merging.

    This function should implement the logic to prepare the Aquatrash dataset
    for merging, such as downloading, extracting, and reformatting the
    data as needed.

    Args:
        root_dir: The root directory where the Aquatrash dataset should be prepared.
        annotations_path: The path to the annotations file.
    """

    dist_dir = os.path.join(OUTPUT_DIR, "aquatrash")
    os.makedirs(dist_dir, exist_ok=True)

    images_dir = os.path.join(root_dir, "Images")

    # we will use json_segmentations that i build and generate bboxes for it
    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    for image_info in annotations["images"]:
        file_name = image_info["file_name"] # filename = "img_name.jpg"
        new_file_name = file_name # img_name.jpg
        image_info["file_name"] = new_file_name

        # move the image to the new location
        old_image_path = os.path.join(images_dir, file_name)
        new_image_path = os.path.join(dist_dir, new_file_name)
        if not os.path.exists(old_image_path):
            raise FileNotFoundError(f"Image file '{old_image_path}' not found. Please ensure the Aquatrash dataset is downloaded and extracted correctly.")
            return
        
        shutil.copy2(old_image_path, new_image_path)

    dist_annotations_path = os.path.join(dist_dir, "annotations.json")
    with open(dist_annotations_path, "w") as f:
        json.dump(annotations, f)


def prepare_realwaste(root_dir: str) -> None:
    """Prepare RealWaste as a COCO-like segmentation dataset.

    RealWaste is stored as a class-folder image dataset and does not ship
    segmentation masks. This preparation step copies images into the
    normalized directory, maps folder labels into the AquaTrash 8-class
    label space, and uses SAM to generate one polygon mask per image.
    """

    root_dir = resolve_existing_dir(root_dir)
    dist_dir = os.path.join(OUTPUT_DIR, "realwaste")
    os.makedirs(dist_dir, exist_ok=True)

    aquatrash_categories = load_aquatrash_categories()
    aquatrash_label_to_id = {
        category["name"]: category["id"]
        for category in aquatrash_categories
    }

    image_paths = []
    for class_name in sorted(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        resolve_realwaste_label(class_name)
        for file_name in sorted(os.listdir(class_dir)):
            extension = os.path.splitext(file_name)[1].lower()
            if extension in IMAGE_EXTENSIONS:
                image_paths.append((class_name, os.path.join(class_dir, file_name)))

    if not image_paths:
        raise FileNotFoundError(f"No RealWaste images found under '{root_dir}'.")

    cache_path = os.path.join(dist_dir, REALWASTE_SAM_CACHE_NAME)
    sam_cache = load_sam_cache(cache_path)
    sam_model = None
    images = []
    annotations = []
    fallback_masks = 0

    print(f"Preparing {len(image_paths)} RealWaste images from '{root_dir}'.")
    for image_index, (class_name, image_path) in enumerate(image_paths, start=1):
        source_file_name = os.path.basename(image_path)
        new_file_name = f"{safe_file_component(class_name)}_{source_file_name.replace(' ', '_')}"
        new_image_path = os.path.join(dist_dir, new_file_name)

        width, height = read_image_size(image_path)
        cache_key = os.path.relpath(image_path, root_dir).replace("\\", "/")
        cached_mask = sam_cache.get(cache_key)

        if (
            cached_mask is not None
            and int(cached_mask.get("width", width)) == width
            and int(cached_mask.get("height", height)) == height
        ):
            segmentation = cached_mask["segmentation"]
            used_sam = bool(cached_mask.get("used_sam", True))
        else:
            if sam_model is None:
                sam_model = load_sam_model()
            segmentation, used_sam = generate_sam_segmentation(sam_model, image_path, width, height)
            sam_cache[cache_key] = {
                "segmentation": segmentation,
                "used_sam": used_sam,
                "width": width,
                "height": height,
            }
            if image_index % 25 == 0:
                save_sam_cache(cache_path, sam_cache)

        if not used_sam:
            fallback_masks += 1

        shutil.copy2(image_path, new_image_path)

        image_id = len(images) + 1
        annotation_id = len(annotations) + 1
        label = resolve_realwaste_label(class_name)
        category_id = aquatrash_label_to_id[label]
        area = polygon_area(segmentation)
        bbox = bbox_from_segmentation(segmentation)

        images.append(
            {
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": new_file_name,
                "source_file_name": cache_key,
            }
        )
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [segmentation],
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
            }
        )

        if image_index % 100 == 0:
            print(f"RealWaste SAM masks prepared {image_index}/{len(image_paths)}")

    save_sam_cache(cache_path, sam_cache)

    annotations_payload = {
        "images": images,
        "annotations": annotations,
        "categories": aquatrash_categories,
        "info": {
            "source": "RealWaste",
            "sam_model": REALWASTE_SAM_MODEL,
            "sam_cache": cache_path,
            "fallback_masks": fallback_masks,
        },
    }
    dist_annotations_path = os.path.join(dist_dir, "annotations.json")
    with open(dist_annotations_path, "w") as f:
        json.dump(annotations_payload, f)

    print(
        f"Prepared RealWaste dataset at '{dist_dir}' with {len(images)} images, "
        f"{len(annotations)} SAM-generated annotations, and {fallback_masks} fallback masks."
    )




def prepare_dataset(dataset_name: str) -> None:
    """Prepare the specified dataset for merging.

    This function dispatches to the appropriate preparation function based
    on the provided dataset name.

    Args:
        dataset_name: The name of the dataset to prepare.
    """

    dataset_name = validate_dataset(dataset_name)

    if dataset_name == "taco":
        prepare_taco(root_dir="data/raw/taco")
    elif dataset_name == "aquatrash":
        # Call the preparation function for Aquatrash
        prepare_aquatrash(root_dir="data/raw/aquatrash", annotations_path="data/raw/aquatrash/annotations.json")
    elif dataset_name == "realwaste":
        prepare_realwaste(root_dir="data/raw/realwaste")





def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
      
    prepare_dataset(args.dataset)

	# TODO: implement dataset merging using `args.inputs`, writing to
	# `args.output`, respecting `args.format` and `args.force`.


if __name__ == "__main__":
	main()

