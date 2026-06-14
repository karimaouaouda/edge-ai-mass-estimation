import argparse
import os
import random
import shutil
import json

from typing import Sequence
from prepare_datasets_to_merge import prepare_dataset

datasets = ["taco", "aquatrash"]



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
		"--visualize-only",
		action="store_true",
		help="Visualize samples from the dataset",
	)
	return parser.parse_args(argv)


DIST_DIR = "data/normalized"


YOLO_READY_DIR = "data/yolo_ready"

TEST_RATE = 0.2
VAL_RATE = 0.1
TRAIN_RATE = 0.7


def visualize_samples(samples: int, image_dir: str, label_dir: str, with_annotations: bool = True) -> None:
    """Visualize a few samples from the dataset.

    This function should implement the logic to visualize a few samples from the dataset,
    optionally with annotations.

    Args:
        samples: The number of samples to visualize.
        image_dir: The directory containing the images.
        label_dir: The directory containing the labels.
        with_annotations: Whether to visualize the annotations on the images.
    """
    if samples <= 0:
        return

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.patches import Polygon

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = []
    with os.scandir(image_dir) as entries:
        for entry in random.sample(list(entries), min(samples * 2, len(os.listdir(image_dir)))):
            if entry.is_file() and os.path.splitext(entry.name)[1].lower() in image_extensions:
                image_paths.append(entry.path)
                if len(image_paths) >= samples:
                    break

    if not image_paths:
        print(f"No images found in '{image_dir}'.")
        return

    cols = min(3, len(image_paths))
    rows = (len(image_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)

    for ax in axes.flat:
        ax.axis("off")

    for ax, image_path in zip(axes.flat, image_paths):
        image = mpimg.imread(image_path)
        height, width = image.shape[:2]
        ax.imshow(image)
        ax.set_title(os.path.basename(image_path), fontsize=9)

        if not with_annotations:
            continue

        label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(image_path))[0] + ".txt")
        if not os.path.exists(label_path):
            continue

        with open(label_path, "r") as label_file:
            for line in label_file:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue

                coord_count = (len(parts) - 1) // 2 * 2
                try:
                    coords = [float(value) for value in parts[1:1 + coord_count]]
                except ValueError:
                    continue

                points = [
                    (
                        max(0.0, min(1.0, coords[index])) * width,
                        max(0.0, min(1.0, coords[index + 1])) * height,
                    )
                    for index in range(0, len(coords), 2)
                ]
                if len(points) < 3:
                    continue

                ax.add_patch(Polygon(points, closed=True, fill=False, edgecolor="yellow", linewidth=1.5))
                ax.text(
                    points[0][0],
                    points[0][1],
                    parts[0],
                    color="black",
                    fontsize=8,
                    bbox={"facecolor": "yellow", "edgecolor": "none", "pad": 1},
                )

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    args = parse_args()
    # 1 - create yolo folders for images/{train, val, test} and labels/{train, val, test}
    os.makedirs(os.path.join(YOLO_READY_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_READY_DIR, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_READY_DIR, "images", "test"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_READY_DIR, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_READY_DIR, "labels", "val"), exist_ok=True)
    os.makedirs(os.path.join(YOLO_READY_DIR, "labels", "test"), exist_ok=True)

    if args.visualize_only:
        visualize_samples(samples=20, image_dir=os.path.join(YOLO_READY_DIR, "images", "train"), label_dir=os.path.join(YOLO_READY_DIR, "labels", "train"))
        exit()

    for dataset_name in datasets:
        print(f"Preparing dataset: {dataset_name}")
        prepare_dataset(dataset_name=dataset_name)
    print(f"All datasets have been prepared and are ready for merging.")


    for dataset_name in datasets:
        dist_dir = os.path.join(DIST_DIR, dataset_name)

        files = [file_name for file_name in os.listdir(dist_dir) if not file_name.endswith(".json")]
        file_set = set(files)
        annotation_file = [os.path.join(dist_dir, file_name) for file_name in os.listdir(dist_dir) if file_name.endswith(".json")][0]

        with open(annotation_file, "r") as f:
            annotations = json.load(f)

        annotations_by_image = {}
        for annotation in annotations["annotations"]:
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

        label_content_by_file = {}
        labeled_files = []

        for image in annotations["images"]:
            file_name = image["file_name"]
            if file_name not in file_set:
                raise FileNotFoundError(f"Image file '{file_name}' not found in '{dist_dir}'. Please ensure the dataset is prepared correctly.")

            id = image["id"]
            image_annotations = annotations_by_image.get(id, [])
            if len(image_annotations) == 0:
                print(f"Warning: No annotations found for image '{file_name}' (id: {id}). This image will be skipped.")
                continue

            width = float(image["width"])
            height = float(image["height"])
            label_lines = []
            for annotation in image_annotations:
                for segmentation in annotation.get("segmentation", []):
                    if not isinstance(segmentation, list) or len(segmentation) < 6:
                        continue

                    points = []
                    for index in range(0, len(segmentation) - 1, 2):
                        x = max(0.0, min(1.0, float(segmentation[index]) / width))
                        y = max(0.0, min(1.0, float(segmentation[index + 1]) / height))
                        points.extend((f"{x:.6f}", f"{y:.6f}"))

                    if len(points) >= 6:
                        label_lines.append(f"{annotation['category_id']} " + " ".join(points))

            label_content_by_file[file_name] = "\n".join(label_lines)
            labeled_files.append(file_name)


        train_size = int(len(labeled_files) * TRAIN_RATE)
        val_size = int(len(labeled_files) * VAL_RATE)
        test_size = len(labeled_files) - train_size - val_size

        train_files = labeled_files[:train_size]
        val_files = labeled_files[train_size:train_size + val_size]
        test_files = labeled_files[train_size + val_size:]

        for file_name in train_files:
            old_image_path = os.path.join(dist_dir, file_name)
            new_image_path = os.path.join(YOLO_READY_DIR, "images", "train", file_name)
            shutil.copy2(old_image_path, new_image_path)
            label_path = os.path.join(YOLO_READY_DIR, "labels", "train", os.path.splitext(file_name)[0] + ".txt")
            with open(label_path, "w") as f:
                f.write(label_content_by_file[file_name])

        for file_name in val_files:
            old_image_path = os.path.join(dist_dir, file_name)
            new_image_path = os.path.join(YOLO_READY_DIR, "images", "val", file_name)
            shutil.copy2(old_image_path, new_image_path)
            label_path = os.path.join(YOLO_READY_DIR, "labels", "val", os.path.splitext(file_name)[0] + ".txt")
            with open(label_path, "w") as f:
                f.write(label_content_by_file[file_name])

        for file_name in test_files:
            old_image_path = os.path.join(dist_dir, file_name)
            new_image_path = os.path.join(YOLO_READY_DIR, "images", "test", file_name)
            shutil.copy2(old_image_path, new_image_path)
            label_path = os.path.join(YOLO_READY_DIR, "labels", "test", os.path.splitext(file_name)[0] + ".txt")
            with open(label_path, "w") as f:
                f.write(label_content_by_file[file_name])



        visualize_samples(samples=2, image_dir=os.path.join(YOLO_READY_DIR, "images", "train"), label_dir=os.path.join(YOLO_READY_DIR, "labels", "train"))

    

