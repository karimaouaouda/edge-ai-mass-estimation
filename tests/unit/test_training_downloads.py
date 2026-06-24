"""Contracts for resumable dataset acquisition without external network calls."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from edge_ai_mass.training.downloads import download_taco_dataset


def test_download_taco_dataset_from_coco_urls_and_resume(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_images = []
    for index in range(2):
        path = source / f"source-{index}.jpg"
        Image.new("RGB", (16, 12), color=(index * 50, 20, 30)).save(path)
        source_images.append(path)

    annotations = source / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": index + 1,
                        "file_name": f"batch_1/image-{index}.jpg",
                        "flickr_url": path.as_uri(),
                    }
                    for index, path in enumerate(source_images)
                ],
                "annotations": [],
                "categories": [],
            }
        ),
        encoding="utf-8",
    )

    destination = tmp_path / "taco"
    first = download_taco_dataset(
        destination,
        annotations_url=annotations.as_uri(),
        max_workers=2,
        retries=1,
        progress_every=0,
    )
    assert first["downloaded"] == 2
    assert first["failed"] == 0
    assert (destination / "batch_1" / "image-0.jpg").is_file()

    resumed = download_taco_dataset(
        destination,
        annotations_url=annotations.as_uri(),
        max_workers=2,
        retries=1,
        progress_every=0,
    )
    assert resumed["downloaded"] == 0
    assert resumed["existing"] == 2
