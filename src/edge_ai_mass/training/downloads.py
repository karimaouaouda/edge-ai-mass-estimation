"""Dataset acquisition helpers used by local and hosted training jobs."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen
from uuid import uuid4


TACO_ANNOTATIONS_URL = (
    "https://huggingface.co/datasets/karimaouaouda/taco/resolve/main/annotations.json"
)


def _download_atomic(
    url: str,
    destination: Path,
    *,
    timeout_seconds: float,
    retries: int,
    validator: Callable[[bytes], None] | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.part")
        try:
            request = Request(url, headers={"User-Agent": "edge-ai-mass-training/1.0"})
            with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                payload = response.read()
            if not payload:
                raise ValueError("download returned an empty response")
            if validator is not None:
                validator(payload)
            temporary.write_bytes(payload)
            temporary.replace(destination)
            return
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(f"Failed to download {url!r} after {retries} attempts: {last_error}")


def _validate_image(payload: bytes) -> None:
    from PIL import Image

    with Image.open(BytesIO(payload)) as image:
        image.verify()


def _safe_relative_path(file_name: str) -> Path:
    relative = Path(file_name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe TACO image path in annotations: {file_name!r}")
    return relative


def _load_coco(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("images"), list) or not isinstance(
        payload.get("annotations"), list
    ):
        raise ValueError(f"TACO annotations are not valid COCO JSON: {path}")
    return payload


def download_taco_dataset(
    output_dir: str | Path,
    *,
    annotations_url: str = TACO_ANNOTATIONS_URL,
    max_workers: int = 8,
    timeout_seconds: float = 30,
    retries: int = 3,
    max_failures: int = 0,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Download TACO from its Hugging Face COCO JSON and referenced image URLs.

    Existing valid images are reused, downloads are written atomically, and a
    machine-readable summary is always saved. By default, any unavailable image
    fails the acquisition stage so training cannot silently use partial data.
    """
    output_dir = Path(output_dir)
    retries = max(1, int(retries))
    output_dir.mkdir(parents=True, exist_ok=True)
    annotations_path = output_dir / "annotations.json"
    if not annotations_path.exists():
        _download_atomic(
            annotations_url,
            annotations_path,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )

    coco = _load_coco(annotations_path)
    images = coco["images"]
    if not images:
        raise ValueError(f"TACO annotations contain no images: {annotations_path}")

    def acquire(image: dict[str, Any]) -> dict[str, str]:
        file_name = str(image.get("file_name", ""))
        if not file_name:
            return {"status": "failed", "file_name": file_name, "error": "missing file_name"}
        try:
            destination = output_dir / _safe_relative_path(file_name)
        except ValueError as exc:
            return {"status": "failed", "file_name": file_name, "error": str(exc)}
        try:
            if destination.exists():
                _validate_image(destination.read_bytes())
                return {"status": "existing", "file_name": file_name}
        except Exception:
            destination.unlink(missing_ok=True)

        urls = [
            str(image[key])
            for key in ("flickr_url", "flickr_640_url", "coco_url")
            if image.get(key)
        ]
        if not urls:
            return {"status": "failed", "file_name": file_name, "error": "no image URL"}
        errors: list[str] = []
        for url in urls:
            try:
                _download_atomic(
                    url,
                    destination,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                    validator=_validate_image,
                )
                return {"status": "downloaded", "file_name": file_name, "url": url}
            except Exception as exc:
                errors.append(str(exc))
        return {
            "status": "failed",
            "file_name": file_name,
            "error": " | ".join(errors),
        }

    counts = {"downloaded": 0, "existing": 0, "failed": 0}
    failures: list[dict[str, str]] = []
    worker_count = max(1, int(max_workers))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(acquire, image) for image in images]
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            counts[result["status"]] += 1
            if result["status"] == "failed":
                failures.append(result)
            if progress_every and completed % progress_every == 0:
                print(
                    f"TACO images checked {completed}/{len(images)} | "
                    f"downloaded={counts['downloaded']} existing={counts['existing']} "
                    f"failed={counts['failed']}"
                )

    summary: dict[str, Any] = {
        "annotations_url": annotations_url,
        "annotations_path": str(annotations_path),
        "annotations_sha256": hashlib.sha256(annotations_path.read_bytes()).hexdigest(),
        "annotation_count": len(coco["annotations"]),
        "category_count": len(coco.get("categories", [])),
        "images_root": str(output_dir),
        "referenced_images": len(images),
        **counts,
        "available": counts["downloaded"] + counts["existing"],
        "failed_examples": sorted(failures, key=lambda item: item["file_name"])[:20],
    }
    summary_path = output_dir / "taco_download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2)[:4000])
    if counts["failed"] > max_failures:
        raise RuntimeError(
            f"TACO acquisition failed for {counts['failed']} images; allowed={max_failures}. "
            f"See {summary_path} and rerun to resume successful downloads."
        )
    return summary


__all__ = ["TACO_ANNOTATIONS_URL", "download_taco_dataset"]
