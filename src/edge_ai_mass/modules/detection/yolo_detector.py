"""YOLOv8 object detector with instance segmentation support.

Wraps Ultralytics YOLOv8 and supports both PyTorch and TensorRT engines.
Returns ``Detection`` objects consumed by the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from edge_ai_mass.modules.base import BaseModule, ModuleResult
from edge_ai_mass.pipeline.pipeline import Detection

logger = logging.getLogger(__name__)

# Default waste-category mapping (aligned with TACO super-categories)
DEFAULT_CLASSES = {
    0: "plastic",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "cardboard",
    5: "organic",
    6: "textile",
    7: "wood",
    8: "other",
}


class YOLODetector(BaseModule):
    """YOLOv8-nano / YOLOv8-seg detector for waste objects."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.model_path: str = config.get("model_path", "yolov8n-seg.pt")
        self.conf_threshold: float = config.get("conf_threshold", 0.25)
        self.iou_threshold: float = config.get("iou_threshold", 0.45)
        self.img_size: int = config.get("img_size", 640)
        self.task: str = str(config.get("task", "segment"))
        self.class_map: dict[int, str] | None = config.get("class_map")
        self._model: Any = None

    def load(self) -> None:
        from ultralytics import YOLO

        print(f"Loading YOLO {self.task} model from {self.model_path}")
        logger.info("Loading YOLO %s model from %s", self.task, self.model_path)
        self._model = YOLO(self.model_path, task=self.task)
        model_task = str(getattr(self._model, "task", self.task))
        if self.task == "segment" and model_task != "segment":
            raise RuntimeError(
                f"YOLO model {self.model_path!r} loaded as task {model_task!r}, "
                "but segmentation was configured"
            )
        self._is_loaded = True

    def _forward(self, image: np.ndarray, **kwargs: Any) -> ModuleResult:
        results = self._model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            verbose=False,
        )
        detections: list[Detection] = []
        image_h, image_w = image.shape[:2]
        for r in results:
            boxes = r.boxes
            masks = r.masks
            names = self.class_map or getattr(r, "names", None) or getattr(self._model, "names", None)
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                mask = None
                if masks is not None and i < len(masks.data):
                    mask = (masks.data[i].cpu().numpy() > 0.5).astype(np.uint8)
                    if mask.shape[:2] != (image_h, image_w):
                        mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
                detections.append(
                    Detection(
                        bbox=box.xyxy[0].cpu().numpy(),
                        mask=mask,
                        class_id=cls_id,
                        class_name=_class_name(names, cls_id),
                        confidence=float(box.conf[0]),
                    )
                )
        mask_count = sum(detection.mask is not None for detection in detections)
        if self.task == "segment" and detections and mask_count != len(detections):
            logger.warning(
                "YOLO segmentation model returned %d detections but only %d masks",
                len(detections),
                mask_count,
            )
        return ModuleResult(
            data=detections,
            metadata={
                "model_path": self.model_path,
                "model_task": str(getattr(self._model, "task", self.task)),
                "mask_count": mask_count,
            },
        )


def _class_name(names: Any, cls_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(cls_id, f"class_{cls_id}"))
    if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
        return str(names[cls_id])
    return DEFAULT_CLASSES.get(cls_id, f"class_{cls_id}")
