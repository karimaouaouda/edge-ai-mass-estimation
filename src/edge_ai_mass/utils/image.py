"""Image utility functions shared across modules."""

from __future__ import annotations

import cv2
import numpy as np


def resize_keep_aspect(
    image: np.ndarray, target_size: int, interpolation: int = cv2.INTER_LINEAR
) -> tuple[np.ndarray, float]:
    """Resize so the longest side equals *target_size*, preserving aspect ratio.

    Returns the resized image and the scale factor applied.
    """
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    return resized, scale


def letterbox(
    image: np.ndarray,
    target_size: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize with letterbox padding (used by YOLO models).

    Returns (padded_image, scale, (pad_w, pad_h)).
    """
    h, w = image.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_w = (tw - new_w) // 2
    pad_h = (th - new_h) // 2
    padded = cv2.copyMakeBorder(
        resized, pad_h, th - new_h - pad_h, pad_w, tw - new_w - pad_w,
        cv2.BORDER_CONSTANT, value=color,
    )
    return padded, scale, (pad_w, pad_h)
