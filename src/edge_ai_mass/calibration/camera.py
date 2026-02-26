"""Camera calibration using checkerboard or ArUco markers.

Computes intrinsic parameters (focal length, principal point, distortion) and
provides pixel-to-metric conversion factors needed for accurate depth scaling
and volume estimation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Stores intrinsic calibration data."""

    camera_matrix: np.ndarray       # 3x3
    dist_coeffs: np.ndarray         # 1x5 or 1x8
    image_size: tuple[int, int]     # (width, height)
    reprojection_error: float = 0.0
    pixel_to_m_at_1m: float = 0.0   # metres per pixel at 1 m depth

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "image_size": list(self.image_size),
            "reprojection_error": self.reprojection_error,
            "pixel_to_m_at_1m": self.pixel_to_m_at_1m,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Calibration saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> CalibrationResult:
        data = json.loads(Path(path).read_text())
        return cls(
            camera_matrix=np.array(data["camera_matrix"]),
            dist_coeffs=np.array(data["dist_coeffs"]),
            image_size=tuple(data["image_size"]),
            reprojection_error=data.get("reprojection_error", 0.0),
            pixel_to_m_at_1m=data.get("pixel_to_m_at_1m", 0.0),
        )


def calibrate_checkerboard(
    image_paths: list[str | Path],
    board_size: tuple[int, int] = (9, 6),
    square_size_m: float = 0.025,
) -> CalibrationResult:
    """Run OpenCV checkerboard calibration on a set of images.

    Parameters
    ----------
    image_paths : list of paths to calibration images
    board_size  : inner corner counts (cols, rows)
    square_size_m : physical size of one square in metres
    """
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : board_size[0], 0 : board_size[1]].T.reshape(-1, 2)
    objp *= square_size_m

    image_size: tuple[int, int] | None = None

    for p in image_paths:
        img = cv2.imread(str(p))
        if img is None:
            logger.warning("Could not read %s — skipping", p)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        found, corners = cv2.findChessboardCorners(gray, board_size, None)
        if not found:
            logger.warning("No corners found in %s — skipping", p)
            continue

        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
        )
        obj_points.append(objp)
        img_points.append(corners)

    if not obj_points:
        raise ValueError("No valid calibration images found")

    rms, cam_mtx, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None
    )
    fx = cam_mtx[0, 0]
    pixel_to_m = 1.0 / fx  # at 1 m depth, 1 pixel ≈ 1/fx metres

    logger.info("Calibration RMS error: %.4f px", rms)
    return CalibrationResult(
        camera_matrix=cam_mtx,
        dist_coeffs=dist,
        image_size=image_size,
        reprojection_error=rms,
        pixel_to_m_at_1m=pixel_to_m,
    )
