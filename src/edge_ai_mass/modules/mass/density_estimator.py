"""Density-based mass estimator: mass = volume x density.

Uses the depth map and object mask to estimate volume (via depth integration),
then multiplies by a material-specific density lookup.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from edge_ai_mass.modules.base import BaseModule, ModuleResult
from edge_ai_mass.modules.material import ClassMaterialMapper

logger = logging.getLogger(__name__)

# Average densities in kg/m^3 for common waste materials
# Sources: Minnesota PCA, EPA, and waste characterisation literature
MATERIAL_DENSITIES: dict[str, float] = {
    "plastic": 40.0,       # mixed loose plastic (PET ~1380 solid, but bottles are hollow)
    "glass": 300.0,        # glass containers (intact, with air)
    "metal": 160.0,        # aluminum cans / mixed metal containers
    "paper": 90.0,         # loose paper
    "cardboard": 60.0,     # corrugated cardboard
    "organic": 500.0,      # food waste
    "textile": 120.0,      # clothing, rags
    "wood": 250.0,         # wood scraps
    "other": 100.0,        # catch-all
}


class DensityMassEstimator(BaseModule):
    """Volume x density mass estimator.

    Features expected in kwargs["features"]:
        - mask: (H, W) binary mask
        - depth_stats: dict with at least "mean" in metres
        - class_name: str material category
        - bbox: (x1, y1, x2, y2)
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.densities: dict[str, float] = config.get("densities", MATERIAL_DENSITIES)
        self.pixel_to_m: float = config.get("pixel_to_m", 0.001)  # default calibration
        self.material_mapper = ClassMaterialMapper.from_config(config)

    def load(self) -> None:
        self._is_loaded = True  # no model to load — pure computation

    def _forward(self, image: np.ndarray, **kwargs: Any) -> dict[str, Any]:
        features: dict[str, Any] = kwargs["features"]
        depth_stats = features.get("depth_stats", {})
        class_name = features.get("class_name", "other")
        material = features.get("material") or self.material_mapper.material_for(class_name)
        mask = features.get("mask")
        bbox = features.get("bbox")
        warnings = list(features.get("warnings", []))

        geometry = features.get("geometry") or {}
        volume_m3 = features.get("volume_m3")
        volume_method = "geometry"
        if volume_m3 is None and isinstance(geometry, dict):
            volume_m3 = geometry.get("volume_m3")
            volume_method = geometry.get("method", "geometry")
        if volume_m3 is None:
            volume_m3 = self._estimate_volume(image, mask, bbox, depth_stats)
            volume_method = "legacy_depth_range"
            warnings.append("mass_used_legacy_volume_fallback")

        density = self.densities.get(material, self.densities["other"])
        mass_kg = volume_m3 * density

        return ModuleResult(
            data={
                "mass_kg": mass_kg,
                "volume_m3": volume_m3,
                "volume_method": volume_method,
                "material": material,
                "density_used": density,
                "warnings": warnings,
            },
            metadata={"method": "density"},
        )

    def _estimate_volume(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        bbox: np.ndarray | None,
        depth_stats: dict[str, float],
    ) -> float:
        """Estimate volume by integrating depth over the object's projected area.

        For each pixel inside the mask, the depth value represents the distance
        from the camera.  We approximate volume as:

            V ≈ pixel_area_m2 * sum(depth_values - background_depth)

        When no mask is available, fall back to bbox area * mean depth.
        """
        mean_depth = depth_stats.get("mean", 0.0)
        if mean_depth <= 0:
            return 0.0

        if mask is not None and mask.any():
            pixel_count = int(np.sum(mask > 0))
        elif bbox is not None:
            x1, y1, x2, y2 = bbox.astype(int)
            pixel_count = max((x2 - x1) * (y2 - y1), 1)
        else:
            return 0.0

        pixel_area_m2 = self.pixel_to_m ** 2
        # Approximate: assume object "thickness" is proportional to depth range
        depth_range = depth_stats.get("max", mean_depth) - depth_stats.get("min", mean_depth)
        thickness = max(depth_range, 0.01)
        volume = pixel_count * pixel_area_m2 * thickness
        return volume
