"""Tests for mass data-collection script utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.mass.create_mass_feature_row import build_feature_row, build_feature_rows_from_config
from scripts.mass.utils import depth_statistics, load_profiled_config, mask_geometry


def test_mask_geometry_uses_pixel_per_cm_scale():
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:6, 3:9] = 1

    geometry = mask_geometry(mask, pixel_per_cm=2.0)

    assert geometry.mask_area_px == 24
    assert geometry.bbox_width_px == 6
    assert geometry.bbox_height_px == 4
    assert geometry.area_cm2 == pytest.approx(6.0)
    assert geometry.bbox_width_cm == pytest.approx(3.0)
    assert geometry.bbox_height_cm == pytest.approx(2.0)


def test_build_feature_row_outputs_mass_pipeline_columns(tmp_path: Path):
    mask_path = tmp_path / "mask.npy"
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:6, 3:9] = 1
    np.save(mask_path, mask)

    row = build_feature_row(
        image_id="img-1",
        object_id="obj-1",
        sample_id=None,
        class_name="plastic_bottle",
        material="plastic",
        mask_path=mask_path,
        pixel_per_cm=2.0,
        background_default_depth=90.0,
        camera_height=100.0,
        depth_unit="cm",
        thickness_mode="camera_minus_background",
        min_thickness_cm=0.0,
        density_priors_path=None,
        class_density_prior=["plastic_bottle=0.04"],
        density_prior_g_cm3=None,
        real_mass_g=20.0,
    )

    assert row["sample_id"] == "img-1"
    assert row["object_id"] == "obj-1"
    assert row["area_cm2"] == pytest.approx(6.0)
    assert row["estimated_thickness_cm"] == pytest.approx(10.0)
    assert row["volume_cm3"] == pytest.approx(60.0)
    assert row["mass_base_g"] == pytest.approx(2.4)
    assert row["correction_g"] == pytest.approx(17.6)
    assert row["estimated_volume_m3"] == pytest.approx(60.0e-6)
    assert row["effective_density_kg_m3"] == pytest.approx(40.0)


def test_feature_rows_can_be_built_from_config_profile(tmp_path: Path):
    mask_path = tmp_path / "mask.npy"
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:6, 3:9] = 1
    np.save(mask_path, mask)
    config_path = tmp_path / "feature_rows.yaml"
    config_path.write_text(
        f"""
active_profile: batch
output: {tmp_path / "rows.csv"}
defaults:
  pixel_per_cm: 2.0
  background_default_depth: 90.0
  camera_height: 100.0
  depth_unit: cm
  thickness_mode: camera_minus_background
class_density_priors_g_cm3:
  plastic_bottle: 0.04
profiles:
  batch:
    rows:
      - image_id: img-1
        object_id: obj-1
        class_name: plastic_bottle
        material: plastic
        mask: {mask_path}
        real_mass_g: 20.0
""",
        encoding="utf-8",
    )

    cfg, profile = load_profiled_config(config_path)
    rows = build_feature_rows_from_config(cfg)

    assert profile == "batch"
    assert len(rows) == 1
    assert rows[0]["volume_cm3"] == pytest.approx(60.0)
    assert rows[0]["correction_g"] == pytest.approx(17.6)


def test_depth_statistics_reports_valid_ratio():
    depth = np.array([[1.0, 2.0], [0.0, np.nan]], dtype=np.float32)

    stats = depth_statistics(depth)

    assert stats["depth_mean"] == pytest.approx(1.5)
    assert stats["depth_valid_ratio"] == pytest.approx(0.5)
