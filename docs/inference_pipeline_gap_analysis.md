# Inference Pipeline Gap Analysis

Date: 2026-06-18

Scope: `src/edge_ai_mass/`, pipeline configs, mass/depth/detection modules, calibration helpers, and supporting scripts/tests.

Implementation update: the first runtime slice has been implemented. The pipeline now has config-driven calibration/depth-scale/background settings, a geometry estimator, background-aware volume integration, density mass from geometry volume, safer regression checkpoint behavior, model-native YOLO class names, mask resizing, JSON inference output, and synthetic tests for the new geometry path.

## Target Runtime Workflow

The desired runtime flow is:

1. Capture or receive an RGB image from a calibrated fixed camera.
2. Detect, segment, and classify each trash item, for example `plastic_bottle`.
3. Run Depth Anything or the configured depth model on the same image.
4. Convert model depth into real-world metric depth using camera calibration, fixed-camera geometry, and a known background or reference plane.
5. Fuse each object mask with metric depth and background depth to recover object-level 3D geometry.
6. Estimate physical dimensions, projected area, object height/thickness, and volume.
7. Estimate mass from volume, material or object class, density priors, and optionally a learned regression/correction model.
8. Return per-object outputs and diagnostics: class, confidence, mask, bbox, depth stats, dimensions, volume, mass, method, and latency.

## Executive Verdict

The repository already has a partial cascade: YOLO detection/segmentation, Depth Anything/MiDaS depth, object-level depth statistics, and two mass estimators. It is not yet a complete calibrated 3D mass-estimation pipeline.

The current implementation can run a rough detection -> depth-stats -> mass estimate path, but the volume and mass values are approximate placeholders unless a trained mass regression checkpoint and calibration-aware geometry layer are added.

## Current Implementation Status

| Stage | Current status | Evidence | Notes |
|---|---|---|---|
| Pipeline orchestration | Partial | `src/edge_ai_mass/pipeline/pipeline.py:139` | Runs detection, depth, then mass per detection. No explicit calibration, background, geometry, or material stages. |
| Config-driven modules | Implemented | `src/edge_ai_mass/pipeline/factory.py:29` | Stages are dynamically loaded from YAML. Good base for adding stages. |
| Detection and instance segmentation | Partial | `src/edge_ai_mass/modules/detection/yolo_detector.py:52` | Ultralytics YOLO returns bboxes, masks, class IDs, confidence. Class map is generic material-like classes and not aligned with `plastic_bottle` object taxonomy or one-class detection mode. |
| Fallback detection | Partial | `src/edge_ai_mass/modules/detection/mobilenet_detector.py` | Bbox-only fallback exists, but no segmentation fallback/refinement for volume. |
| Depth model | Partial | `src/edge_ai_mass/modules/depth/depth_anything.py:24` | Loads HuggingFace Depth Anything and MiDaS. Output is treated as depth, but no repo-level metric calibration or depth scale correction is applied. |
| Depth-object fusion | Partial | `src/edge_ai_mass/pipeline/pipeline.py:200` | Computes mean, median, std, min, max inside mask or bbox. Does not produce metric dimensions, point cloud, background-relative height, or volume. |
| Camera calibration | Partial | `src/edge_ai_mass/calibration/camera.py:55` | Checkerboard intrinsic calibration exists. It is not loaded or used by the runtime pipeline. No extrinsics, table plane, background depth, ArUco/reference scaling, or depth calibration model. |
| Volume estimation | Placeholder | `src/edge_ai_mass/modules/mass/density_estimator.py:70` | Uses constant `pixel_to_m` and depth range as thickness. It does not integrate object height against a calibrated background. |
| Mass estimation | Partial | `src/edge_ai_mass/modules/mass/regression_estimator.py:47` | Regression head exists, but can silently run untrained if checkpoint is missing. Density fallback exists but depends on placeholder volume. |
| Material classification | Missing | `src/edge_ai_mass/modules/material/__init__.py` | No standalone material classifier. Current mass estimator uses detector `class_name` as material. |
| Segmentation refinement | Missing | `src/edge_ai_mass/modules/segmentation/__init__.py` | No SAM/refinement module in runtime pipeline. |
| CLI inference output | Partial | `src/edge_ai_mass/cli.py:93` | Prints class, confidence, mass, and method. Does not expose volume, depth stats, geometry, calibration version, or JSON output. |
| Feature collection for mass regression | Partial | `scripts/collect_mass_features.py:34` | Extracts bbox, mask area, depth stats, and mass labels. Does not include calibrated metric geometry or volume. |
| Tests | Partial | `tests/unit/test_pipeline.py:15` | Covers crop and depth stats. Missing calibration, geometry, volume, detector-mask alignment, and end-to-end pipeline contract tests. |

## Important Gaps

### 1. Calibration is not part of runtime inference

The code can create a checkerboard calibration artifact, but pipeline config does not load it and the pipeline does not undistort images, use `fx/fy/cx/cy`, or convert pixel measurements at object depth to metric area.

Needed:

- `calibration_path` in pipeline YAML.
- Runtime loader for camera intrinsics and distortion coefficients.
- Optional image undistortion before detection/depth.
- Metric projection helpers:
  - pixel area at depth: approximately `z^2 / (fx * fy)`;
  - 3D point projection: `X = (u - cx) * z / fx`, `Y = (v - cy) * z / fy`, `Z = z`.

### 2. Depth Anything output is not calibrated to real-world scale

`DepthAnythingModule` returns the model output resized to the input image. The code comments call it metric depth, but there is no validation or scale correction. Depth Anything variants may be relative or metric depending on checkpoint and training. The runtime must not assume reliable meters without calibration.

Needed:

- Depth model metadata: `relative`, `metric`, or `inverse_depth`.
- Depth scale/calibration component using known reference object, ArUco marker, background plane, or measured fixed-camera scene.
- Conversion from raw depth to metric depth with recorded scale, offset, and confidence.
- Tests that verify the sign convention: object closer to camera should have smaller distance than background for normal depth.

### 3. Background-aware volume is not implemented

The target workflow depends on the background. Current volume is:

```text
pixel_count * pixel_to_m^2 * max(depth_max - depth_min, 0.01)
```

That is a rough placeholder. It does not use background depth, camera intrinsics, object mask shape, or per-pixel height.

Needed:

- Background capture or background-depth artifact for the fixed camera.
- Optional plane/table model when background is flat.
- For each object mask:
  - align mask to metric depth;
  - compute object depth values;
  - compute background depth for the same pixels;
  - compute object height/thickness as `background_depth - object_depth` or the configured signed equivalent;
  - reject invalid, negative, and noisy pixels;
  - integrate volume as `sum(pixel_area_m2(z) * object_height_m)`.

### 4. Object class and material are conflated

The workflow needs a trash item class such as `plastic_bottle`, while mass needs material/density such as `plastic`. The current `YOLODetector.DEFAULT_CLASSES` are broad materials (`plastic`, `glass`, `metal`, etc.). That is not enough to model object shape priors, hollow objects, or bottle/can/cardboard-specific density factors.

Needed:

- Decide runtime taxonomy:
  - object classes: `plastic_bottle`, `metal_can`, `glass_bottle`, etc.;
  - material classes: `plastic`, `metal`, `glass`, `paper`, etc.;
  - one-class `trash` detection only when paired with a material/object classifier.
- Add class-to-material and class-to-shape-prior mapping.
- Add a material classifier module if detector classes are one-class or too coarse.

### 5. Regression mass estimator can run untrained

`RegressionMassEstimator.load()` warns when no checkpoint exists, but still runs a randomly initialized network. That can produce plausible-looking but meaningless mass values.

Needed:

- Config option such as `require_checkpoint: true`.
- Fall back to density estimator when regression checkpoint is missing.
- Include calibration and geometry features in the regression schema.
- Save feature schema/version with the checkpoint.

### 6. Mass method metadata is not set by modules

`Pipeline.run()` reads `mass_result.metadata["method"]`, but `BaseModule.predict()` creates empty metadata and neither mass estimator sets method metadata. Current CLI can show `unknown`.

Needed:

- Let module `_forward()` return richer `ModuleResult`, or allow modules to define metadata.
- Set `method = density`, `regression`, or `hybrid`.

### 7. Runtime outputs are too small for debugging and evaluation

The CLI only prints class, confidence, mass, and method. A physical-estimation pipeline needs structured diagnostics.

Needed:

- JSON output mode with:
  - bbox, class, confidence;
  - mask area;
  - depth stats;
  - metric dimensions;
  - volume;
  - mass;
  - calibration/background artifact IDs;
  - stage latencies.
- Optional debug artifacts: annotated image, depth image, mask overlay, per-object geometry summary.

## Recommended Implementation Tasks

### Phase 1: Define Runtime Contracts

- [x] Add explicit dataclasses for geometry outputs:
  - `RuntimeCalibration`;
  - `DepthScaleConfig`;
  - `ObjectGeometry`;
  - `MassEstimateDetails`.
- [x] Extend `ObjectEstimate` with:
  - `geometry`;
  - `volume_method`;
  - `calibration_id`;
  - `background_id`;
  - `warnings`.
- [x] Add pipeline config keys:
  - `calibration.path`;
  - `background.depth_path` or `background.capture_path`;
  - `depth.output_type`;
  - `depth_scale.output_type`, `depth_scale.scale`, and `depth_scale.offset`;
  - `geometry.height_mode`;
  - `mass.params.class_to_material`;
  - `mass.require_checkpoint`.

### Phase 2: Make Detection, Segmentation, and Classification Explicit

- [x] Update `YOLODetector` to use model-native names when available, with config override.
- [x] Support both:
  - segmentation models that return masks directly;
  - detection-only models that require a segmentation refinement/fallback.
- [x] Add a mask-alignment helper that guarantees masks are aligned to original image dimensions.
- [x] Add `ClassMaterialMapper` or a material classifier module.
- [ ] Add tests for:
  - segmented detection output;
  - detection-only output;
  - one-class `trash` plus classifier mapping;
  - class-to-density lookup.

### Phase 3: Calibrate Depth into Metric Space

- [x] Add `edge_ai_mass.calibration.runtime` helpers to load calibration JSON.
- [ ] Add image undistortion/preprocessing using intrinsics and distortion coefficients.
- [x] Add a depth calibration module:
  - raw depth to metric depth;
  - scale/offset correction;
  - optional inverse-depth conversion;
  - confidence/valid masks.
- [ ] Add a calibration command or script for reference-object/depth-scale fitting.
- [ ] Record calibration metadata in inference results.

### Phase 4: Implement Background-Aware Geometry

- [x] Add a `geometry` module, for example `src/edge_ai_mass/modules/geometry/volume_estimator.py`.
- [x] Add background artifact support:
  - background metric depth file;
  - constant background depth fallback;
  - background ID.
- [ ] Add richer background capture support:
  - background RGB;
  - table/plane model;
  - capture timestamp.
- [x] Compute per-object geometry:
  - real-world bbox width/height;
  - mask projected area in square meters;
  - object height/thickness percentiles;
  - volume in cubic meters;
  - quality flags for invalid depth, poor mask, or background mismatch.
- [x] Replace placeholder volume in `DensityMassEstimator` with geometry-provided volume.
- [x] Add synthetic tests where a known box over a background plane produces expected volume.

### Phase 5: Upgrade Mass Estimation

- [x] Make density estimator consume `geometry.volume_m3`.
- [ ] Move density priors to a versioned config file.
- [ ] Add object-shape priors for hollow/crushed items:
  - plastic bottle;
  - metal can;
  - glass bottle/jar;
  - paper/cardboard.
- [x] Enforce regression checkpoint loading or automatic density fallback.
- [ ] Add a hybrid estimator:
  - `mass = density_volume_mass + learned_residual(features)`;
  - or a gated ensemble between density and regression.
- [x] Update `scripts/collect_mass_features.py` to collect calibrated geometry and volume features.

### Phase 6: Expose Professional Inference Outputs

- [x] Add `edge-ai-mass infer --json output.json`.
- [x] Include full per-object metrics and warnings in JSON.
- [ ] Add debug artifact options:
  - depth visualization;
  - mask overlay;
  - background subtraction/height map;
  - object geometry report.
- [ ] Update demo overlay to show class, confidence, volume, mass, and warning state.

### Phase 7: Evaluation and Promotion Gates

- [ ] Build fixture tests using fake detector/depth/mass modules.
- [x] Add unit tests for:
  - raw-to-metric depth conversion;
  - mask/depth alignment;
  - background-aware volume integration;
  - density mass estimation.
- [ ] Add tests for:
  - calibration JSON file loading;
  - hybrid mass estimation.
- [x] Add integration tests for the full cascade with synthetic depth/background data.
- [ ] Add metrics:
  - detection mAP;
  - mask mAP;
  - depth scale error;
  - dimension MAE;
  - volume MAE/MAPE;
  - mass MAE/RMSE/MAPE;
  - latency/FPS/memory.
- [ ] Define promotion gates before Jetson deployment.

## Suggested Code Structure

```text
src/edge_ai_mass/
  calibration/
    camera.py
    depth_scale.py              # new
    runtime.py                  # new
  modules/
    geometry/
      __init__.py               # new
      volume_estimator.py       # new
    material/
      classifier.py             # new if one-class detector is used
    mass/
      density_estimator.py      # update to consume geometry volume
      hybrid_estimator.py       # new
  pipeline/
    pipeline.py                 # extend result contracts and stages
    factory.py                  # support calibration/background config
configs/
  pipeline/
    default.yaml                # add calibration/background/geometry keys
  calibration/
    camera_default.json         # generated, not hand-authored
  priors/
    material_densities.yaml     # new
    object_shape_priors.yaml    # new
```

## Minimal Next Implementation Slice

The best first implementation slice is:

1. Add runtime calibration loading and a `GeometryEstimator`.
2. Use existing YOLO masks and existing depth maps.
3. Add a simple background-depth artifact.
4. Compute `volume_m3` from calibrated per-pixel background/object depth.
5. Make `DensityMassEstimator` consume that volume.
6. Add tests with synthetic mask, depth, background, and known camera intrinsics.

This slice gives the pipeline its missing physical core without changing the training notebooks or model export flow.

## Key Risks

- Monocular depth may not be metric without scale correction.
- Transparent or reflective trash can break both depth and segmentation.
- Hollow objects need object-specific density or shape priors.
- A one-class detector cannot estimate density without a separate classifier or metadata source.
- Background-based volume depends on a stable camera, stable surface, and repeatable lighting.
- Randomly initialized regression output must never be treated as a valid mass estimate.
