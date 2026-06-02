"""Command-line interface for the edge-ai-mass system.

Usage examples::

    # Run inference on a single image
    edge-ai-mass infer --image photo.jpg --config configs/pipeline/default.yaml

    # Run live camera demo
    edge-ai-mass demo --config configs/pipeline/jetson_nano.yaml --camera 0

    # Benchmark the pipeline
    edge-ai-mass benchmark --config configs/pipeline/default.yaml --iterations 100

    # Calibrate camera
    edge-ai-mass calibrate --images data/calibration/ --output configs/calibration.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import cv2
import numpy as np

from edge_ai_mass.utils.logging import setup_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="edge-ai-mass", description="Edge AI Mass Estimation")
    parser.add_argument("--log-level", default="INFO")
    sub = parser.add_subparsers(dest="command")

    # ── infer ──────────────────────────────────────────────────────
    p_infer = sub.add_parser("infer", help="Run inference on an image")
    p_infer.add_argument("--image", required=True)
    p_infer.add_argument("--config", default="configs/pipeline/default.yaml")
    p_infer.add_argument("--output", default=None, help="Path to save annotated image")

    # ── demo ───────────────────────────────────────────────────────
    p_demo = sub.add_parser("demo", help="Live camera demo")
    p_demo.add_argument("--config", default="configs/pipeline/default.yaml")
    p_demo.add_argument("--camera", type=int, default=0)

    # ── benchmark ──────────────────────────────────────────────────
    p_bench = sub.add_parser("benchmark", help="Benchmark pipeline latency")
    p_bench.add_argument("--config", default="configs/pipeline/default.yaml")
    p_bench.add_argument("--iterations", type=int, default=50)
    p_bench.add_argument("--warmup", type=int, default=5)

    # ── calibrate ──────────────────────────────────────────────────
    p_cal = sub.add_parser("calibrate", help="Camera calibration")
    p_cal.add_argument("--images", required=True, help="Directory of checkerboard images")
    p_cal.add_argument("--output", default="configs/calibration.json")
    p_cal.add_argument("--board-size", default="9,6")
    p_cal.add_argument("--square-size", type=float, default=0.025)

    args = parser.parse_args(argv)
    setup_logging(args.log_level)

    if args.command == "infer":
        _cmd_infer(args)
    elif args.command == "demo":
        _cmd_demo(args)
    elif args.command == "benchmark":
        _cmd_benchmark(args)
    elif args.command == "calibrate":
        _cmd_calibrate(args)
    else:
        parser.print_help()
        sys.exit(1)


# ------------------------------------------------------------------
# Sub-commands
# ------------------------------------------------------------------
def _cmd_infer(args: argparse.Namespace) -> None:
    from edge_ai_mass.pipeline.factory import build_pipeline

    pipeline = build_pipeline(args.config)
    pipeline.load_all()

    image = cv2.imread(args.image)
    if image is None:
        logging.error("Cannot read image: %s", args.image)
        sys.exit(1)

    result = pipeline.run(image)

    if not result.objects:
        print("No objects detected.")
        return

    for obj in result.objects:
        d = obj.detection
        print(
            f"  {d.class_name:12s}  conf={d.confidence:.2f}  "
            f"mass={obj.mass_kg:.4f} kg  method={obj.mass_method}"
        )

    print(f"\nTotal frame time: {result.frame_time_ms:.1f} ms")

    if args.output:
        annotated = _draw_results(image, result)
        cv2.imwrite(args.output, annotated)
        print(f"Saved annotated image to {args.output}")


def _cmd_demo(args: argparse.Namespace) -> None:
    from edge_ai_mass.pipeline.factory import build_pipeline

    pipeline = build_pipeline(args.config)
    pipeline.load_all()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logging.error("Cannot open camera %d", args.camera)
        sys.exit(1)

    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = pipeline.run(frame)
        annotated = _draw_results(frame, result)
        cv2.imshow("Edge AI Mass Estimation", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def _cmd_benchmark(args: argparse.Namespace) -> None:
    from edge_ai_mass.evaluation.benchmark import benchmark_pipeline
    from edge_ai_mass.pipeline.factory import build_pipeline

    pipeline = build_pipeline(args.config)
    pipeline.load_all()

    result = benchmark_pipeline(
        pipeline, num_warmup=args.warmup, num_iterations=args.iterations
    )
    print(json.dumps(result.summary(), indent=2))


def _cmd_calibrate(args: argparse.Namespace) -> None:
    from pathlib import Path

    from edge_ai_mass.calibration.camera import calibrate_checkerboard

    images = sorted(Path(args.images).glob("*.[jp][pn]g"))
    board = tuple(int(x) for x in args.board_size.split(","))
    cal = calibrate_checkerboard(images, board_size=board, square_size_m=args.square_size)
    cal.save(args.output)
    print(f"Calibration saved to {args.output}  (RMS error = {cal.reprojection_error:.4f})")


# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------
def _draw_results(image: np.ndarray, result) -> np.ndarray:
    """Draw bounding boxes and mass labels on the image."""
    out = image.copy()
    for obj in result.objects:
        d = obj.detection
        x1, y1, x2, y2 = d.bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d.class_name} {obj.mass_kg:.3f}kg"
        cv2.putText(out, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    fps_label = f"{1000 / max(result.frame_time_ms, 1):.1f} FPS"
    cv2.putText(out, fps_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return out


if __name__ == "__main__":
    main()
