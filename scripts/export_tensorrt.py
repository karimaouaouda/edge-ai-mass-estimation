"""Export models to TensorRT engines for Jetson deployment.

Usage::

    python scripts/export_tensorrt.py --model models/weights/yolov8n-seg.pt --format engine
    python scripts/export_tensorrt.py --model models/weights/yolov8n-seg.pt --format onnx
"""

from __future__ import annotations

import argparse
import logging

from edge_ai_mass.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt model")
    parser.add_argument(
        "--format",
        choices=["onnx", "engine"],
        default="engine",
        help="Export format (onnx or TensorRT engine)",
    )
    parser.add_argument("--half", action="store_true", help="FP16 quantisation")
    parser.add_argument("--int8", action="store_true", help="INT8 quantisation")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    setup_logging()

    from ultralytics import YOLO

    model = YOLO(args.model)
    logger.info("Exporting %s → %s (half=%s, int8=%s)", args.model, args.format, args.half, args.int8)
    model.export(
        format=args.format,
        half=args.half,
        int8=args.int8,
        imgsz=args.imgsz,
    )
    logger.info("Export complete")


if __name__ == "__main__":
    main()
