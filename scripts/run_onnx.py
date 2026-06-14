from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


import cv2
from ultralytics import YOLO
from ultralytics.models.rtdetr import RTDETR


ROOT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT_DIR / "weights"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "runs" / "onnx"


def find_default_model(weights_dir: Path) -> Path:
	if not weights_dir.exists():
		raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

	onnx_files = sorted(weights_dir.glob("*.onnx"))
	if not onnx_files:
		raise FileNotFoundError(
			f"No ONNX model found in {weights_dir}. Put a .onnx file there or pass --model."
		)
	return onnx_files[0]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run ONNX model inference on image or video sources."
	)
	parser.add_argument(
		"--model",
		type=str,
		default=None,
		help="Path to ONNX model. If omitted, first .onnx file in /weights is used.",
	)

	# one for dir
	parser.add_argument(
		"--source-dir",
		type=str,
		help="Path to directory containing images for inference."
	)


	parser.add_argument(
		"--type",
		type=str,
		default=None,
		help="the type of model to run. If omitted, RTDETR is used by default. Options: yolo, rtdetr.",
	)
	parser.add_argument(
		"--mode",
		type=str,
		choices=["image", "video"],
		required=True,
		help="Inference mode.",
	)
	parser.add_argument(
		"--source",
		type=str,
		required=True,
		help="Path to image/video file. For video mode, pass 0 for webcam.",
	)
	parser.add_argument(
		"--imgsz",
		type=int,
		default=640,
		help="Inference image size.",
	)
	parser.add_argument(
		"--conf",
		type=float,
		default=0.25,
		help="Confidence threshold.",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cpu",
		help="Inference device (e.g., cpu, 0).",
	)
	parser.add_argument(
		"--task",
		type=str,
		choices=["auto", "detect", "segment", "classify", "pose", "obb"],
		default="auto",
		help="Model task override. Use auto to infer task from model.",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default=str(DEFAULT_OUTPUT_DIR),
		help="Directory to save outputs.",
	)
	return parser.parse_args()


def resolve_source(mode: str, source_value: str, ignore: bool = True) -> str | int:
	if mode == "video" and source_value.isdigit():
		return int(source_value)

	source_path = Path(source_value)
	if not source_path.exists() and not ignore:
		raise FileNotFoundError(f"Source file not found: {source_path}")
	return str(source_path)


def run_image_inference(
	model_path: Path,
	source: str,
	imgsz: int,
	conf: float,
	device: str,
	task: str | None,
	output_dir: Path,
	type: str = 'yolo',
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	task = task if task else ("detect" if type == 'yolo' else "auto")

	model = YOLO(str(model_path), task=task) if task or type == 'yolo' else RTDETR(str(model_path))
	results = model.predict(
		source=source,
		imgsz=imgsz,
		conf=conf,
		device=device,
		save=True,
		project=str(output_dir),
		name="image_pred",
		exist_ok=True,
	)

	print("results:", results[0].masks if results[0].masks is not None else "No masks")

	# save the result as json in the ourput dir
	print("Saving results to JSON in : " , output_dir / "results.json")
	with open(output_dir / "results.json", "w") as f:
		# save the boxes and segmentations (all result) in json format with proper keys
		json.dump([{
			"boxes": result.boxes.xyxy.tolist(),
			'segmentations':[mask.tolist() for mask in result.masks.xy] if result.masks is not None else None,
			"labels": result.boxes.cls.tolist(),
			"scores": result.boxes.conf.numpy().tolist(),
		} for result in results], f, indent=4)

	# save the annotated image to the output dir
	annotated_img = results[0].plot()
	output_img_path = output_dir / f"{Path(source).stem}_pred{Path(source).suffix}"
	cv2.imwrite(str(output_img_path), annotated_img)


def run_video_inference(
	model_path: Path,
	source,
	imgsz: int,
	conf: float,
	device: str,
	task: str | None,
	output_dir: Path,
	type: str = 'yolo',
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	model = YOLO(str(model_path), task=task) if type == 'yolo' else RTDETR(str(model_path))
	capture = cv2.VideoCapture(source)
	if not capture.isOpened():
		raise RuntimeError(f"Unable to open video source: {source}")

	width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
	height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
	fps = float(capture.get(cv2.CAP_PROP_FPS))
	if fps <= 0:
		fps = 30.0

	output_path = output_dir / "video_pred.mp4"
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
	if not writer.isOpened():
		capture.release()
		raise RuntimeError(f"Unable to create output video: {output_path}")

	window_name = "ONNX Video Inference (press q to quit)"

	try:
		while True:
			ok, frame = capture.read()
			if not ok:
				break

			results = model.predict(
				source=frame,
				imgsz=imgsz,
				conf=conf,
				device=device,
				verbose=False,
			)
			annotated_frame = results[0].plot()
			writer.write(annotated_frame)
			cv2.imshow(window_name, annotated_frame)

			key = cv2.waitKey(1) & 0xFF
			if key in (ord("q"), 27):
				break
	finally:
		capture.release()
		writer.release()
		cv2.destroyAllWindows()

	print(f"Saved video prediction to: {output_path}")


def main() -> int:
	args = parse_args()

	try:
		model_path = Path(args.model) if args.model else find_default_model(WEIGHTS_DIR)
		if not model_path.exists():
			raise FileNotFoundError(f"Model file not found: {model_path}")
		if model_path.suffix.lower() != ".onnx" and model_path.suffix.lower() != ".pt":
			raise ValueError(f"Model must be an ONNX or PyTorch file: {model_path}")

		task = None if args.task == "auto" else args.task
		source = resolve_source(args.mode, args.source)
		if args.mode == "image":
			if args.source_dir:
				run_folder_inference(
					model_path=model_path,
					source=args.source_dir,
					imgsz=args.imgsz,
					conf=args.conf,
					device=args.device,
					task=task,
					output_dir=Path(args.output_dir),
					type=args.type or 'yolo'
				)
			run_image_inference(
				model_path=model_path,
				source=source,
				imgsz=args.imgsz,
				conf=args.conf,
				device=args.device,
				task=task,
				output_dir=Path(args.output_dir),
				type=args.type or 'yolo'
			)
		else:
			run_video_inference(
				model_path=model_path,
				source=source,
				imgsz=args.imgsz,
				conf=args.conf,
				device=args.device,
				task=task,
				output_dir=Path(args.output_dir),
				type=args.type or 'yolo'
			)
	except Exception as exc:
		print(f"Error: {exc}")
		return 1

	print("Inference completed successfully.")
	return 0


# function that use folder as image source and result on folder with same name + _pred in csv and visualization
def run_folder_inference(
	model_path: Path,
	source: str,
	imgsz: int,
	conf: float,
	device: str,
	task: str | None,
	output_dir: Path,
	type: str = 'yolo',
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	model = YOLO(str(model_path), task=task) if type == 'yolo' else RTDETR(str(model_path))
	for img_path in Path(source).glob("*.*"):
		if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
			continue

		results = model.predict(
			source=str(img_path),
			imgsz=imgsz,
			conf=conf,
			device=device,
			verbose=False,
		)
		annotated_img = results[0].plot()
		output_img_path = output_dir / f"{img_path.stem}_pred{img_path.suffix}"
		cv2.imwrite(str(output_img_path), annotated_img)
		print(f"Saved prediction for {img_path.name} to {output_img_path.name}")

if __name__ == "__main__":
	sys.exit(main())
