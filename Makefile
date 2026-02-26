.PHONY: install dev test lint format export-onnx export-trt clean

install:
	pip install -e .

dev:
	pip install -e ".[all]"

test:
	pytest tests/unit/ -v --tb=short

lint:
	ruff check src/ tests/
	mypy src/edge_ai_mass/ --ignore-missing-imports

format:
	ruff format src/ tests/

export-onnx:
	python scripts/export_tensorrt.py --model models/weights/yolov8n-seg.pt --format onnx

export-trt:
	python scripts/export_tensorrt.py --model models/weights/yolov8n-seg.pt --format engine --half

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf dist build *.egg-info runs/
