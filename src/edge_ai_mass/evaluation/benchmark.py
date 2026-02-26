"""Latency and throughput benchmarking for edge deployment.

Profiles each pipeline stage independently and the full cascade, reporting
per-frame latency, FPS, and memory usage.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    stage_latencies: dict[str, list[float]] = field(default_factory=dict)
    total_latencies: list[float] = field(default_factory=list)

    @property
    def avg_fps(self) -> float:
        if not self.total_latencies:
            return 0.0
        avg_ms = np.mean(self.total_latencies)
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, vals in self.stage_latencies.items():
            arr = np.array(vals)
            out[name] = {
                "mean_ms": float(np.mean(arr)),
                "p50_ms": float(np.median(arr)),
                "p95_ms": float(np.percentile(arr, 95)),
                "p99_ms": float(np.percentile(arr, 99)),
            }
        arr = np.array(self.total_latencies)
        out["total"] = {
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "avg_fps": self.avg_fps,
        }
        return out


def benchmark_pipeline(
    pipeline: Any,
    input_shape: tuple[int, ...] = (480, 640, 3),
    num_warmup: int = 5,
    num_iterations: int = 50,
) -> BenchmarkResult:
    """Run the full pipeline repeatedly on a dummy image and collect timings."""
    dummy = np.random.randint(0, 255, input_shape, dtype=np.uint8)

    # Warmup
    for _ in range(num_warmup):
        pipeline.run(dummy)

    result = BenchmarkResult()
    for i in range(num_iterations):
        t0 = time.perf_counter()
        pr = pipeline.run(dummy)
        elapsed = (time.perf_counter() - t0) * 1000
        result.total_latencies.append(elapsed)

        for stage_name, lat in pr.latency_ms.items():
            if isinstance(lat, (int, float)):
                result.stage_latencies.setdefault(stage_name, []).append(lat)

    logger.info("Benchmark done — avg FPS: %.1f", result.avg_fps)
    return result
