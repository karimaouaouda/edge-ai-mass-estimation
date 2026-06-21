"""Deterministic native-library bootstrap for Jetson inference processes."""

from __future__ import annotations

import ctypes
import importlib
import logging
import sys
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)

ModuleImporter = Callable[[str], Any]
LibraryLoader = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class NativeRuntimeReport:
    sklearn_version: str
    openmp_threads: int | None


def preload_inference_native_dependencies(
    *,
    importer: ModuleImporter = importlib.import_module,
    preload_system_openmp: bool | None = None,
    library_loader: LibraryLoader = ctypes.CDLL,
) -> NativeRuntimeReport:
    """Load scikit-learn's OpenMP runtime before any Torch-backed model module."""

    should_preload_openmp = (
        sys.platform.startswith("linux")
        if preload_system_openmp is None
        else preload_system_openmp
    )
    if should_preload_openmp:
        try:
            library_loader(
                "libgomp.so.1",
                mode=getattr(ctypes, "RTLD_GLOBAL", 0),
            )
        except OSError as exc:
            raise RuntimeError(
                "Could not preload system libgomp before Python ML libraries"
            ) from exc

    try:
        sklearn = importer("sklearn")
        openmp_helpers = importer("sklearn.utils._openmp_helpers")
        thread_probe = getattr(openmp_helpers, "_openmp_effective_n_threads", None)
        openmp_threads = int(thread_probe()) if callable(thread_probe) else None
    except Exception as exc:
        raise RuntimeError(
            "Could not preload scikit-learn's native OpenMP runtime before model loading: "
            f"{exc}"
        ) from exc

    report = NativeRuntimeReport(
        sklearn_version=str(getattr(sklearn, "__version__", "unknown")),
        openmp_threads=openmp_threads,
    )
    logger.info(
        "Native inference dependencies ready before Torch model imports: "
        "sklearn=%s openmp_threads=%s",
        report.sklearn_version,
        report.openmp_threads,
    )
    return report
