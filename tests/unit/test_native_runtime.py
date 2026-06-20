"""Tests for deterministic native dependency loading order."""

from types import SimpleNamespace

import pytest

from edge_ai_mass.agent.native_runtime import preload_inference_native_dependencies


def test_sklearn_openmp_runtime_is_loaded_in_explicit_order():
    imported = []
    modules = {
        "sklearn": SimpleNamespace(__version__="1.8.0"),
        "sklearn.utils._openmp_helpers": SimpleNamespace(
            _openmp_effective_n_threads=lambda: 4
        ),
    }

    def importer(name):
        imported.append(name)
        return modules[name]

    report = preload_inference_native_dependencies(importer=importer)

    assert imported == ["sklearn", "sklearn.utils._openmp_helpers"]
    assert report.sklearn_version == "1.8.0"
    assert report.openmp_threads == 4


def test_native_dependency_preload_fails_before_model_loading():
    def failing_importer(_name):
        raise ImportError("sklearn unavailable")

    with pytest.raises(RuntimeError, match="Could not preload scikit-learn"):
        preload_inference_native_dependencies(importer=failing_importer)
