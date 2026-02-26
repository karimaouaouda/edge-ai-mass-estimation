"""Unit tests for config utilities."""

import os
import tempfile

from edge_ai_mass.utils.config import load_config, merge_configs


def test_load_config_basic():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("key: value\nnested:\n  a: 1\n")
        f.flush()
        cfg = load_config(f.name)
    assert cfg["key"] == "value"
    assert cfg["nested"]["a"] == 1
    os.unlink(f.name)


def test_load_config_env_interpolation():
    os.environ["TEST_VAR_EDGE_AI"] = "hello"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("val: ${TEST_VAR_EDGE_AI}\n")
        f.flush()
        cfg = load_config(f.name)
    assert cfg["val"] == "hello"
    os.unlink(f.name)
    del os.environ["TEST_VAR_EDGE_AI"]


def test_merge_configs():
    a = {"x": 1, "nested": {"a": 1, "b": 2}}
    b = {"x": 2, "nested": {"b": 3}}
    result = merge_configs(a, b)
    assert result["x"] == 2
    assert result["nested"]["a"] == 1
    assert result["nested"]["b"] == 3
