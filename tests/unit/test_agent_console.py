"""Tests for human-readable agent console formatting."""

from __future__ import annotations

from edge_ai_mass.agent.console import format_panel, format_stage_update


def test_format_panel_renders_bordered_key_value_table():
    output = format_panel(
        "Inference request",
        {"request_id": "req-1", "correlation_id": "corr-1"},
        status="received",
        width=72,
    )

    lines = output.splitlines()
    assert lines[0] == "+" + "-" * 70 + "+"
    assert "Inference request [RECEIVED]" in output
    assert "| request_id" in output
    assert "req-1" in output
    assert lines[-1] == lines[0]


def test_format_stage_update_includes_stage_source_and_fallback_details():
    output = format_stage_update(
        {
            "stage_key": "depth-estimation",
            "sequence": 3,
            "status": "completed",
            "title": "Estimating depth",
            "progress_percent": 65,
            "metadata": {
                "module": "depth.host_primary",
                "host_stage": "depth",
                "fallback_reason": "primary_unavailable",
                "primary_error": "depth load failed",
            },
        },
        width=96,
    )

    assert "INFERENCE STAGE :: Estimating depth" in output
    assert "depth-estimation" in output
    assert "COMPLETED" in output
    assert "65%" in output
    assert "host_primary" in output
    assert "primary_unavailable" in output
