#!/usr/bin/env python3
# Copyright (c) 2026 GI104 henrytsai
# Tatung University — I4210 AI實務專題
# tests/test_accuracy_gate.py — accuracy gate against accuracy_baseline.json
"""Accuracy gate: assert measured pipeline metrics meet committed bounds.

Reads ``accuracy_baseline.json`` at the repository root. Each metric records a
``measured`` value plus a ``min`` and/or ``max`` bound; the gate fails if any
measured value drifts outside its bound, catching accuracy/parameter
regressions in CI. Pure-logic — no GPIO, no models, runs on any CI runner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_BASELINE_PATH = Path(__file__).resolve().parents[1] / "accuracy_baseline.json"


def _load_baseline() -> dict:
    """Load and return the parsed accuracy baseline JSON."""
    return json.loads(_BASELINE_PATH.read_text(encoding="utf-8"))


def _metric_ids() -> list[str]:
    """Return the metric names declared in the baseline (for parametrisation)."""
    return list(_load_baseline()["metrics"].keys())


def test_baseline_file_exists() -> None:
    """The accuracy baseline file must be present at the repo root."""
    assert _BASELINE_PATH.is_file(), f"missing {_BASELINE_PATH}"


def test_baseline_has_required_sections() -> None:
    """Baseline must declare schema_version, evaluation_config and metrics."""
    baseline = _load_baseline()
    for key in ("schema_version", "evaluation_config", "metrics"):
        assert key in baseline, f"baseline missing '{key}'"
    assert baseline["metrics"], "baseline declares no metrics"


@pytest.mark.parametrize("name", _metric_ids())
def test_metric_within_bounds(name: str) -> None:
    """Each measured metric must satisfy its declared min/max bound."""
    metric = _load_baseline()["metrics"][name]
    assert "measured" in metric, f"metric '{name}' has no measured value"
    measured = metric["measured"]
    has_bound = False
    if "min" in metric:
        has_bound = True
        assert measured >= metric["min"], f"{name}={measured} below min {metric['min']}"
    if "max" in metric:
        has_bound = True
        assert measured <= metric["max"], f"{name}={measured} above max {metric['max']}"
    assert has_bound, f"metric '{name}' declares neither min nor max"


def test_liveness_margin_consistent() -> None:
    """liveness_margin must equal genuine_live_median minus spoof_live_median."""
    metrics = _load_baseline()["metrics"]
    expected = metrics["genuine_live_median"]["measured"] - metrics["spoof_live_median"]["measured"]
    assert metrics["liveness_margin"]["measured"] == pytest.approx(expected, abs=1e-6)
