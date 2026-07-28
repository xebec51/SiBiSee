from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark import p95, summarize_latencies  # noqa: E402


def test_summarize_latencies_reports_expected_statistics() -> None:
    summary = summarize_latencies([10.0, 20.0, 30.0])

    assert summary["iterations"] == 3
    assert summary["mean_ms"] == 20.0
    assert summary["median_ms"] == 20.0
    assert summary["p95_ms"] == 30.0
    assert summary["fps_mean"] == 50.0


def test_p95_uses_twentieth_quantile_for_larger_samples() -> None:
    values = [float(index) for index in range(1, 101)]

    assert p95(values) == 95.95
