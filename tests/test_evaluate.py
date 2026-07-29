from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import extract_per_class_metrics  # noqa: E402


def test_extract_per_class_metrics_records_weakest_classes() -> None:
    results = SimpleNamespace(
        names={0: "A", 1: "B"},
        box=SimpleNamespace(
            p=[0.9, 0.7],
            r=[0.8, 0.6],
            ap50=[0.95, 0.75],
            maps=[0.85, 0.45],
        ),
    )

    metrics = extract_per_class_metrics(results)

    assert metrics["classes"][0]["name"] == "A"
    assert metrics["classes"][1]["map50_95"] == 0.45
    assert metrics["weakest_classes"][0]["name"] == "B"
