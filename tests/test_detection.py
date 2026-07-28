from __future__ import annotations

from sibisee.domain.detection import (
    BoundingBox,
    Detection,
    dedupe_by_class,
    filter_detections,
    select_primary_detection,
)


def test_filter_detections_by_confidence_and_allowlist() -> None:
    detections = (
        Detection("A", 0.9),
        Detection("B", 0.2),
        Detection("C", 0.8),
    )

    filtered = filter_detections(detections, confidence_threshold=0.5, allowlist={"A", "B"})

    assert filtered == (Detection("A", 0.9),)


def test_dedupe_by_class_keeps_highest_confidence() -> None:
    detections = (
        Detection("A", 0.7, BoundingBox(0, 0, 2, 2)),
        Detection("A", 0.9, BoundingBox(0, 0, 1, 1)),
        Detection("B", 0.8),
    )

    deduped = dedupe_by_class(detections)

    assert [item.label for item in deduped] == ["A", "B"]
    assert deduped[0].confidence == 0.9


def test_select_primary_detection_supports_confidence_and_area() -> None:
    detections = (
        Detection("small", 0.95, BoundingBox(0, 0, 1, 1)),
        Detection("large", 0.80, BoundingBox(0, 0, 4, 4)),
    )

    assert select_primary_detection(detections, "confidence").label == "small"
    assert select_primary_detection(detections, "area").label == "large"
