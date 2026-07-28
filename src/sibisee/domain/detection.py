from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: BoundingBox | None = None


@dataclass(frozen=True)
class PredictionResult:
    detections: tuple[Detection, ...]
    latency_ms: float = 0.0
    annotated_frame: Any | None = None

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(detection.label for detection in self.detections)


def filter_detections(
    detections: list[Detection] | tuple[Detection, ...],
    confidence_threshold: float,
    allowlist: set[str] | None = None,
) -> tuple[Detection, ...]:
    filtered = [
        detection
        for detection in detections
        if detection.confidence >= confidence_threshold and (allowlist is None or detection.label in allowlist)
    ]
    return tuple(sorted(filtered, key=lambda item: item.confidence, reverse=True))


def dedupe_by_class(detections: tuple[Detection, ...]) -> tuple[Detection, ...]:
    best_by_label: dict[str, Detection] = {}
    for detection in detections:
        current = best_by_label.get(detection.label)
        if current is None or detection.confidence > current.confidence:
            best_by_label[detection.label] = detection
    return tuple(sorted(best_by_label.values(), key=lambda item: item.confidence, reverse=True))


def select_primary_detection(detections: tuple[Detection, ...], strategy: str = "confidence") -> Detection | None:
    if not detections:
        return None
    if strategy == "area":
        return max(detections, key=lambda item: ((item.box.area if item.box else 0.0), item.confidence))
    return max(detections, key=lambda item: (item.confidence, item.box.area if item.box else 0.0))


def detections_from_ultralytics(result: Any, names: dict[int, str] | list[str]) -> tuple[Detection, ...]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.cls is None:
        return ()

    classes = boxes.cls.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist() if boxes.conf is not None else [0.0] * len(classes)
    xyxy = boxes.xyxy.detach().cpu().tolist() if boxes.xyxy is not None else [None] * len(classes)

    detections: list[Detection] = []
    for class_id, confidence, coords in zip(classes, confidences, xyxy, strict=False):
        label = names[int(class_id)] if not isinstance(names, dict) else names.get(int(class_id), str(int(class_id)))
        box = BoundingBox(*map(float, coords)) if coords is not None else None
        detections.append(Detection(label=str(label), confidence=float(confidence), box=box))
    return tuple(detections)
