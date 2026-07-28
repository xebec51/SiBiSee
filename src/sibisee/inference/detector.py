from __future__ import annotations

import threading
import time
from typing import Any

from PIL import Image

from sibisee.config import InferenceSettings
from sibisee.domain.detection import (
    PredictionResult,
    dedupe_by_class,
    detections_from_ultralytics,
    filter_detections,
)
from sibisee.inference.preprocessing import preprocess_for_inference


class YoloDetector:
    def __init__(self, model: Any, settings: InferenceSettings) -> None:
        self._model = model
        self._settings = settings
        self._lock = threading.Lock()

    @property
    def names(self) -> dict[int, str] | list[str]:
        return self._model.names

    def predict(self, image: Image.Image | Any) -> PredictionResult:
        prepared = (
            preprocess_for_inference(image, self._settings.image_size) if isinstance(image, Image.Image) else image
        )
        start = time.perf_counter()
        with self._lock:
            raw_results = self._model.predict(
                prepared,
                conf=self._settings.confidence_threshold,
                iou=self._settings.iou_threshold,
                max_det=self._settings.max_detections,
                verbose=False,
            )
        latency_ms = (time.perf_counter() - start) * 1000
        detections = detections_from_ultralytics(raw_results[0], self.names)
        detections = filter_detections(
            detections,
            confidence_threshold=self._settings.confidence_threshold,
            allowlist=self._settings.class_allowlist,
        )
        return PredictionResult(detections=dedupe_by_class(detections), latency_ms=latency_ms)

    def annotate(self, image: Any) -> Any:
        with self._lock:
            raw_results = self._model.predict(
                image,
                conf=self._settings.confidence_threshold,
                iou=self._settings.iou_threshold,
                max_det=self._settings.max_detections,
                verbose=False,
            )
        return raw_results[0].plot()
