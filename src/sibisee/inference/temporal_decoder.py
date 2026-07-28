from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass

from sibisee.config import TemporalSettings
from sibisee.domain.detection import Detection


@dataclass(frozen=True)
class DecoderState:
    stable_token: str | None
    stable_confidence: float
    confirmed_token: str | None
    window_size: int


class TemporalDecoder:
    def __init__(self, settings: TemporalSettings) -> None:
        self.settings = settings
        self._window: deque[Detection | None] = deque(maxlen=settings.window_size)
        self._last_confirmed_token: str | None = None
        self._last_confirmed_at = 0.0
        self._neutral_seen = True

    def reset(self) -> None:
        self._window.clear()
        self._neutral_seen = True

    def update(self, detections: tuple[Detection, ...], now: float | None = None) -> DecoderState:
        now = time.monotonic() if now is None else now
        primary = self._select_primary(detections)
        self._window.append(primary)
        stable_token, stable_confidence, matching_count = self._stable_vote()

        confirmed: str | None = None
        if stable_token is None:
            self._neutral_seen = True
        elif (
            matching_count >= self.settings.min_matching_frames
            and stable_confidence >= self.settings.min_average_confidence
            and self._can_confirm(stable_token, now)
        ):
            confirmed = stable_token
            self._last_confirmed_token = stable_token
            self._last_confirmed_at = now
            self._neutral_seen = False

        return DecoderState(
            stable_token=stable_token,
            stable_confidence=stable_confidence,
            confirmed_token=confirmed,
            window_size=len(self._window),
        )

    def _can_confirm(self, token: str, now: float) -> bool:
        if token != self._last_confirmed_token:
            return True
        if self._neutral_seen:
            return True
        return now - self._last_confirmed_at >= self.settings.confirmation_cooldown_seconds

    @staticmethod
    def _select_primary(detections: tuple[Detection, ...]) -> Detection | None:
        if not detections:
            return None
        return max(detections, key=lambda item: (item.confidence, item.box.area if item.box else 0.0))

    def _stable_vote(self) -> tuple[str | None, float, int]:
        real_detections = [item for item in self._window if item is not None]
        if not real_detections:
            return None, 0.0, 0

        weights: dict[str, float] = {}
        counts: Counter[str] = Counter()
        for detection in real_detections:
            weights[detection.label] = weights.get(detection.label, 0.0) + detection.confidence
            counts[detection.label] += 1
        token = max(weights, key=weights.get)
        count = counts[token]
        average_confidence = weights[token] / count
        if count < self.settings.min_matching_frames or average_confidence < self.settings.min_average_confidence:
            return None, average_confidence, count
        return token, average_confidence, count
