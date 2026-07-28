from __future__ import annotations

from app import apply_static_prediction, get_temporal_decoder_for_settings
from sibisee.config import AppSettings, TemporalSettings
from sibisee.domain.detection import Detection, PredictionResult
from sibisee.domain.transcript import TranscriptBuilder


class SessionState(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def test_static_prediction_uses_primary_detection_without_temporal_window() -> None:
    transcript = TranscriptBuilder()
    result = PredictionResult(detections=(Detection("A", 0.7), Detection("B", 0.9)))

    primary = apply_static_prediction(result, transcript, AppSettings())

    assert primary.label == "B"
    assert transcript.snapshot() == ("B",)


def test_temporal_decoder_recreated_when_settings_change() -> None:
    session = SessionState()
    first = get_temporal_decoder_for_settings(session, AppSettings())
    settings = AppSettings(temporal=TemporalSettings(window_size=11))

    second = get_temporal_decoder_for_settings(session, settings)

    assert first is not second
    assert session.temporal_settings_fingerprint[0] == 11
