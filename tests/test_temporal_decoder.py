from __future__ import annotations

from sibisee.config import TemporalSettings
from sibisee.domain.detection import Detection
from sibisee.inference.temporal_decoder import TemporalDecoder


def test_temporal_decoder_confirms_after_matching_window() -> None:
    decoder = TemporalDecoder(TemporalSettings(window_size=5, min_matching_frames=3, min_average_confidence=0.7))

    states = [decoder.update((Detection("A", 0.8),), now=float(index)) for index in range(3)]

    assert states[-1].stable_token == "A"
    assert states[-1].confirmed_token == "A"


def test_temporal_decoder_prevents_duplicate_while_held() -> None:
    decoder = TemporalDecoder(
        TemporalSettings(
            window_size=3,
            min_matching_frames=2,
            min_average_confidence=0.7,
            confirmation_cooldown_seconds=10,
        )
    )

    assert decoder.update((Detection("A", 0.9),), now=0).confirmed_token is None
    assert decoder.update((Detection("A", 0.9),), now=1).confirmed_token == "A"
    assert decoder.update((Detection("A", 0.9),), now=2).confirmed_token is None


def test_temporal_decoder_allows_same_token_after_neutral() -> None:
    decoder = TemporalDecoder(TemporalSettings(window_size=2, min_matching_frames=2, min_average_confidence=0.7))

    decoder.update((Detection("A", 0.9),), now=0)
    assert decoder.update((Detection("A", 0.9),), now=1).confirmed_token == "A"
    decoder.update((), now=2)
    decoder.update((), now=3)
    decoder.update((Detection("A", 0.9),), now=4)
    assert decoder.update((Detection("A", 0.9),), now=5).confirmed_token == "A"
