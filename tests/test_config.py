from __future__ import annotations

from sibisee.config import load_settings


def test_load_settings_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("SIBISEE_CONFIDENCE_THRESHOLD", "0.65")
    monkeypatch.setenv("SIBISEE_INFER_EVERY_N_FRAMES", "5")
    monkeypatch.setenv("SIBISEE_CLASS_ALLOWLIST", "A,B,Saya")

    settings = load_settings()

    assert settings.inference.confidence_threshold == 0.65
    assert settings.inference.infer_every_n_frames == 5
    assert settings.inference.class_allowlist == {"A", "B", "Saya"}
