from __future__ import annotations

from sibisee.config import _metadata_model_sha256, load_settings


def test_load_settings_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("SIBISEE_CONFIDENCE_THRESHOLD", "0.65")
    monkeypatch.setenv("SIBISEE_INFER_EVERY_N_FRAMES", "5")
    monkeypatch.setenv("SIBISEE_CLASS_ALLOWLIST", "A,B,Saya")

    settings = load_settings()

    assert settings.inference.confidence_threshold == 0.65
    assert settings.inference.infer_every_n_frames == 5
    assert settings.inference.class_allowlist == {"A", "B", "Saya"}


def test_metadata_model_sha256_reads_production_metadata(tmp_path) -> None:
    metadata_path = tmp_path / "best.metadata.json"
    metadata_path.write_text('{"encrypted_artifact_sha256": "abc123"}', encoding="utf-8")

    assert _metadata_model_sha256(metadata_path) == "abc123"


def test_metadata_model_sha256_returns_none_for_missing_or_invalid_metadata(tmp_path) -> None:
    assert _metadata_model_sha256(tmp_path / "missing.json") is None
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    assert _metadata_model_sha256(invalid) is None
