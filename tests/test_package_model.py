from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import package_model as packaging


class FakeParameter:
    def __init__(self, count: int) -> None:
        self.count = count

    def numel(self) -> int:
        return self.count


class FakeTorchModel:
    def parameters(self) -> list[FakeParameter]:
        return [FakeParameter(10), FakeParameter(15)]


class FakeYolo:
    def __init__(self) -> None:
        self.model = FakeTorchModel()


def _fake_smoke(path: Path) -> tuple[FakeYolo, list[str]]:
    assert path.exists()
    return FakeYolo(), [f"class-{index}" for index in range(49)]


def test_package_model_requires_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(packaging.KEY_ENV_VAR, raising=False)
    monkeypatch.setattr(packaging, "_smoke_model", _fake_smoke)
    source = tmp_path / "best.pt"
    source.write_bytes(b"checkpoint")

    with pytest.raises(packaging.ModelPackagingError, match=packaging.KEY_ENV_VAR):
        packaging.package_model(source, tmp_path / "best.pt.enc", tmp_path / "best.metadata.json")


def test_package_model_rejects_invalid_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(packaging.KEY_ENV_VAR, "not-a-fernet-key")
    monkeypatch.setattr(packaging, "_smoke_model", _fake_smoke)
    source = tmp_path / "best.pt"
    source.write_bytes(b"checkpoint")

    with pytest.raises(packaging.ModelPackagingError, match="valid"):
        packaging.package_model(source, tmp_path / "best.pt.enc", tmp_path / "best.metadata.json")


def test_package_model_round_trip_metadata_and_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv(packaging.KEY_ENV_VAR, key)
    monkeypatch.setattr(packaging, "_smoke_model", _fake_smoke)
    source = tmp_path / "best.pt"
    source.write_bytes(b"checkpoint")
    output = tmp_path / "best.pt.enc"
    metadata_output = tmp_path / "best.metadata.json"

    metadata = packaging.package_model(source, output, metadata_output)
    loaded_metadata = json.loads(metadata_output.read_text(encoding="utf-8"))

    assert output.exists()
    assert loaded_metadata["encrypted_artifact_sha256"] == packaging.sha256_file(output)
    assert loaded_metadata["source_checkpoint_sha256"] == packaging.sha256_file(source)
    assert loaded_metadata["class_names"] == [f"class-{index}" for index in range(49)]
    assert loaded_metadata["parameter_count"] == 25
    assert metadata["backend"] == "pytorch"
    assert metadata["holdout_status"] == "not_run"
    assert not list(tmp_path.glob("*.tmp"))


def test_package_model_refuses_overwrite_and_replaces_atomically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv(packaging.KEY_ENV_VAR, key)
    monkeypatch.setattr(packaging, "_smoke_model", _fake_smoke)
    source = tmp_path / "best.pt"
    source.write_bytes(b"checkpoint")
    output = tmp_path / "best.pt.enc"
    metadata_output = tmp_path / "best.metadata.json"
    output.write_bytes(b"existing")

    with pytest.raises(packaging.ModelPackagingError, match="Output sudah ada"):
        packaging.package_model(source, output, metadata_output)

    packaging.package_model(source, output, metadata_output, overwrite=True)

    assert output.read_bytes() != b"existing"
    assert metadata_output.exists()


def test_package_model_rejects_wrong_decrypted_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv(packaging.KEY_ENV_VAR, key)
    monkeypatch.setattr(packaging, "_smoke_model", _fake_smoke)
    source = tmp_path / "best.pt"
    source.write_bytes(b"checkpoint")
    output = tmp_path / "best.pt.enc"
    metadata_output = tmp_path / "best.metadata.json"

    def corrupt_verify(encrypted_path: Path, fernet: Fernet, expected_source_sha256: str) -> None:
        raise packaging.ModelPackagingError("Checksum plaintext hasil decrypt tidak cocok dengan source checkpoint.")

    monkeypatch.setattr(packaging, "_verify_encrypted_artifact", corrupt_verify)

    with pytest.raises(packaging.ModelPackagingError, match="Checksum plaintext"):
        packaging.package_model(source, output, metadata_output)
