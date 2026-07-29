from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from sibisee.inference.model_loader import ModelLoadError, load_encrypted_model, sha256_file


def test_load_encrypted_model_cleans_temporary_file(tmp_path: Path) -> None:
    key = Fernet.generate_key()
    encrypted_path = tmp_path / "model.pt.enc"
    encrypted_path.write_bytes(Fernet(key).encrypt(b"dummy model"))
    seen_path: list[Path] = []

    def loader(path: Path) -> bytes:
        seen_path.append(path)
        assert path.exists()
        return path.read_bytes()

    result = load_encrypted_model(encrypted_path, key, expected_sha256=sha256_file(encrypted_path), loader=loader)

    assert result == b"dummy model"
    assert seen_path
    assert not seen_path[0].exists()


def test_load_encrypted_model_rejects_wrong_key(tmp_path: Path) -> None:
    encrypted_path = tmp_path / "model.pt.enc"
    encrypted_path.write_bytes(Fernet(Fernet.generate_key()).encrypt(b"dummy model"))

    with pytest.raises(ModelLoadError, match="Kunci model tidak cocok"):
        load_encrypted_model(encrypted_path, Fernet.generate_key(), loader=lambda path: path.read_bytes())


def test_load_encrypted_model_rejects_wrong_checksum(tmp_path: Path) -> None:
    key = Fernet.generate_key()
    encrypted_path = tmp_path / "model.pt.enc"
    encrypted_path.write_bytes(Fernet(key).encrypt(b"dummy model"))

    with pytest.raises(ModelLoadError, match="Checksum"):
        load_encrypted_model(encrypted_path, key, expected_sha256="0" * 64, loader=lambda path: path.read_bytes())
