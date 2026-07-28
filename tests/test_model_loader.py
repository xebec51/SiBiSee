from __future__ import annotations

from pathlib import Path

from cryptography.fernet import Fernet

from sibisee.inference.model_loader import load_encrypted_model, sha256_file


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
