from __future__ import annotations

import hashlib
import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken

LOGGER = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the model artifact cannot be loaded safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected_sha256: str | None) -> None:
    if not expected_sha256:
        return
    actual = sha256_file(path)
    if actual.lower() != expected_sha256.lower():
        raise ModelLoadError("Checksum artifact model tidak sesuai metadata.")


def load_yolo_model(path: Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(path))


def load_encrypted_model(
    encrypted_path: Path,
    encryption_key: str | bytes | None,
    *,
    expected_sha256: str | None = None,
    loader: Callable[[Path], Any] = load_yolo_model,
) -> Any:
    if not encrypted_path.exists():
        raise ModelLoadError("Artifact model produksi tidak ditemukan.")
    verify_checksum(encrypted_path, expected_sha256)
    if not encryption_key:
        raise ModelLoadError("Kunci model belum dikonfigurasi.")

    try:
        fernet = Fernet(encryption_key)
        encrypted_data = encrypted_path.read_bytes()
        decrypted_data = fernet.decrypt(encrypted_data)
    except InvalidToken as exc:
        LOGGER.exception("Model decryption failed with invalid token.")
        raise ModelLoadError("Kunci model tidak cocok dengan artifact.") from exc
    except Exception as exc:
        LOGGER.exception("Unexpected encrypted model read/decrypt failure.")
        raise ModelLoadError("Model terenkripsi gagal dibaca.") from exc

    temp_path: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="sibisee-model-") as temp_dir:
            temp_path = Path(temp_dir) / "model.pt"
            temp_path.write_bytes(decrypted_data)
            return loader(temp_path)
    except ModelLoadError:
        raise
    except Exception as exc:
        LOGGER.exception("Model loader failed for decrypted artifact.")
        raise ModelLoadError("Model gagal dimuat. Periksa kompatibilitas artifact dan dependency.") from exc
    finally:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                LOGGER.warning("Temporary model file cleanup failed.", exc_info=True)
