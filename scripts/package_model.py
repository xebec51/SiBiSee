from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from PIL import Image

from mlops_utils import environment_snapshot, git_commit, sha256_file, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sibisee.models import register_yolo_modules  # noqa: E402

KEY_ENV_VAR = "SIBISEE_MODEL_ENCRYPTION_KEY"
EXPECTED_CLASS_COUNT = 49
SELECTED_PARAMETER_COUNT = 11_199_426
SELECTED_GFLOPS = 28.869352


class ModelPackagingError(RuntimeError):
    pass


def _fernet_from_env() -> Fernet:
    key = os.getenv(KEY_ENV_VAR)
    if not key:
        raise ModelPackagingError(f"{KEY_ENV_VAR} belum dikonfigurasi.")
    try:
        return Fernet(key.encode("utf-8"))
    except (ValueError, TypeError) as exc:
        raise ModelPackagingError(f"{KEY_ENV_VAR} bukan Fernet key yang valid.") from exc


def _load_model(path: Path) -> Any:
    from ultralytics import YOLO

    register_yolo_modules()
    return YOLO(str(path))


def _smoke_model(path: Path) -> tuple[Any, list[str]]:
    model = _load_model(path)
    names = getattr(model, "names", {})
    class_names = list(names.values()) if isinstance(names, dict) else list(names)
    if len(class_names) != EXPECTED_CLASS_COUNT:
        raise ModelPackagingError(
            f"Checkpoint harus memiliki {EXPECTED_CLASS_COUNT} class, ditemukan {len(class_names)}."
        )
    image = Image.new("RGB", (640, 640), "white")
    results = model.predict(image, conf=0.25, verbose=False)
    if len(results) != 1:
        raise ModelPackagingError("Smoke inference checkpoint tidak menghasilkan satu result.")
    return model, [str(name) for name in class_names]


def _atomic_write(path: Path, data: bytes, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ModelPackagingError(f"Output sudah ada: {path}. Gunakan --overwrite untuk mengganti.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(data)
        temp_path.replace(path)
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def _verify_encrypted_artifact(encrypted_path: Path, fernet: Fernet, expected_source_sha256: str) -> None:
    plaintext = fernet.decrypt(encrypted_path.read_bytes())
    temp_path: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix="sibisee-package-") as temp_dir:
            temp_path = Path(temp_dir) / "model.pt"
            temp_path.write_bytes(plaintext)
            if sha256_file(temp_path).lower() != expected_source_sha256.lower():
                raise ModelPackagingError("Checksum plaintext hasil decrypt tidak cocok dengan source checkpoint.")
            _smoke_model(temp_path)
    except InvalidToken as exc:
        raise ModelPackagingError("Verifikasi decrypt artifact gagal.") from exc
    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink()


def build_metadata(
    *,
    model_path: Path,
    encrypted_path: Path,
    class_names: list[str],
    source_sha256: str,
    encrypted_sha256: str,
    parameter_count: int,
    gflops: float,
) -> dict[str, Any]:
    snapshot = environment_snapshot()
    return {
        "model_family": "YOLO",
        "architecture": "YOLOv8s-CBAM",
        "selected_run": "final-cbam-seed42",
        "seed": 42,
        "source_git_sha": git_commit(),
        "source_checkpoint_sha256": source_sha256,
        "encrypted_artifact_sha256": encrypted_sha256,
        "artifact_size_bytes": encrypted_path.stat().st_size,
        "source_checkpoint_size_bytes": model_path.stat().st_size,
        "parameter_count": parameter_count,
        "gflops": gflops,
        "image_size": 640,
        "class_names": class_names,
        "internal_test_metrics": {
            "precision": 0.94001,
            "recall": 0.92946,
            "map50": 0.96593,
            "map50_95": 0.84722,
        },
        "validation_aggregate": {
            "cbam_map50_95_mean": 0.83886,
            "cbam_map50_95_std": 0.00433,
            "selected_seed42_map50_95": 0.84158,
        },
        "backend": "pytorch",
        "onnx_status": "not_run",
        "onnx_status_detail": "NOT RUN - not required by the selected PyTorch deployment backend.",
        "holdout_status": "not_run",
        "holdout_status_detail": "NOT RUN - intentionally excluded from the current release scope.",
        "known_limitations": [
            "Real-world holdout was not run; real-world generalization is not claimed.",
            "Subject-independent evaluation cannot be proven because signer metadata is unavailable.",
            "The model recognizes isolated SIBI signs and is not a complete sign-language translator.",
            "The model is not suitable for safety-critical or accessibility-critical decisions.",
        ],
        "torch_version": snapshot.get("torch"),
        "ultralytics_version": snapshot.get("ultralytics"),
        "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def package_model(
    model_path: Path, output_path: Path, metadata_output: Path, overwrite: bool = False
) -> dict[str, Any]:
    if not model_path.exists():
        raise ModelPackagingError(f"Checkpoint source tidak ditemukan: {model_path}")
    if metadata_output.exists() and not overwrite:
        raise ModelPackagingError(f"Metadata output sudah ada: {metadata_output}. Gunakan --overwrite untuk mengganti.")

    fernet = _fernet_from_env()
    _model, class_names = _smoke_model(model_path)
    source_sha256 = sha256_file(model_path)
    encrypted_data = fernet.encrypt(model_path.read_bytes())
    _atomic_write(output_path, encrypted_data, overwrite=overwrite)
    encrypted_sha256 = sha256_file(output_path)
    _verify_encrypted_artifact(output_path, fernet, source_sha256)

    metadata = build_metadata(
        model_path=model_path,
        encrypted_path=output_path,
        class_names=class_names,
        source_sha256=source_sha256,
        encrypted_sha256=encrypted_sha256,
        parameter_count=SELECTED_PARAMETER_COUNT,
        gflops=SELECTED_GFLOPS,
    )
    temp_metadata = metadata_output.with_name(f".{metadata_output.name}.tmp")
    try:
        write_json(temp_metadata, metadata)
        temp_metadata.replace(metadata_output)
    finally:
        if temp_metadata.exists():
            temp_metadata.unlink()
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Encrypt and verify the selected SiBiSee production checkpoint.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        metadata = package_model(args.model, args.output, args.metadata_output, overwrite=args.overwrite)
    except ModelPackagingError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"output: {args.output}")
    print(f"metadata: {args.metadata_output}")
    print(f"source_checkpoint_sha256: {metadata['source_checkpoint_sha256']}")
    print(f"encrypted_artifact_sha256: {metadata['encrypted_artifact_sha256']}")
    print(f"class_count: {len(metadata['class_names'])}")


if __name__ == "__main__":
    main()
