from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_WORKSPACE = "sibi-detection-nftzq"
DEFAULT_PROJECT = "sibi-bieme"
DEFAULT_VERSION = 2
DEFAULT_FORMAT = "yolov8"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class DatasetDownloadError(RuntimeError):
    """Raised when Roboflow does not produce a usable dataset."""


@dataclass(frozen=True)
class DatasetDownloadResult:
    dataset_dir: Path
    data_yaml: Path
    image_count: int
    label_count: int
    manifest_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_manifest(dataset_dir: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.name not in {"source_manifest.json", "source_metadata.json"}:
            rows.append(
                {
                    "path": str(path.relative_to(dataset_dir)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def manifest_sha256(manifest: list[dict[str, str | int]]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def count_images(dataset_dir: Path) -> int:
    return sum(1 for path in dataset_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def count_labels(dataset_dir: Path) -> int:
    labels_dir = dataset_dir / "labels"
    if labels_dir.exists():
        return sum(1 for path in labels_dir.rglob("*.txt") if path.is_file())
    return sum(1 for path in dataset_dir.rglob("labels/*.txt") if path.is_file())


def has_reasonable_split_structure(dataset_dir: Path) -> bool:
    train_dir = dataset_dir / "train"
    valid_dir = dataset_dir / "valid"
    val_dir = dataset_dir / "val"
    return train_dir.exists() and (valid_dir.exists() or val_dir.exists())


def validate_downloaded_dataset(dataset_dir: Path) -> tuple[Path, int, int, list[dict[str, str | int]], str]:
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise DatasetDownloadError("Roboflow SDK tidak menghasilkan dataset valid: directory hasil tidak ditemukan.")

    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists() or not data_yaml.is_file():
        raise DatasetDownloadError("Roboflow SDK tidak menghasilkan dataset valid: data.yaml tidak ditemukan.")

    image_count = count_images(dataset_dir)
    if image_count < 1:
        raise DatasetDownloadError("Roboflow SDK tidak menghasilkan dataset valid: image tidak ditemukan.")

    label_count = count_labels(dataset_dir)
    if label_count < 1:
        raise DatasetDownloadError("Roboflow SDK tidak menghasilkan dataset valid: label tidak ditemukan.")

    manifest = build_source_manifest(dataset_dir)
    if not manifest:
        raise DatasetDownloadError("Roboflow SDK tidak menghasilkan dataset valid: file_count nol.")

    if not has_reasonable_split_structure(dataset_dir):
        raise DatasetDownloadError(
            "Roboflow SDK tidak menghasilkan dataset valid: struktur train/valid atau train/val tidak ditemukan."
        )

    return data_yaml, image_count, label_count, manifest, manifest_sha256(manifest)


def assert_safe_overwrite_target(output_dir: Path) -> Path:
    resolved_output = output_dir.resolve()
    if resolved_output == Path(resolved_output.anchor):
        raise DatasetDownloadError("Output directory overwrite tidak aman: target adalah filesystem root.")
    return resolved_output


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    resolved_output = output_dir.resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise DatasetDownloadError(f"Output path bukan directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise DatasetDownloadError(
                f"Output directory tidak kosong: {output_dir}. Bersihkan manual atau gunakan --overwrite secara eksplisit."
            )
        resolved_output = assert_safe_overwrite_target(output_dir)
        shutil.rmtree(resolved_output)
    return resolved_output


def move_staging_to_output(staging_dataset_dir: Path, output_dir: Path) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        output_dir.rmdir()
    shutil.move(str(staging_dataset_dir), str(output_dir))


def default_roboflow_factory(api_key: str) -> Any:
    from roboflow import Roboflow

    return Roboflow(api_key=api_key)


def write_success_metadata(
    dataset_dir: Path,
    manifest: list[dict[str, str | int]],
    checksum: str,
    image_count: int,
    label_count: int,
) -> None:
    metadata = {
        "workspace": DEFAULT_WORKSPACE,
        "project": DEFAULT_PROJECT,
        "version": DEFAULT_VERSION,
        "format": DEFAULT_FORMAT,
        "source_url": f"https://universe.roboflow.com/{DEFAULT_WORKSPACE}/{DEFAULT_PROJECT}/dataset/{DEFAULT_VERSION}",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(manifest),
        "image_count": image_count,
        "label_count": label_count,
        "manifest_sha256": checksum,
    }
    (dataset_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (dataset_dir / "source_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def download_dataset(
    output_dir: Path,
    overwrite: bool = False,
    *,
    roboflow_factory: Callable[[str], Any] = default_roboflow_factory,
) -> DatasetDownloadResult:
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise DatasetDownloadError("ROBOFLOW_API_KEY belum diset. Isi lewat environment variable, bukan argument CLI.")

    resolved_output = prepare_output_dir(output_dir, overwrite)
    with tempfile.TemporaryDirectory(prefix="sibisee-roboflow-") as temp_parent:
        staging_dataset_path = Path(temp_parent) / "dataset"
        rf = roboflow_factory(api_key)
        project = rf.workspace(DEFAULT_WORKSPACE).project(DEFAULT_PROJECT)
        version = project.version(DEFAULT_VERSION)
        dataset = version.download(
            model_format=DEFAULT_FORMAT,
            location=str(staging_dataset_path),
            overwrite=False,
        )
        dataset_dir = Path(getattr(dataset, "location", staging_dataset_path)).resolve()
        data_yaml, image_count, label_count, manifest, checksum = validate_downloaded_dataset(dataset_dir)

        move_staging_to_output(dataset_dir, resolved_output)

    final_data_yaml = resolved_output / data_yaml.relative_to(dataset_dir)
    write_success_metadata(resolved_output, manifest, checksum, image_count, label_count)
    return DatasetDownloadResult(
        dataset_dir=resolved_output,
        data_yaml=final_data_yaml,
        image_count=image_count,
        label_count=label_count,
        manifest_sha256=checksum,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SiBiSee dataset from Roboflow without exposing credentials.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        result = download_dataset(args.output_dir, overwrite=args.overwrite)
    except DatasetDownloadError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"dataset_dir: {result.dataset_dir}")
    print(f"data_yaml: {result.data_yaml}")
    print(f"image_count: {result.image_count}")
    print(f"label_count: {result.label_count}")
    print(f"manifest_sha256: {result.manifest_sha256}")


if __name__ == "__main__":
    main()
