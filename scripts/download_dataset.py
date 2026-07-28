from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_WORKSPACE = "sibi-detection-nftzq"
DEFAULT_PROJECT = "sibi-bieme"
DEFAULT_VERSION = 2
DEFAULT_FORMAT = "yolov8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_manifest(dataset_dir: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.name != "source_metadata.json":
            rows.append(
                {
                    "path": str(path.relative_to(dataset_dir)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def download_dataset(output_dir: Path, overwrite: bool = False) -> Path:
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY belum diset. Isi lewat environment variable, bukan argument CLI.")

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise SystemExit(f"Output directory tidak kosong: {output_dir}. Gunakan --overwrite jika ingin mengganti.")
    output_dir.mkdir(parents=True, exist_ok=True)

    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(DEFAULT_WORKSPACE).project(DEFAULT_PROJECT)
    version = project.version(DEFAULT_VERSION)
    dataset = version.download(DEFAULT_FORMAT, location=str(output_dir), overwrite=overwrite)
    dataset_dir = Path(getattr(dataset, "location", output_dir)).resolve()

    manifest = build_source_manifest(dataset_dir)
    metadata = {
        "workspace": DEFAULT_WORKSPACE,
        "project": DEFAULT_PROJECT,
        "version": DEFAULT_VERSION,
        "format": DEFAULT_FORMAT,
        "source_url": f"https://universe.roboflow.com/{DEFAULT_WORKSPACE}/{DEFAULT_PROJECT}/dataset/{DEFAULT_VERSION}",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(dataset_dir),
        "file_count": len(manifest),
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    (dataset_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (dataset_dir / "source_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return dataset_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SiBiSee dataset from Roboflow without exposing credentials.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    dataset_dir = download_dataset(args.output_dir, overwrite=args.overwrite)
    print(f"dataset_dir: {dataset_dir}")
    print(f"data_yaml: {dataset_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
