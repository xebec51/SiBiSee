from __future__ import annotations

import argparse
from pathlib import Path

from mlops_utils import sha256_file, write_json


def export_model(model_path: Path, output_dir: Path, fmt: str = "onnx") -> Path:
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    exported = Path(model.export(format=fmt))
    target = output_dir / exported.name
    if exported.resolve() != target.resolve():
        target.write_bytes(exported.read_bytes())
    write_json(
        output_dir / "model_metadata.json",
        {"format": fmt, "artifact": str(target), "artifact_sha256": sha256_file(target)},
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a SiBiSee model artifact.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/models"))
    parser.add_argument("--format", default="onnx")
    args = parser.parse_args()
    target = export_model(args.model, args.output_dir, args.format)
    print(f"artifact: {target}")


if __name__ == "__main__":
    main()
