from __future__ import annotations

import argparse
from pathlib import Path

from mlops_utils import environment_snapshot, write_json


def evaluate(model_path: Path, data_yaml: Path, output_dir: Path, split: str = "test") -> Path:
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    results = model.val(data=str(data_yaml), split=split, plots=True)
    write_json(output_dir / "environment.json", environment_snapshot())
    write_json(output_dir / "metrics.json", getattr(results, "results_dict", {}) or {})
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SiBiSee model on a fixed split.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    output_dir = evaluate(args.model, args.data, args.output_dir, args.split)
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
