from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from mlops_utils import environment_snapshot, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _to_list(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, int | float):
        return [float(value)]
    return [float(item) for item in value]


def extract_per_class_metrics(results: Any) -> dict[str, Any]:
    box = getattr(results, "box", None)
    names = getattr(results, "names", {}) or {}
    if box is None:
        return {"classes": [], "weakest_classes": []}

    precision = _to_list(getattr(box, "p", None))
    recall = _to_list(getattr(box, "r", None))
    ap50 = _to_list(getattr(box, "ap50", None))
    maps = _to_list(getattr(box, "maps", None))
    class_count = max(len(precision), len(recall), len(ap50), len(maps), len(names))

    rows = []
    for class_id in range(class_count):
        row = {
            "class_id": class_id,
            "name": names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id),
            "precision": precision[class_id] if class_id < len(precision) else None,
            "recall": recall[class_id] if class_id < len(recall) else None,
            "ap50": ap50[class_id] if class_id < len(ap50) else None,
            "map50_95": maps[class_id] if class_id < len(maps) else None,
        }
        rows.append(row)

    weakest = sorted(
        rows,
        key=lambda row: (
            row["map50_95"] is None,
            row["map50_95"] if row["map50_95"] is not None else 1.0,
            row["recall"] if row["recall"] is not None else 1.0,
        ),
    )[:10]
    return {"classes": rows, "weakest_classes": weakest}


def evaluate(model_path: Path, data_yaml: Path, output_dir: Path, split: str = "test") -> Path:
    from ultralytics import YOLO

    from sibisee.models import register_yolo_modules

    register_yolo_modules()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    results = model.val(data=str(data_yaml), split=split, plots=True)
    write_json(output_dir / "environment.json", environment_snapshot())
    write_json(output_dir / "metrics.json", getattr(results, "results_dict", {}) or {})
    write_json(output_dir / "per_class_metrics.json", extract_per_class_metrics(results))
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
