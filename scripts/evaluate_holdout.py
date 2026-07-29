from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mlops_utils import environment_snapshot, sha256_file  # noqa: E402
from sibisee.config import InferenceSettings  # noqa: E402
from sibisee.domain.detection import Detection, select_primary_detection  # noqa: E402
from sibisee.inference.detector import YoloDetector  # noqa: E402
from sibisee.models import register_yolo_modules  # noqa: E402
from validate_holdout import HoldoutValidationError, load_class_names, read_metadata  # noqa: E402


class Predictor(Protocol):
    def predict(self, image: Image.Image) -> tuple[tuple[Detection, ...], float]:
        pass


@dataclass(frozen=True)
class HoldoutPrediction:
    relative_path: str
    expected_class: str
    predicted_class: str
    confidence: float | None
    latency_ms: float
    participant_id: str
    session_id: str
    device_label: str
    background: str
    lighting: str
    distance: str

    @property
    def correct(self) -> bool:
        return self.expected_class == self.predicted_class


class UltralyticsHoldoutPredictor:
    def __init__(self, model_path: Path, settings: InferenceSettings, device: str) -> None:
        from ultralytics import YOLO

        register_yolo_modules()
        model = YOLO(str(model_path))
        model.to(device)
        self.detector = YoloDetector(model, settings)

    def predict(self, image: Image.Image) -> tuple[tuple[Detection, ...], float]:
        result = self.detector.predict(image, annotate=False)
        return result.detections, result.latency_ms


def _safe_holdout_image(holdout_dir: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise HoldoutValidationError(f"Path holdout tidak aman: {relative_path}")
    root = holdout_dir.resolve()
    resolved = (root / path).resolve()
    if root != resolved and root not in resolved.parents:
        raise HoldoutValidationError(f"Path holdout keluar root: {relative_path}")
    return resolved


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def evaluate_rows(
    rows: list[dict[str, str]],
    holdout_dir: Path,
    class_names: list[str],
    predictor: Predictor,
    strategy: str,
) -> list[HoldoutPrediction]:
    class_name_set = set(class_names)
    predictions: list[HoldoutPrediction] = []
    for row in rows:
        expected_class = row["class_name"]
        if expected_class not in class_name_set:
            raise HoldoutValidationError(f"Unknown holdout class: {expected_class}")
        image_path = _safe_holdout_image(holdout_dir, row["relative_path"])
        with Image.open(image_path) as image:
            detections, latency_ms = predictor.predict(image.convert("RGB"))
        primary = select_primary_detection(detections, strategy)
        predictions.append(
            HoldoutPrediction(
                relative_path=row["relative_path"],
                expected_class=expected_class,
                predicted_class=primary.label if primary else "<no_detection>",
                confidence=primary.confidence if primary else None,
                latency_ms=latency_ms,
                participant_id=row["participant_id"],
                session_id=row["session_id"],
                device_label=row["device_label"],
                background=row["background"],
                lighting=row["lighting"],
                distance=row["distance"],
            )
        )
    return predictions


def _class_metrics(predictions: list[HoldoutPrediction], class_names: list[str]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for class_name in class_names:
        tp = sum(1 for prediction in predictions if prediction.expected_class == class_name and prediction.correct)
        fp = sum(
            1
            for prediction in predictions
            if prediction.expected_class != class_name and prediction.predicted_class == class_name
        )
        fn = sum(
            1
            for prediction in predictions
            if prediction.expected_class == class_name and prediction.predicted_class != class_name
        )
        support = sum(1 for prediction in predictions if prediction.expected_class == class_name)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "class_name": class_name,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def _group_metrics(predictions: list[HoldoutPrediction], field: str) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[HoldoutPrediction]] = defaultdict(list)
    for prediction in predictions:
        grouped[getattr(prediction, field)].append(prediction)
    rows: list[dict[str, float | int | str]] = []
    for value, members in sorted(grouped.items()):
        detected = sum(1 for prediction in members if prediction.predicted_class != "<no_detection>")
        correct = sum(1 for prediction in members if prediction.correct)
        rows.append(
            {
                "field": field,
                "value": value,
                "count": len(members),
                "accuracy": correct / len(members) if members else 0.0,
                "detection_coverage": detected / len(members) if members else 0.0,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_predictions(
    predictions: list[HoldoutPrediction],
    class_names: list[str],
    model_path: Path | None = None,
) -> dict[str, object]:
    class_rows = _class_metrics(predictions, class_names)
    total = len(predictions)
    correct = sum(1 for prediction in predictions if prediction.correct)
    detected = sum(1 for prediction in predictions if prediction.predicted_class != "<no_detection>")
    latencies = [prediction.latency_ms for prediction in predictions]
    supports = {row["class_name"]: int(row["support"]) for row in class_rows}
    macro_f1 = statistics.fmean(float(row["f1"]) for row in class_rows) if class_rows else 0.0
    weighted_f1 = (
        sum(float(row["f1"]) * supports[str(row["class_name"])] for row in class_rows) / total if total else 0.0
    )
    weakest_classes = sorted(
        class_rows,
        key=lambda row: (int(row["support"]) == 0, float(row["f1"]), float(row["recall"]), str(row["class_name"])),
    )[:10]
    summary: dict[str, object] = {
        "sample_count": total,
        "accuracy": correct / total if total else 0.0,
        "detection_coverage": detected / total if total else 0.0,
        "no_detection_rate": (total - detected) / total if total else 0.0,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": _percentile(latencies, 0.95),
        },
        "weakest_classes": weakest_classes,
    }
    if model_path is not None:
        summary["model"] = {
            "filename": model_path.name,
            "sha256": sha256_file(model_path),
        }
    return summary


def evaluate_holdout(
    model_path: Path,
    holdout_dir: Path,
    data_yaml: Path,
    output_dir: Path,
    device: str,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
    image_size: int,
    strategy: str,
) -> dict[str, object]:
    rows = read_metadata(holdout_dir)
    class_names = load_class_names(data_yaml)
    settings = InferenceSettings(
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        image_size=image_size,
        primary_detection_strategy=strategy,
    )
    predictor = UltralyticsHoldoutPredictor(model_path, settings, device)
    predictions = evaluate_rows(rows, holdout_dir, class_names, predictor, strategy)
    class_rows = _class_metrics(predictions, class_names)
    summary = summarize_predictions(predictions, class_names, model_path)

    prediction_rows = [
        {
            "relative_path": prediction.relative_path,
            "expected_class": prediction.expected_class,
            "predicted_class": prediction.predicted_class,
            "confidence": prediction.confidence,
            "latency_ms": prediction.latency_ms,
            "correct": prediction.correct,
            "participant_id": prediction.participant_id,
            "session_id": prediction.session_id,
            "device_label": prediction.device_label,
            "background": prediction.background,
            "lighting": prediction.lighting,
            "distance": prediction.distance,
        }
        for prediction in predictions
    ]
    confusion_counts = Counter((prediction.expected_class, prediction.predicted_class) for prediction in predictions)
    confusion_rows = [
        {"expected_class": expected, "predicted_class": predicted, "count": count}
        for (expected, predicted), count in sorted(confusion_counts.items())
    ]
    group_rows = []
    for field in ("participant_id", "session_id", "device_label", "background", "lighting", "distance"):
        group_rows.extend(_group_metrics(predictions, field))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "environment.json").write_text(
        json.dumps(environment_snapshot(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "predictions.csv",
        prediction_rows,
        [
            "relative_path",
            "expected_class",
            "predicted_class",
            "confidence",
            "latency_ms",
            "correct",
            "participant_id",
            "session_id",
            "device_label",
            "background",
            "lighting",
            "distance",
        ],
    )
    _write_csv(
        output_dir / "per-class.csv",
        class_rows,
        ["class_name", "support", "tp", "fp", "fn", "precision", "recall", "f1"],
    )
    _write_csv(output_dir / "confusion-matrix.csv", confusion_rows, ["expected_class", "predicted_class", "count"])
    _write_csv(
        output_dir / "group-metrics.csv", group_rows, ["field", "value", "count", "accuracy", "detection_coverage"]
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a model on private real-world SiBiSee holdout images.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/private/holdout-evaluation"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--strategy", choices=("confidence", "area"), default="confidence")
    args = parser.parse_args()
    summary = evaluate_holdout(
        args.model,
        args.holdout_dir,
        args.data,
        args.output_dir,
        args.device,
        args.conf,
        args.iou,
        args.max_det,
        args.imgsz,
        args.strategy,
    )
    print(f"output_dir: {args.output_dir}")
    print(f"sample_count: {summary['sample_count']}")
    print(f"accuracy: {summary['accuracy']}")


if __name__ == "__main__":
    main()
