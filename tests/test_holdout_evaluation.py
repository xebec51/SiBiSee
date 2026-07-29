from __future__ import annotations

import csv
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_holdout import evaluate_rows, summarize_predictions
from sibisee.domain.detection import BoundingBox, Detection


class CountingPredictor:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, image: Image.Image) -> tuple[tuple[Detection, ...], float]:
        self.calls += 1
        detections = (
            Detection("B", 0.91, BoundingBox(0, 0, 10, 10)),
            Detection("A", 0.82, BoundingBox(0, 0, 20, 20)),
        )
        return detections, 12.5


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(40, 50, 60)).save(path)


def _row(relative_path: str, class_name: str) -> dict[str, str]:
    return {
        "relative_path": relative_path,
        "class_name": class_name,
        "participant_id": "P1",
        "session_id": "S1",
        "device_label": "webcam",
        "background": "plain",
        "lighting": "indoor",
        "distance": "desk",
    }


def test_evaluate_rows_calls_predictor_once_per_image_and_uses_confidence_primary(tmp_path: Path) -> None:
    holdout_dir = tmp_path / "holdout"
    _write_image(holdout_dir / "images" / "one.jpg")
    _write_image(holdout_dir / "images" / "two.jpg")
    predictor = CountingPredictor()

    predictions = evaluate_rows(
        [_row("images/one.jpg", "B"), _row("images/two.jpg", "A")],
        holdout_dir,
        ["A", "B"],
        predictor,
        strategy="confidence",
    )
    summary = summarize_predictions(predictions, ["A", "B"])

    assert predictor.calls == 2
    assert [prediction.predicted_class for prediction in predictions] == ["B", "B"]
    assert summary["sample_count"] == 2
    assert summary["accuracy"] == 0.5
    assert summary["detection_coverage"] == 1.0


def test_evaluate_rows_area_strategy_can_select_larger_lower_confidence_box(tmp_path: Path) -> None:
    holdout_dir = tmp_path / "holdout"
    _write_image(holdout_dir / "images" / "one.jpg")
    predictor = CountingPredictor()

    predictions = evaluate_rows([_row("images/one.jpg", "A")], holdout_dir, ["A", "B"], predictor, strategy="area")

    assert predictor.calls == 1
    assert predictions[0].predicted_class == "A"


def test_prediction_rows_are_csv_serializable(tmp_path: Path) -> None:
    holdout_dir = tmp_path / "holdout"
    _write_image(holdout_dir / "images" / "one.jpg")
    predictor = CountingPredictor()
    predictions = evaluate_rows(
        [_row("images/one.jpg", "B")], holdout_dir, ["A", "B"], predictor, strategy="confidence"
    )
    output_path = tmp_path / "predictions.csv"

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["expected_class", "predicted_class", "correct"])
        writer.writeheader()
        writer.writerow(
            {
                "expected_class": predictions[0].expected_class,
                "predicted_class": predictions[0].predicted_class,
                "correct": predictions[0].correct,
            }
        )

    rows = list(csv.DictReader(output_path.open(encoding="utf-8")))
    assert rows == [{"expected_class": "B", "predicted_class": "B", "correct": "True"}]
