from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from validate_holdout import HoldoutValidationConfig, HoldoutValidationError, validate_holdout


def _write_data_yaml(path: Path) -> None:
    path.write_text("names:\n  0: A\n  1: B\n", encoding="utf-8")


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), color=color).save(path)


def _write_metadata(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "relative_path",
        "class_name",
        "participant_id",
        "session_id",
        "device_label",
        "background",
        "lighting",
        "distance",
        "captured_at_unix",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row(relative_path: str, class_name: str = "A", session_id: str = "S1") -> dict[str, str]:
    return {
        "relative_path": relative_path,
        "class_name": class_name,
        "participant_id": "P1",
        "session_id": session_id,
        "device_label": "webcam",
        "background": "plain",
        "lighting": "indoor",
        "distance": "desk",
        "captured_at_unix": "1",
        "notes": "",
    }


def test_validate_holdout_rejects_unknown_class(tmp_path: Path) -> None:
    holdout_dir = tmp_path / "holdout"
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "out"
    _write_data_yaml(tmp_path / "data.yaml")
    _write_image(holdout_dir / "images" / "sample.jpg", (20, 30, 40))
    _write_metadata(holdout_dir / "metadata.csv", [_row("images/sample.jpg", class_name="TidakAda")])

    summary = validate_holdout(holdout_dir, dataset_dir, tmp_path / "data.yaml", output_dir)

    assert summary["status"] == "BLOCKED"
    assert any("Unknown class_name" in issue for issue in summary["issues"])


def test_validate_holdout_blocks_absolute_or_parent_relative_paths(tmp_path: Path) -> None:
    holdout_dir = tmp_path / "holdout"
    dataset_dir = tmp_path / "dataset"
    _write_data_yaml(tmp_path / "data.yaml")
    _write_metadata(holdout_dir / "metadata.csv", [_row("../outside.jpg")])

    summary = validate_holdout(holdout_dir, dataset_dir, tmp_path / "data.yaml", tmp_path / "out")

    assert summary["status"] == "BLOCKED"
    assert any("Path holdout" in issue for issue in summary["issues"])


def test_validate_holdout_reports_exact_near_duplicate_and_dataset_overlap(tmp_path: Path) -> None:
    holdout_dir = tmp_path / "holdout"
    dataset_dir = tmp_path / "dataset"
    output_dir = tmp_path / "out"
    _write_data_yaml(tmp_path / "data.yaml")
    _write_image(holdout_dir / "images" / "one.jpg", (200, 200, 200))
    _write_image(holdout_dir / "images" / "two.jpg", (200, 200, 200))
    _write_image(dataset_dir / "train" / "images" / "source.jpg", (200, 200, 200))
    _write_metadata(
        holdout_dir / "metadata.csv",
        [_row("images/one.jpg", session_id="S1"), _row("images/two.jpg", session_id="S2")],
    )

    summary = validate_holdout(
        holdout_dir,
        dataset_dir,
        tmp_path / "data.yaml",
        output_dir,
        HoldoutValidationConfig(min_per_class=1),
    )
    duplicate_rows = list(csv.DictReader((output_dir / "duplicate-groups.csv").open(encoding="utf-8")))
    overlap_rows = list(csv.DictReader((output_dir / "dataset-overlap.csv").open(encoding="utf-8")))

    assert summary["status"] == "BLOCKED"
    assert summary["exact_duplicate_rows"] == 2
    assert summary["near_duplicate_rows"] == 2
    assert len(duplicate_rows) == 2
    assert len(overlap_rows) == 2
    assert str(tmp_path) not in json.dumps(summary)


def test_validate_holdout_fails_missing_metadata() -> None:
    try:
        validate_holdout(Path("missing"), Path("dataset"), Path("data.yaml"), Path("out"))
    except HoldoutValidationError as exc:
        assert "metadata.csv" in str(exc)
    else:
        raise AssertionError("validate_holdout should fail when metadata is missing")
