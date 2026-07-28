from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_dataset import audit_dataset  # noqa: E402


def test_audit_dataset_outputs_expected_artifacts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    image_dir = dataset / "train" / "images"
    label_dir = dataset / "train" / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(image_dir / "a.jpg")
    (label_dir / "a.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (dataset / "data.yaml").write_text("names: ['A']\n", encoding="utf-8")

    output_dir = tmp_path / "artifacts"
    summary = audit_dataset(dataset, output_dir, doc_path=tmp_path / "dataset-audit.md")

    assert summary["images"] == 1
    assert summary["annotations"] == 1
    assert summary["classes_from_data_yaml"] == 1
    assert (output_dir / "manifest.csv").exists()
    assert (output_dir / "summary.json").exists()
