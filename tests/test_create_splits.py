from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from create_splits import create_splits  # noqa: E402


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_create_splits_keeps_duplicate_cluster_together(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    duplicates = tmp_path / "duplicate_groups.csv"
    rows = [
        {"image_path": "a.jpg", "class_ids": "0"},
        {"image_path": "b.jpg", "class_ids": "0"},
        {"image_path": "c.jpg", "class_ids": "1"},
    ]
    write_rows(manifest, rows)
    write_rows(
        duplicates,
        [{"group_id": "1", "kind": "phash", "member_count": "2", "splits": "train|val", "members": "a.jpg|b.jpg"}],
    )

    output_dir = tmp_path / "out"
    summary = create_splits(manifest, duplicates, output_dir, seed=7)
    split_rows = list(csv.DictReader((output_dir / "split_manifest.csv").open(encoding="utf-8")))
    cluster_splits = {row["new_split"] for row in split_rows if row["image_path"] in {"a.jpg", "b.jpg"}}

    assert len(cluster_splits) == 1
    assert summary["strategy"] == "duplicate_cluster"
    assert (output_dir / "sibisee_splits.yaml").exists()


def test_create_splits_preserves_class_coverage_when_groups_allow_it(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    duplicates = tmp_path / "duplicate_groups.csv"
    rows = []
    for class_id in range(3):
        for index in range(3):
            rows.append({"image_path": f"class-{class_id}-{index}.jpg", "class_ids": str(class_id)})
    write_rows(manifest, rows)
    write_rows(duplicates, [{"group_id": "none", "kind": "phash", "member_count": "0", "splits": "", "members": ""}])

    output_dir = tmp_path / "out"
    create_splits(manifest, duplicates, output_dir, seed=7)

    split_rows = list(csv.DictReader((output_dir / "split_manifest.csv").open(encoding="utf-8")))
    for split in {"val", "test"}:
        covered = {row["class_ids"] for row in split_rows if row["new_split"] == split}
        assert covered == {"0", "1", "2"}
