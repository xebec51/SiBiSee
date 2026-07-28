from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
SIGNER_COLUMNS = ("signer", "signer_id", "person", "person_id", "subject", "subject_id")
SESSION_COLUMNS = ("session", "session_id", "source_video", "video", "video_id", "collection_session")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))


def write_csv(path: Path, rows: list[dict[str, Any]], fallback_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else fallback_fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def first_available_value(row: dict[str, str], columns: tuple[str, ...]) -> str | None:
    for column in columns:
        value = row.get(column)
        if value:
            return value
    return None


def duplicate_group_lookup(duplicate_groups_path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in read_csv(duplicate_groups_path):
        group_id = row.get("group_id", "")
        for member in row.get("members", "").split("|"):
            if member:
                lookup[member] = f"duplicate:{group_id}"
    return lookup


def group_rows(
    rows: list[dict[str, str]], duplicate_lookup: dict[str, str]
) -> tuple[dict[str, list[dict[str, str]]], str]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    strategy = "stratified_random"
    for row in rows:
        signer = first_available_value(row, SIGNER_COLUMNS)
        session = first_available_value(row, SESSION_COLUMNS)
        if signer:
            group_key = f"signer:{signer}"
            strategy = "signer"
        elif session:
            group_key = f"session:{session}"
            strategy = "session" if strategy != "signer" else strategy
        elif row["image_path"] in duplicate_lookup:
            group_key = duplicate_lookup[row["image_path"]]
            if strategy == "stratified_random":
                strategy = "duplicate_cluster"
        else:
            group_key = f"image:{row['image_path']}"
        groups[group_key].append(row)
    return groups, strategy


def class_key(rows: list[dict[str, str]]) -> str:
    counts: Counter[str] = Counter()
    for row in rows:
        for class_id in row.get("class_ids", "").split("|"):
            if class_id:
                counts[class_id] += 1
    return counts.most_common(1)[0][0] if counts else "unknown"


def target_counts(total: int) -> dict[str, int]:
    train = round(total * SPLIT_RATIOS["train"])
    val = round(total * SPLIT_RATIOS["val"])
    test = max(0, total - train - val)
    return {"train": train, "val": val, "test": test}


def assign_splits(groups: dict[str, list[dict[str, str]]], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    grouped_by_class: defaultdict[str, list[tuple[str, list[dict[str, str]]]]] = defaultdict(list)
    for group_key, members in groups.items():
        grouped_by_class[class_key(members)].append((group_key, members))

    assignments: dict[str, str] = {}
    split_image_counts: Counter[str] = Counter()
    total_images = sum(len(members) for members in groups.values())
    targets = target_counts(total_images)
    split_order = ("train", "val", "test")

    for class_groups in grouped_by_class.values():
        rng.shuffle(class_groups)
        for group_key, members in class_groups:
            split = min(
                split_order,
                key=lambda name: (
                    split_image_counts[name] / max(targets[name], 1),
                    split_image_counts[name],
                ),
            )
            assignments[group_key] = split
            split_image_counts[split] += len(members)
    return assignments


def create_splits(
    manifest_path: Path,
    duplicate_groups_path: Path,
    output_dir: Path,
    dataset_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    rows = read_csv(manifest_path)
    if not rows:
        raise FileNotFoundError(f"Manifest tidak ditemukan atau kosong: {manifest_path}")

    duplicate_lookup = duplicate_group_lookup(duplicate_groups_path)
    groups, strategy = group_rows(rows, duplicate_lookup)
    assignments = assign_splits(groups, seed)
    output_rows: list[dict[str, Any]] = []
    for group_key, members in groups.items():
        split = assignments[group_key]
        for row in members:
            updated = dict(row)
            updated["split_group"] = group_key
            updated["new_split"] = split
            output_rows.append(updated)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = output_dir / "split_manifest.csv"
    write_csv(split_manifest, output_rows, ["image_path", "new_split"])

    split_files: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        split_path = output_dir / f"{split}.txt"
        image_paths = [
            str((dataset_dir / row["image_path"]).resolve()) if dataset_dir else row["image_path"]
            for row in output_rows
            if row["new_split"] == split
        ]
        split_path.write_text("\n".join(image_paths) + ("\n" if image_paths else ""), encoding="utf-8")
        split_files[split] = split_path

    dataset_yaml = output_dir / "sibisee_splits.yaml"
    yaml_payload = {
        "path": str(dataset_dir.resolve()) if dataset_dir else ".",
        "train": str(split_files["train"].resolve()),
        "val": str(split_files["val"].resolve()),
        "test": str(split_files["test"].resolve()),
    }
    source_data_yaml = (dataset_dir / "data.yaml") if dataset_dir else None
    if source_data_yaml and source_data_yaml.exists():
        source_payload = yaml.safe_load(source_data_yaml.read_text(encoding="utf-8")) or {}
        yaml_payload["names"] = source_payload.get("names", [])
    dataset_yaml.write_text(yaml.safe_dump(yaml_payload, sort_keys=False), encoding="utf-8")

    summary = {
        "seed": seed,
        "strategy": strategy,
        "subject_independent_evaluation_claim": strategy == "signer",
        "image_counts": Counter(row["new_split"] for row in output_rows),
        "group_count": len(groups),
        "split_manifest": str(split_manifest),
        "dataset_yaml": str(dataset_yaml),
    }
    serializable_summary = {key: dict(value) if isinstance(value, Counter) else value for key, value in summary.items()}
    (output_dir / "split_summary.json").write_text(
        json.dumps(serializable_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return serializable_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leakage-aware deterministic train/val/test split manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/dataset/manifest.csv"))
    parser.add_argument("--duplicates", type=Path, default=Path("artifacts/dataset/duplicate_groups.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dataset"))
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = create_splits(args.manifest, args.duplicates, args.output_dir, args.dataset_dir, args.seed)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
