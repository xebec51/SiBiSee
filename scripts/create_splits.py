from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def create_random_splits(manifest_path: Path, output_path: Path, seed: int = 42) -> None:
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    by_label: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row.get("class_ids") or "unknown"].append(row)

    rng = random.Random(seed)
    output_rows: list[dict[str, str]] = []
    for group_rows in by_label.values():
        rng.shuffle(group_rows)
        total = len(group_rows)
        train_end = int(total * 0.7)
        val_end = int(total * 0.85)
        for index, row in enumerate(group_rows):
            split = "train" if index < train_end else "val" if index < val_end else "test"
            row = dict(row)
            row["new_split"] = split
            output_rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(output_rows[0].keys()) if output_rows else ["image_path", "new_split"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic fallback train/val/test split manifest.")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/dataset/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/dataset/split_manifest.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_random_splits(args.manifest, args.output, args.seed)
    print(f"split_manifest: {args.output}")


if __name__ == "__main__":
    main()
