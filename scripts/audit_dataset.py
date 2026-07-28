from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KNOWN_SPLITS = {"train", "valid", "val", "validation", "test", "holdout"}
PHASH_SIZE = 32
PHASH_LOW_SIZE = 8
PHASH_THRESHOLD = 8


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left

    def groups(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            grouped[self.find(item)].append(item)
        return grouped


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("L").resize((PHASH_SIZE, PHASH_SIZE), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32)
    dct = cv2.dct(pixels)
    low = dct[:PHASH_LOW_SIZE, :PHASH_LOW_SIZE]
    median = np.median(low[1:, 1:])
    bits = low >= median
    return "".join("1" if bit else "0" for bit in bits.flatten())


def hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def yolo_label_path(image_path: Path, dataset_root: Path) -> Path:
    parts = list(image_path.relative_to(dataset_root).parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
    return dataset_root.joinpath(*parts).with_suffix(".txt")


def split_name(path: Path) -> str:
    for part in path.parts:
        normalized = part.lower()
        if normalized in KNOWN_SPLITS:
            return "val" if normalized in {"valid", "validation"} else normalized
    return "unknown"


def load_data_yaml(dataset_root: Path) -> tuple[Path | None, dict[int, str]]:
    candidates = [dataset_root / "data.yaml", dataset_root / "data.yml", *dataset_root.glob("*/data.yaml")]
    data_yaml = next((path for path in candidates if path.exists()), None)
    if data_yaml is None:
        return None, {}
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = payload.get("names", {})
    if isinstance(names, list):
        return data_yaml, {index: str(name) for index, name in enumerate(names)}
    if isinstance(names, dict):
        return data_yaml, {int(index): str(name) for index, name in names.items()}
    return data_yaml, {}


def parse_label_file(label_path: Path, class_names: dict[int, str]) -> tuple[list[dict[str, Any]], Counter[str]]:
    annotations: list[dict[str, Any]] = []
    issues: Counter[str] = Counter()
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        issues["empty_label"] += 1
        return annotations, issues

    for line_number, line in enumerate(lines, start=1):
        parts = line.split()
        annotation: dict[str, Any] = {"line_number": line_number, "raw": line}
        if len(parts) < 5:
            issues["invalid_label_columns"] += 1
            annotation["valid"] = False
            annotations.append(annotation)
            continue

        class_token = parts[0]
        try:
            class_id_float = float(class_token)
            if not class_id_float.is_integer():
                issues["non_integer_class_id"] += 1
                raise ValueError
            class_id = int(class_id_float)
        except ValueError:
            issues["invalid_class_id"] += 1
            annotation["valid"] = False
            annotations.append(annotation)
            continue

        try:
            x_center, y_center, width, height = (float(value) for value in parts[1:5])
        except ValueError:
            issues["invalid_coordinate_number"] += 1
            annotation["valid"] = False
            annotations.append(annotation)
            continue

        values = [x_center, y_center, width, height]
        if any(math.isnan(value) or math.isinf(value) for value in values):
            issues["nan_or_infinite_coordinate"] += 1
        if any(value < 0 or value > 1 for value in values):
            issues["coordinate_out_of_range"] += 1
        if width <= 0 or height <= 0:
            issues["non_positive_bbox_size"] += 1

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
            issues["bbox_outside_image"] += 1
        if class_names and class_id not in class_names:
            issues["unknown_class_id"] += 1

        annotation.update(
            {
                "valid": not any(
                    key in issues
                    for key in [
                        "nan_or_infinite_coordinate",
                        "coordinate_out_of_range",
                        "non_positive_bbox_size",
                        "bbox_outside_image",
                        "unknown_class_id",
                    ]
                ),
                "class_id": class_id,
                "class_name": class_names.get(class_id, str(class_id)),
                "x_center": x_center,
                "y_center": y_center,
                "bbox_width": width,
                "bbox_height": height,
                "bbox_area": width * height,
            }
        )
        annotations.append(annotation)
    return annotations, issues


def write_csv(path: Path, rows: list[dict[str, Any]], fallback_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else fallback_fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_duplicate_groups(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    duplicate_rows: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []

    exact_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("image_sha256"):
            exact_groups[str(row["image_sha256"])].append(row)

    union_find = UnionFind()
    phashes = [(str(row["image_path"]), str(row["phash"])) for row in rows if row.get("phash")]
    for path, _ in phashes:
        union_find.find(path)
    for index, (left_path, left_hash) in enumerate(phashes):
        for right_path, right_hash in phashes[index + 1 :]:
            if hamming(left_hash, right_hash) <= PHASH_THRESHOLD:
                union_find.union(left_path, right_path)

    group_id = 0
    for kind, groups in (
        ("exact", exact_groups.values()),
        ("phash", [members for members in union_find.groups().values() if len(members) > 1]),
    ):
        for members in groups:
            if len(members) <= 1:
                continue
            group_id += 1
            member_rows = members if kind == "exact" else [row for row in rows if row["image_path"] in members]
            splits = sorted({str(row["split"]) for row in member_rows})
            paths = sorted(str(row["image_path"]) for row in member_rows)
            duplicate_rows.append(
                {
                    "group_id": group_id,
                    "kind": kind,
                    "member_count": len(paths),
                    "splits": "|".join(splits),
                    "members": "|".join(paths),
                }
            )
            if len(splits) > 1:
                leakage_rows.append(
                    {
                        "group_id": group_id,
                        "kind": kind,
                        "splits": "|".join(splits),
                        "members": "|".join(paths),
                    }
                )
    return duplicate_rows, leakage_rows


def write_dataset_audit_doc(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Dataset Audit",
        "",
        f"- Dataset source metadata: {summary.get('data_yaml', 'not found')}",
        f"- Images: {summary['images']}",
        f"- Annotations: {summary['annotations']}",
        f"- Classes in data.yaml: {summary['classes_from_data_yaml']}",
        f"- Corrupt images: {summary['corrupt_images']}",
        f"- Missing labels: {summary['missing_labels']}",
        f"- Empty labels: {summary['empty_labels']}",
        f"- Unknown split images: {summary['unknown_split_images']}",
        f"- Exact/near duplicate groups: {summary['duplicate_groups']}",
        f"- Cross-split leakage groups: {summary['cross_split_leakage_groups']}",
        f"- Class imbalance ratio: {summary['class_imbalance_ratio']}",
        "",
        "Generated artifacts:",
        "",
        "- `artifacts/dataset/manifest.csv`",
        "- `artifacts/dataset/class_distribution.csv`",
        "- `artifacts/dataset/split_distribution.csv`",
        "- `artifacts/dataset/duplicate_groups.csv`",
        "- `artifacts/dataset/cross_split_leakage.csv`",
        "- `artifacts/dataset/bbox_statistics.csv`",
        "- `artifacts/dataset/summary.json`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_dataset(
    dataset_root: Path,
    output_dir: Path,
    doc_path: Path = Path("docs/audits/dataset-audit.md"),
) -> dict[str, Any]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_yaml, class_names = load_data_yaml(dataset_root)
    rows: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    split_annotation_counts: Counter[str] = Counter()
    bbox_values: defaultdict[tuple[str, str], list[float]] = defaultdict(list)
    issues: Counter[str] = Counter()

    images = sorted(path for path in dataset_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    labels = sorted(path for path in dataset_root.rglob("*.txt") if "labels" in path.parts)
    expected_image_labels = {yolo_label_path(path, dataset_root).resolve() for path in images}
    orphan_labels = [path for path in labels if path.resolve() not in expected_image_labels]
    issues["orphan_labels"] = len(orphan_labels)

    for image_path in images:
        relative_image = str(image_path.relative_to(dataset_root)).replace("\\", "/")
        split = split_name(image_path.relative_to(dataset_root))
        split_counts[split] += 1
        if split == "unknown":
            issues["unknown_split_images"] += 1

        label_path = yolo_label_path(image_path, dataset_root)
        width = height = 0
        image_sha = ""
        image_phash = ""
        corrupt = False
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
            if width <= 0 or height <= 0:
                issues["invalid_image_dimensions"] += 1
            image_sha = file_sha256(image_path)
            image_phash = phash(image_path)
        except (UnidentifiedImageError, OSError, ValueError):
            corrupt = True
            issues["corrupt_images"] += 1

        label_exists = label_path.exists()
        annotations: list[dict[str, Any]] = []
        label_issues: Counter[str] = Counter()
        if not label_exists:
            issues["missing_labels"] += 1
        elif not corrupt:
            annotations, label_issues = parse_label_file(label_path, class_names)
            issues.update(label_issues)

        image_class_ids = sorted({str(item["class_id"]) for item in annotations if "class_id" in item})
        for annotation in annotations:
            if "class_id" not in annotation:
                continue
            class_name = str(annotation["class_name"])
            class_counts[class_name] += 1
            split_annotation_counts[split] += 1
            bbox_values[(split, class_name)].append(float(annotation.get("bbox_area", 0.0)))
            annotation_rows.append(
                {
                    "image_path": relative_image,
                    "split": split,
                    "class_id": annotation["class_id"],
                    "class_name": class_name,
                    "bbox_area": annotation.get("bbox_area", 0.0),
                }
            )

        rows.append(
            {
                "image_path": relative_image,
                "split": split,
                "label_path": str(label_path.relative_to(dataset_root)).replace("\\", "/") if label_exists else "",
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 6) if height else 0,
                "annotation_count": len([item for item in annotations if "class_id" in item]),
                "class_ids": "|".join(image_class_ids),
                "corrupt": corrupt,
                "missing_label": not label_exists,
                "empty_label": bool(label_issues.get("empty_label")),
                "image_sha256": image_sha,
                "phash": image_phash,
            }
        )

    duplicate_rows, leakage_rows = build_duplicate_groups(rows)
    bbox_rows = [
        {
            "split": split,
            "class_name": class_name,
            "count": len(values),
            "bbox_area_min": min(values),
            "bbox_area_mean": mean(values),
            "bbox_area_max": max(values),
        }
        for (split, class_name), values in sorted(bbox_values.items())
        if values
    ]
    class_distribution_rows = [
        {"class_name": class_name, "annotation_count": count}
        for class_name, count in sorted(class_counts.items(), key=lambda item: item[0])
    ]
    split_distribution_rows = [
        {
            "split": split,
            "image_count": split_counts[split],
            "annotation_count": split_annotation_counts[split],
        }
        for split in sorted(split_counts)
    ]

    nonzero_class_counts = [count for count in class_counts.values() if count > 0]
    imbalance_ratio = (
        round(max(nonzero_class_counts) / min(nonzero_class_counts), 4) if len(nonzero_class_counts) >= 2 else 0
    )
    summary: dict[str, Any] = {
        "data_yaml": str(data_yaml.relative_to(dataset_root)).replace("\\", "/") if data_yaml else "not found",
        "images": len(images),
        "annotations": sum(class_counts.values()),
        "classes_from_data_yaml": len(class_names),
        "classes_observed": len(class_counts),
        "class_imbalance_ratio": imbalance_ratio,
        "duplicate_groups": len(duplicate_rows),
        "cross_split_leakage_groups": len(leakage_rows),
        "corrupt_images": issues["corrupt_images"],
        "missing_labels": issues["missing_labels"],
        "empty_labels": issues["empty_label"],
        "unknown_split_images": issues["unknown_split_images"],
        "issues": dict(sorted(issues.items())),
    }

    write_csv(output_dir / "manifest.csv", rows, ["image_path"])
    write_csv(output_dir / "annotations.csv", annotation_rows, ["image_path", "class_name"])
    write_csv(output_dir / "class_distribution.csv", class_distribution_rows, ["class_name", "annotation_count"])
    write_csv(
        output_dir / "split_distribution.csv", split_distribution_rows, ["split", "image_count", "annotation_count"]
    )
    write_csv(output_dir / "duplicate_groups.csv", duplicate_rows, ["group_id", "kind", "members"])
    write_csv(output_dir / "cross_split_leakage.csv", leakage_rows, ["group_id", "kind", "members"])
    write_csv(output_dir / "bbox_statistics.csv", bbox_rows, ["split", "class_name", "count"])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_dataset_audit_doc(doc_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit YOLO-format dataset for SiBiSee.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dataset"))
    args = parser.parse_args()
    summary = audit_dataset(args.dataset_dir, args.output_dir)
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
