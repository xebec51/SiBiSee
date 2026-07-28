from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def average_hash(path: Path) -> str:
    with Image.open(path) as image:
        image = image.convert("L").resize((8, 8))
        pixels = list(image.getdata())
    mean = sum(pixels) / len(pixels)
    return "".join("1" if pixel >= mean else "0" for pixel in pixels)


def yolo_label_path(image_path: Path, dataset_root: Path) -> Path:
    parts = list(image_path.relative_to(dataset_root).parts)
    if "images" in parts:
        parts[parts.index("images")] = "labels"
    return dataset_root.joinpath(*parts).with_suffix(".txt")


def audit_dataset(dataset_root: Path, output_dir: Path) -> dict[str, int]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str | int | float | bool]] = []
    class_counts: Counter[str] = Counter()
    exact_hashes: defaultdict[str, list[str]] = defaultdict(list)
    perceptual_hashes: defaultdict[str, list[str]] = defaultdict(list)
    issues: Counter[str] = Counter()

    images = sorted(path for path in dataset_root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    for image_path in images:
        split = next(
            (part for part in image_path.parts if part in {"train", "valid", "val", "test", "holdout"}), "unknown"
        )
        label_path = yolo_label_path(image_path, dataset_root)
        width = height = 0
        corrupt = False
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
            exact_hashes[file_sha256(image_path)].append(str(image_path))
            perceptual_hashes[average_hash(image_path)].append(str(image_path))
        except (UnidentifiedImageError, OSError):
            corrupt = True
            issues["corrupt_image"] += 1

        label_exists = label_path.exists()
        label_empty = False
        invalid_label = False
        annotation_count = 0
        image_class_ids: set[str] = set()
        if not label_exists:
            issues["missing_label"] += 1
        else:
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            label_empty = len(lines) == 0
            if label_empty:
                issues["empty_label"] += 1
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    invalid_label = True
                    continue
                class_id, *coords = parts[:5]
                try:
                    values = [float(value) for value in coords]
                    if any(value < 0 or value > 1 for value in values):
                        invalid_label = True
                    class_counts[class_id] += 1
                    image_class_ids.add(class_id)
                    annotation_count += 1
                except ValueError:
                    invalid_label = True
            if invalid_label:
                issues["invalid_yolo_label"] += 1

        rows.append(
            {
                "image_path": str(image_path.relative_to(dataset_root)),
                "split": split,
                "label_path": str(label_path.relative_to(dataset_root)) if label_exists else "",
                "width": width,
                "height": height,
                "aspect_ratio": round(width / height, 6) if height else 0,
                "annotation_count": annotation_count,
                "class_ids": "|".join(sorted(image_class_ids)),
                "corrupt": corrupt,
                "missing_label": not label_exists,
                "empty_label": label_empty,
                "invalid_label": invalid_label,
            }
        )

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["image_path"])
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "class_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "annotation_count"])
        writer.writeheader()
        for class_id, count in sorted(
            class_counts.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]
        ):
            writer.writerow({"class_id": class_id, "annotation_count": count})

    duplicate_rows = []
    for kind, groups in {"exact": exact_hashes, "perceptual": perceptual_hashes}.items():
        for digest, members in groups.items():
            if len(members) > 1:
                duplicate_rows.append({"kind": kind, "hash": digest, "members": "|".join(members)})
    with (output_dir / "duplicate_groups.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "hash", "members"])
        writer.writeheader()
        writer.writerows(duplicate_rows)

    summary = {
        "images": len(images),
        "annotations": sum(class_counts.values()),
        "classes": len(class_counts),
        "duplicate_groups": len(duplicate_rows),
        **issues,
    }
    return dict(summary)


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
