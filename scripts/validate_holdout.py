from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
REQUIRED_FIELDS = {
    "relative_path",
    "class_name",
    "participant_id",
    "session_id",
    "device_label",
    "background",
    "lighting",
    "distance",
}
SENSITIVE_PATTERNS = {
    "email": re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE),
    "phone": re.compile(r"(?:\+?\d[\s().-]*){8,}"),
}
WEAK_CLASSES = ("Dua", "Sembilan", "Bodoh", "Makan", "Minum", "Masuk", "N", "Rumah", "Tidur", "Saya")


class HoldoutValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class HoldoutValidationConfig:
    min_participants: int = 3
    min_sessions: int = 3
    min_devices: int = 2
    min_backgrounds: int = 2
    min_lighting: int = 3
    min_distances: int = 3
    min_per_class: int = 10
    near_duplicate_threshold: int = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_class_names(data_yaml: Path) -> list[str]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = payload.get("names")
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise HoldoutValidationError(f"Class names tidak ditemukan di {data_yaml}.")


def read_metadata(holdout_dir: Path) -> list[dict[str, str]]:
    metadata_path = holdout_dir / "metadata.csv"
    if not metadata_path.exists():
        raise HoldoutValidationError("metadata.csv tidak ditemukan di holdout directory.")
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_fields = REQUIRED_FIELDS.difference(reader.fieldnames or [])
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise HoldoutValidationError(f"metadata.csv kehilangan field wajib: {missing}.")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def _safe_relative_path(holdout_dir: Path, relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise HoldoutValidationError(f"Path holdout harus relative dan tetap di root holdout: {relative_path}")
    resolved_root = holdout_dir.resolve()
    resolved_path = (resolved_root / path).resolve()
    if resolved_root != resolved_path and resolved_root not in resolved_path.parents:
        raise HoldoutValidationError(f"Path holdout keluar dari root holdout: {relative_path}")
    return resolved_path


def _phash(path: Path) -> int:
    import cv2
    import numpy as np

    with Image.open(path) as image:
        gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.float32)
    dct = cv2.dct(pixels)
    low_frequency = dct[:8, :8].flatten()
    comparable = low_frequency[1:]
    median = float(np.median(comparable))
    bits = comparable > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_image_hashes(dataset_dir: Path) -> dict[str, list[str]]:
    hashes: dict[str, list[str]] = defaultdict(list)
    if not dataset_dir.exists():
        return hashes
    for path in sorted(dataset_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            hashes[sha256_file(path)].append(path.relative_to(dataset_dir).as_posix())
    return hashes


def _cluster_near_duplicates(
    rows: list[dict[str, str]],
    image_paths: dict[str, Path],
    threshold: int,
) -> list[dict[str, str]]:
    parent = {row["relative_path"]: row["relative_path"] for row in rows}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)

    hashes = {relative_path: _phash(path) for relative_path, path in image_paths.items()}
    items = sorted(hashes.items())
    for left_index, (left_path, left_hash) in enumerate(items):
        for right_path, right_hash in items[left_index + 1 :]:
            if _hamming(left_hash, right_hash) <= threshold:
                union(left_path, right_path)

    grouped: dict[str, list[str]] = defaultdict(list)
    for relative_path in sorted(parent):
        grouped[find(relative_path)].append(relative_path)

    output_rows: list[dict[str, str]] = []
    group_id = 0
    for members in grouped.values():
        if len(members) < 2:
            continue
        group_id += 1
        for member in members:
            output_rows.append({"group_id": str(group_id), "relative_path": member})
    return output_rows


def validate_holdout(
    holdout_dir: Path,
    dataset_dir: Path,
    data_yaml: Path,
    output_dir: Path,
    config: HoldoutValidationConfig | None = None,
) -> dict[str, Any]:
    config = config or HoldoutValidationConfig()
    rows = read_metadata(holdout_dir)
    class_names = load_class_names(data_yaml)
    class_name_set = set(class_names)
    issues: list[str] = []

    if not rows:
        issues.append("metadata.csv tidak berisi sample holdout.")

    relative_paths = [row["relative_path"] for row in rows]
    duplicate_path_counts = {path: count for path, count in Counter(relative_paths).items() if count > 1}
    if duplicate_path_counts:
        issues.append("metadata.csv berisi relative_path duplikat.")

    image_paths: dict[str, Path] = {}
    image_hashes: dict[str, str] = {}
    image_errors: list[dict[str, str]] = []
    sensitive_rows: list[dict[str, str]] = []

    for row in rows:
        for field in REQUIRED_FIELDS:
            if not row.get(field):
                issues.append(f"Field {field} kosong pada row {row.get('relative_path', '<unknown>')}.")
        if row.get("class_name") not in class_name_set:
            issues.append(f"Unknown class_name pada holdout: {row.get('class_name')}")
        for field in ("participant_id", "session_id", "device_label", "background", "lighting", "distance", "notes"):
            value = row.get(field, "")
            if any(pattern.search(value) for pattern in SENSITIVE_PATTERNS.values()):
                sensitive_rows.append({"relative_path": row.get("relative_path", ""), "field": field})

        try:
            image_path = _safe_relative_path(holdout_dir, row["relative_path"])
        except HoldoutValidationError as exc:
            issues.append(str(exc))
            continue
        if not image_path.exists():
            image_errors.append({"relative_path": row["relative_path"], "error": "missing"})
            continue
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            image_errors.append({"relative_path": row["relative_path"], "error": "unsupported_extension"})
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError) as exc:
            image_errors.append({"relative_path": row["relative_path"], "error": exc.__class__.__name__})
            continue
        image_paths[row["relative_path"]] = image_path
        image_hashes[row["relative_path"]] = sha256_file(image_path)

    if sensitive_rows:
        issues.append("Metadata holdout memuat pola data pribadi yang tidak boleh disimpan.")
    if image_errors:
        issues.append("Holdout memiliki image yang hilang, corrupt, atau extension tidak didukung.")

    duplicate_hash_groups: dict[str, list[str]] = defaultdict(list)
    for relative_path, digest in image_hashes.items():
        duplicate_hash_groups[digest].append(relative_path)
    exact_duplicate_rows = [
        {"sha256": digest, "relative_path": relative_path}
        for digest, members in sorted(duplicate_hash_groups.items())
        if len(members) > 1
        for relative_path in sorted(members)
    ]
    if exact_duplicate_rows:
        issues.append("Holdout memiliki exact duplicate.")

    near_duplicate_rows = (
        _cluster_near_duplicates(rows, image_paths, config.near_duplicate_threshold) if image_paths else []
    )
    if near_duplicate_rows:
        issues.append("Holdout memiliki near-duplicate cluster.")

    dataset_hashes = _dataset_image_hashes(dataset_dir)
    overlap_rows: list[dict[str, str]] = []
    for relative_path, digest in image_hashes.items():
        for dataset_path in dataset_hashes.get(digest, []):
            overlap_rows.append(
                {
                    "holdout_relative_path": relative_path,
                    "dataset_relative_path": dataset_path,
                    "sha256": digest,
                }
            )
    if overlap_rows:
        issues.append("Holdout memiliki exact image overlap dengan dataset training/validation/test.")

    class_counts = Counter(row["class_name"] for row in rows)
    session_counts = Counter(row["session_id"] for row in rows)
    participants = {row["participant_id"] for row in rows if row["participant_id"]}
    sessions = {row["session_id"] for row in rows if row["session_id"]}
    devices = {row["device_label"] for row in rows if row["device_label"]}
    backgrounds = {row["background"] for row in rows if row["background"]}
    lighting = {row["lighting"] for row in rows if row["lighting"]}
    distances = {row["distance"] for row in rows if row["distance"]}
    classes_below_minimum = sorted(name for name in class_names if class_counts[name] < config.min_per_class)
    weak_classes_below_minimum = sorted(name for name in WEAK_CLASSES if class_counts[name] < config.min_per_class)

    minimums = {
        "participants": {"actual": len(participants), "minimum": config.min_participants},
        "sessions": {"actual": len(sessions), "minimum": config.min_sessions},
        "devices": {"actual": len(devices), "minimum": config.min_devices},
        "backgrounds": {"actual": len(backgrounds), "minimum": config.min_backgrounds},
        "lighting": {"actual": len(lighting), "minimum": config.min_lighting},
        "distances": {"actual": len(distances), "minimum": config.min_distances},
        "per_class_samples": {"actual_below_minimum": len(classes_below_minimum), "minimum": config.min_per_class},
    }
    if len(participants) < config.min_participants:
        issues.append("Jumlah participant holdout belum memenuhi minimum.")
    if len(sessions) < config.min_sessions:
        issues.append("Jumlah session holdout belum memenuhi minimum.")
    if len(devices) < config.min_devices:
        issues.append("Jumlah device holdout belum memenuhi minimum.")
    if len(backgrounds) < config.min_backgrounds:
        issues.append("Variasi background holdout belum memenuhi minimum.")
    if len(lighting) < config.min_lighting:
        issues.append("Variasi lighting holdout belum memenuhi minimum.")
    if len(distances) < config.min_distances:
        issues.append("Variasi distance holdout belum memenuhi minimum.")
    if classes_below_minimum:
        issues.append("Sebagian class belum memenuhi minimum sample per class.")

    class_rows = [
        {
            "class_name": name,
            "count": class_counts[name],
            "meets_minimum": class_counts[name] >= config.min_per_class,
            "is_weak_class": name in WEAK_CLASSES,
        }
        for name in class_names
    ]
    session_rows = [
        {"session_id": session_id, "count": count}
        for session_id, count in sorted(session_counts.items(), key=lambda item: item[0])
    ]

    summary: dict[str, Any] = {
        "holdout_sample_count": len(rows),
        "valid_image_count": len(image_paths),
        "class_count": len(class_names),
        "participant_count": len(participants),
        "session_count": len(sessions),
        "device_count": len(devices),
        "background_count": len(backgrounds),
        "lighting_count": len(lighting),
        "distance_count": len(distances),
        "classes_below_minimum": classes_below_minimum,
        "weak_classes_below_minimum": weak_classes_below_minimum,
        "duplicate_relative_path_count": len(duplicate_path_counts),
        "exact_duplicate_rows": len(exact_duplicate_rows),
        "near_duplicate_rows": len(near_duplicate_rows),
        "dataset_overlap_rows": len(overlap_rows),
        "image_error_count": len(image_errors),
        "sensitive_metadata_rows": len(sensitive_rows),
        "minimums": minimums,
        "status": "PASS" if not issues else "BLOCKED",
        "issues": sorted(set(issues)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output_dir / "class-coverage.csv", class_rows, ["class_name", "count", "meets_minimum", "is_weak_class"])
    _write_csv(output_dir / "session-coverage.csv", session_rows, ["session_id", "count"])
    _write_csv(output_dir / "duplicate-groups.csv", near_duplicate_rows, ["group_id", "relative_path"])
    _write_csv(
        output_dir / "dataset-overlap.csv",
        overlap_rows,
        ["holdout_relative_path", "dataset_relative_path", "sha256"],
    )
    _write_csv(output_dir / "image-errors.csv", image_errors, ["relative_path", "error"])
    _write_csv(output_dir / "sensitive-metadata.csv", sensitive_rows, ["relative_path", "field"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a private real-world SiBiSee holdout set.")
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/private/holdout-validation"))
    parser.add_argument("--min-participants", type=int, default=3)
    parser.add_argument("--min-sessions", type=int, default=3)
    parser.add_argument("--min-devices", type=int, default=2)
    parser.add_argument("--min-backgrounds", type=int, default=2)
    parser.add_argument("--min-lighting", type=int, default=3)
    parser.add_argument("--min-distances", type=int, default=3)
    parser.add_argument("--min-per-class", type=int, default=10)
    parser.add_argument("--near-duplicate-threshold", type=int, default=8)
    args = parser.parse_args()

    config = HoldoutValidationConfig(
        min_participants=args.min_participants,
        min_sessions=args.min_sessions,
        min_devices=args.min_devices,
        min_backgrounds=args.min_backgrounds,
        min_lighting=args.min_lighting,
        min_distances=args.min_distances,
        min_per_class=args.min_per_class,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    summary = validate_holdout(args.holdout_dir, args.dataset_dir, args.data, args.output_dir, config)
    print(f"output_dir: {args.output_dir}")
    print(f"status: {summary['status']}")
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
