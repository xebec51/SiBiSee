from __future__ import annotations

import argparse
import os
from pathlib import Path

from audit_dataset import audit_dataset


def discover_dataset_dir(explicit: Path | None = None) -> Path:
    candidates = [
        explicit,
        Path("SIBI-2"),
        Path("sibi-bieme-2"),
        Path("../SIBI-2"),
        Path("../sibi-bieme-2"),
        Path(os.environ["SIBISEE_DATASET_DIR"]) if os.getenv("SIBISEE_DATASET_DIR") else None,
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError("Dataset tidak ditemukan. Set SIBISEE_DATASET_DIR atau berikan path eksplisit.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SiBiSee dataset manifest.")
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/dataset"))
    args = parser.parse_args()
    try:
        dataset_dir = discover_dataset_dir(args.dataset_dir)
        summary = audit_dataset(dataset_dir, args.output_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"dataset_dir: {dataset_dir}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
