from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def compare_runs(run_dirs: list[Path], output_path: Path) -> None:
    rows = []
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {"run": run_dir.name}
        row.update(metrics)
        rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare SiBiSee experiment metrics.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks/model_comparison.csv"))
    args = parser.parse_args()
    compare_runs(args.run_dirs, args.output)
    print(f"comparison: {args.output}")


if __name__ == "__main__":
    main()
