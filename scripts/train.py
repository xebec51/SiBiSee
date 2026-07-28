from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from mlops_utils import environment_snapshot, write_json


def expand_env(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


def train(config_path: Path) -> Path:
    from ultralytics import YOLO

    config = expand_env(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    run_dir = Path(config["output_dir"]) / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "environment.json", environment_snapshot())
    write_json(run_dir / "training_config.json", config)

    model = YOLO(config["model"])
    results = model.train(**config["train_args"])
    metrics = getattr(results, "results_dict", {}) or {}
    write_json(run_dir / "metrics.json", metrics)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SiBiSee YOLO experiment.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_dir = train(args.config)
    print(f"run_dir: {run_dir}")


if __name__ == "__main__":
    main()
