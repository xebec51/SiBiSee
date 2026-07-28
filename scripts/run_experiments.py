from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mlops_utils import environment_snapshot, git_commit, sha256_file, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))


@dataclass(frozen=True)
class ModelCandidate:
    key: str
    label: str
    model_config: Path
    pretrained_weights: str


MODEL_CATALOG = {
    "baseline": ModelCandidate("baseline", "YOLOv8s baseline", Path("configs/models/yolov8s.yaml"), "yolov8s.pt"),
    "cbam": ModelCandidate("cbam", "YOLOv8s + CBAM", Path("configs/models/yolov8s-cbam.yaml"), "yolov8s.pt"),
    "light": ModelCandidate("light", "YOLOv8n lightweight", Path("configs/models/yolov8n.yaml"), "yolov8n.pt"),
}

STAGE_DEFAULTS = {
    "screening": {"epochs": 25, "patience": 8},
    "final": {"epochs": 150, "patience": 25},
}


def build_training_config(
    stage: str,
    candidate: ModelCandidate,
    seed: int,
    data_yaml: Path,
    runs_dir: Path,
    device: str,
    workers: int,
) -> dict[str, Any]:
    run_name = f"{stage}-{candidate.key}-seed{seed}"
    stage_defaults = STAGE_DEFAULTS[stage]
    return {
        "run_name": run_name,
        "output_dir": str(runs_dir),
        "model_config": str(candidate.model_config),
        "pretrained_weights": candidate.pretrained_weights,
        "train_args": {
            "data": str(data_yaml),
            "epochs": stage_defaults["epochs"],
            "imgsz": 640,
            "batch": 16,
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "seed": seed,
            "deterministic": True,
            "patience": stage_defaults["patience"],
            "device": device,
            "workers": workers,
            "project": str(runs_dir),
            "name": run_name,
        },
    }


def hash_optional(path: Path) -> str | None:
    return sha256_file(path) if path.exists() and path.is_file() else None


def architecture_summary(model_config: Path, image_size: int = 640) -> dict[str, Any]:
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    from sibisee.models import register_yolo_modules

    register_yolo_modules()
    model = YOLO(str(model_config))
    return {
        "model_config": str(model_config),
        "parameters": get_num_params(model.model),
        "gflops": get_flops(model.model, image_size),
    }


def collect_artifact_checksums(run_dir: Path) -> dict[str, str]:
    checksums = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            checksums[str(path.relative_to(run_dir))] = sha256_file(path)
    return checksums


def best_epoch_from_results(results_csv: Path) -> int | None:
    if not results_csv.exists():
        return None
    rows = list(csv.DictReader(results_csv.open(encoding="utf-8")))
    if not rows:
        return None
    metric_keys = [key for key in rows[0] if "mAP50-95" in key or "map50-95" in key.lower()]
    if not metric_keys:
        return None
    metric_key = metric_keys[0]
    best_index, _ = max(enumerate(rows), key=lambda item: float(item[1].get(metric_key) or 0.0))
    return best_index


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["run_name"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_inputs(
    stage: str, data_yaml: Path, dataset_manifest: Path, split_manifest: Path, output_dir: Path
) -> None:
    missing = [
        path for path in (data_yaml, dataset_manifest, split_manifest) if not path.exists() or not path.is_file()
    ]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Experiment input belum lengkap: {joined}")
    if stage == "final" and not (output_dir / "screening" / "summary.csv").exists():
        raise FileNotFoundError("Final training ditolak: jalankan dan review screening summary terlebih dahulu.")


def run_experiments(
    stage: str,
    model_keys: list[str],
    seeds: list[int],
    data_yaml: Path,
    output_dir: Path,
    dataset_manifest: Path,
    split_manifest: Path,
    device: str,
    workers: int,
    plan_only: bool = False,
) -> list[dict[str, Any]]:
    from train import train

    if not plan_only:
        validate_inputs(stage, data_yaml, dataset_manifest, split_manifest, output_dir)

    stage_dir = output_dir / stage
    runs_dir = stage_dir / "runs"
    config_dir = stage_dir / "configs"
    rows: list[dict[str, Any]] = []
    snapshot = environment_snapshot()

    for model_key in model_keys:
        candidate = MODEL_CATALOG[model_key]
        arch = architecture_summary(candidate.model_config)
        for seed in seeds:
            config = build_training_config(stage, candidate, seed, data_yaml, runs_dir, device, workers)
            config_path = config_dir / f"{config['run_name']}.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            run_dir = runs_dir / config["run_name"]
            metadata = {
                "stage": stage,
                "model": model_key,
                "model_label": candidate.label,
                "seed": seed,
                "git_sha": git_commit(),
                "environment": snapshot,
                "dataset_manifest_sha256": hash_optional(dataset_manifest),
                "split_manifest_sha256": hash_optional(split_manifest),
                "config_sha256": sha256_file(config_path),
                "architecture": arch,
                "plan_only": plan_only,
            }
            write_json(run_dir / "experiment_metadata.json", metadata)

            status = "planned"
            if not plan_only:
                last_checkpoint = run_dir / "weights" / "last.pt"
                best_checkpoint = run_dir / "weights" / "best.pt"
                if best_checkpoint.exists():
                    status = "existing"
                else:
                    if last_checkpoint.exists():
                        config["train_args"]["resume"] = str(last_checkpoint)
                        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
                    train(config_path)
                    status = "trained"
                write_json(run_dir / "artifact_checksums.json", collect_artifact_checksums(run_dir))

            rows.append(
                {
                    "run_name": config["run_name"],
                    "stage": stage,
                    "model": model_key,
                    "seed": seed,
                    "status": status,
                    "parameters": arch["parameters"],
                    "gflops": arch["gflops"],
                    "best_epoch": best_epoch_from_results(run_dir / "results.csv"),
                    "config": str(config_path),
                    "run_dir": str(run_dir),
                }
            )

    write_json(stage_dir / "summary.json", {"runs": rows})
    write_csv(stage_dir / "summary.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible SiBiSee multi-seed experiments.")
    parser.add_argument("--stage", choices=sorted(STAGE_DEFAULTS), required=True)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CATALOG), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--data", type=Path, default=Path(os.getenv("SIBISEE_DATASET_YAML", "")))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments"))
    parser.add_argument("--dataset-manifest", type=Path, default=Path("artifacts/dataset/manifest.csv"))
    parser.add_argument("--split-manifest", type=Path, default=Path("artifacts/dataset/split_manifest.csv"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    rows = run_experiments(
        args.stage,
        args.models,
        args.seeds,
        args.data,
        args.output_dir,
        args.dataset_manifest,
        args.split_manifest,
        args.device,
        args.workers,
        args.plan_only,
    )
    print(f"summary: {args.output_dir / args.stage / 'summary.csv'}")
    print(f"runs: {len(rows)}")


if __name__ == "__main__":
    main()
