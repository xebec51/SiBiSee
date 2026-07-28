from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_experiments import MODEL_CATALOG, build_training_config, run_experiments  # noqa: E402


def test_build_training_config_keeps_protocol_constant_except_seed_and_name() -> None:
    candidate = MODEL_CATALOG["baseline"]
    first = build_training_config("screening", candidate, 0, Path("data.yaml"), Path("runs"), "0", 2)
    second = build_training_config("screening", candidate, 42, Path("data.yaml"), Path("runs"), "0", 2)

    first_args = dict(first["train_args"])
    second_args = dict(second["train_args"])
    first_args.pop("seed")
    first_args.pop("name")
    second_args.pop("seed")
    second_args.pop("name")

    assert first_args == second_args
    assert first["pretrained_weights"] == second["pretrained_weights"] == "yolov8s.pt"
    assert first["model_config"] == second["model_config"]


def test_run_experiments_plan_only_writes_reproducible_plan(tmp_path: Path) -> None:
    rows = run_experiments(
        stage="screening",
        model_keys=["baseline"],
        seeds=[0, 42],
        data_yaml=tmp_path / "data.yaml",
        output_dir=tmp_path / "experiments",
        dataset_manifest=tmp_path / "manifest.csv",
        split_manifest=tmp_path / "split_manifest.csv",
        device="cpu",
        workers=0,
        plan_only=True,
    )

    summary_path = tmp_path / "experiments" / "screening" / "summary.csv"
    config_path = tmp_path / "experiments" / "screening" / "configs" / "screening-baseline-seed0.yaml"
    metadata_path = (
        tmp_path / "experiments" / "screening" / "runs" / "screening-baseline-seed0" / "experiment_metadata.json"
    )

    assert [row["status"] for row in rows] == ["planned", "planned"]
    assert summary_path.exists()
    assert config_path.exists()
    assert metadata_path.exists()

    summary_rows = list(csv.DictReader(summary_path.open(encoding="utf-8")))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert len(summary_rows) == 2
    assert config["train_args"]["seed"] == 0
    assert Path(config["model_config"]).as_posix() == "configs/models/yolov8s.yaml"
