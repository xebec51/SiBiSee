from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from mlops_utils import environment_snapshot, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def expand_env(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


def detect_layer_index(torch_model) -> int | None:
    from ultralytics.nn.modules import Detect

    for index, module in enumerate(torch_model.model):
        if isinstance(module, Detect):
            return index
    return None


def load_pretrained_weights(model, weights_path: str | Path) -> dict[str, int | str]:
    from ultralytics.nn.tasks import attempt_load_one_weight
    from ultralytics.utils.torch_utils import intersect_dicts

    loaded_weights, _ = attempt_load_one_weight(weights_path)
    candidate_state = loaded_weights.float().state_dict()
    target_state = model.model.state_dict()
    matched_state = intersect_dicts(candidate_state, target_state)
    name_matched_count = len(matched_state)

    source_detect_index = detect_layer_index(loaded_weights)
    target_detect_index = detect_layer_index(model.model)
    remapped_detect_count = 0
    if (
        source_detect_index is not None
        and target_detect_index is not None
        and source_detect_index != target_detect_index
    ):
        source_prefix = f"model.{source_detect_index}."
        target_prefix = f"model.{target_detect_index}."
        for key, value in candidate_state.items():
            if not key.startswith(source_prefix):
                continue
            remapped_key = target_prefix + key.removeprefix(source_prefix)
            if (
                remapped_key not in matched_state
                and remapped_key in target_state
                and value.shape == target_state[remapped_key].shape
            ):
                matched_state[remapped_key] = value
                remapped_detect_count += 1

    model.model.load_state_dict(matched_state, strict=False)
    model.overrides["pretrained"] = str(weights_path)
    return {
        "source": str(weights_path),
        "name_matched_items": name_matched_count,
        "remapped_detect_items": remapped_detect_count,
        "matched_items": len(matched_state),
        "target_items": len(target_state),
        "unmatched_items": len(target_state) - len(matched_state),
    }


def train(config_path: Path) -> Path:
    from ultralytics import YOLO

    from sibisee.models import register_yolo_modules

    register_yolo_modules()
    config = expand_env(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    run_dir = Path(config["output_dir"]) / config["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "environment.json", environment_snapshot())
    write_json(run_dir / "training_config.json", config)

    model_source = config.get("model_config") or config["model"]
    model = YOLO(model_source)
    pretrained_weights = config.get("pretrained_weights")
    if pretrained_weights:
        transfer_summary = load_pretrained_weights(model, pretrained_weights)
        write_json(run_dir / "pretrained_transfer.json", transfer_summary)
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
