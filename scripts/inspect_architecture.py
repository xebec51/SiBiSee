from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

from mlops_utils import write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def detect_input_channels(model: Any) -> list[int]:
    detect = model.model.model[-1]
    return [branch[0].conv.in_channels for branch in detect.cv2]


def inspect_architecture(
    baseline_yaml: Path,
    candidate_yaml: Path,
    output_path: Path,
    image_size: int = 640,
) -> dict[str, Any]:
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    from sibisee.models import register_yolo_modules

    register_yolo_modules()
    baseline = YOLO(str(baseline_yaml))
    candidate = YOLO(str(candidate_yaml))
    baseline.info(verbose=True)
    candidate.info(verbose=True)

    dummy = torch.zeros(1, 3, 64, 64)
    baseline.model.eval()
    candidate.model.eval()
    with torch.no_grad():
        baseline.model(dummy)
        candidate.model(dummy)

    payload = {
        "image_size": image_size,
        "baseline": {
            "config": str(baseline_yaml),
            "parameters": get_num_params(baseline.model),
            "gflops": get_flops(baseline.model, image_size),
            "detect_input_channels": detect_input_channels(baseline),
        },
        "candidate": {
            "config": str(candidate_yaml),
            "parameters": get_num_params(candidate.model),
            "gflops": get_flops(candidate.model, image_size),
            "detect_input_channels": detect_input_channels(candidate),
        },
    }
    payload["delta"] = {
        "parameters": payload["candidate"]["parameters"] - payload["baseline"]["parameters"],
        "gflops": payload["candidate"]["gflops"] - payload["baseline"]["gflops"],
        "detect_input_channels_equal": payload["candidate"]["detect_input_channels"]
        == payload["baseline"]["detect_input_channels"],
    }
    write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect baseline and CBAM YOLOv8s architectures.")
    parser.add_argument("--baseline", type=Path, default=Path("configs/models/yolov8s.yaml"))
    parser.add_argument("--candidate", type=Path, default=Path("configs/models/yolov8s-cbam.yaml"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/models/architecture_smoke.json"))
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args()
    payload = inspect_architecture(args.baseline, args.candidate, args.output, args.image_size)
    print(f"output: {args.output}")
    print(f"baseline_parameters: {payload['baseline']['parameters']}")
    print(f"candidate_parameters: {payload['candidate']['parameters']}")
    print(f"parameter_delta: {payload['delta']['parameters']}")
    print(f"detect_input_channels_equal: {payload['delta']['detect_input_channels_equal']}")


if __name__ == "__main__":
    main()
