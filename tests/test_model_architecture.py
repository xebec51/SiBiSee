from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params

from sibisee.models import CBAM, register_yolo_modules


def detect_input_channels(model: YOLO) -> list[int]:
    detect = model.model.model[-1]
    return [branch[0].conv.in_channels for branch in detect.cv2]


def test_yolov8s_cbam_constructs_with_preserved_detect_channels() -> None:
    register_yolo_modules()
    baseline = YOLO("configs/models/yolov8s.yaml")
    candidate = YOLO("configs/models/yolov8s-cbam.yaml")

    assert any(isinstance(module, CBAM) for module in candidate.model.modules())
    assert detect_input_channels(candidate) == detect_input_channels(baseline)
    assert get_num_params(candidate.model) > get_num_params(baseline.model)
    assert get_num_params(candidate.model) - get_num_params(baseline.model) < 70_000

    dummy = torch.zeros(1, 3, 64, 64)
    baseline.model.eval()
    candidate.model.eval()
    with torch.no_grad():
        baseline.model(dummy)
        candidate.model(dummy)

    baseline.info(verbose=False)
    candidate.info(verbose=False)


def test_training_configs_reference_existing_model_configs() -> None:
    for config_path in Path("configs/training").glob("*.yaml"):
        text = config_path.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("model_config:"):
                model_config = Path(line.split(":", maxsplit=1)[1].strip())
                assert model_config.exists()
