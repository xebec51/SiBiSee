from __future__ import annotations

from sibisee.config import InferenceSettings
from sibisee.inference.detector import YoloDetector


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class FakeBoxes:
    cls = FakeTensor([0])
    conf = FakeTensor([0.9])
    xyxy = FakeTensor([[0, 0, 10, 10]])


class FakeResult:
    boxes = FakeBoxes()

    def plot(self):
        return "annotated"


class FakeModel:
    names = {0: "A"}

    def __init__(self):
        self.calls = 0

    def predict(self, *args, **kwargs):
        self.calls += 1
        return [FakeResult()]


def test_detector_annotates_from_single_model_call() -> None:
    model = FakeModel()
    detector = YoloDetector(model, InferenceSettings())

    result = detector.predict("frame", annotate=True)

    assert model.calls == 1
    assert result.annotated_frame == "annotated"
    assert result.detections[0].label == "A"
