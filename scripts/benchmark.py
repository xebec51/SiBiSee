from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from mlops_utils import environment_snapshot, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def p95(values: list[float]) -> float:
    if not values:
        raise ValueError("Tidak ada latency yang bisa diringkas.")
    if len(values) < 20:
        return max(values)
    return statistics.quantiles(values, n=20)[18]


def summarize_latencies(latencies_ms: list[float]) -> dict[str, float | int]:
    mean_ms = statistics.mean(latencies_ms)
    return {
        "iterations": len(latencies_ms),
        "mean_ms": mean_ms,
        "median_ms": statistics.median(latencies_ms),
        "p95_ms": p95(latencies_ms),
        "fps_mean": 1000 / mean_ms,
    }


def measure(callable_obj, iterations: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        callable_obj()
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        callable_obj()
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def benchmark_pytorch(
    model_path: Path,
    image_size: int,
    iterations: int,
    warmup: int,
    device: str,
    include_annotation: bool,
) -> dict[str, Any]:
    from ultralytics import YOLO

    from sibisee.models import register_yolo_modules

    register_yolo_modules()
    model = YOLO(str(model_path))
    sample = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    prediction_kwargs = {"verbose": False, "device": device}
    inference_latencies = measure(lambda: model.predict(sample, **prediction_kwargs), iterations, warmup)

    payload: dict[str, Any] = {
        "backend": "pytorch",
        "device": device,
        "image_size": image_size,
        "warmup": warmup,
        "pure_inference": summarize_latencies(inference_latencies),
    }
    try:
        import torch

        if torch.cuda.is_available() and device != "cpu":
            payload["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    except Exception as exc:
        payload["memory_error"] = exc.__class__.__name__

    if include_annotation:
        result = model.predict(sample, **prediction_kwargs)[0]
        annotation_latencies = measure(lambda: result.plot(), iterations, warmup)
        payload["annotation"] = summarize_latencies(annotation_latencies)
    return payload


def benchmark_onnx(model_path: Path, image_size: int, iterations: int, warmup: int) -> dict[str, Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    sample = np.zeros((1, 3, image_size, image_size), dtype=np.float32)
    latencies = measure(lambda: session.run(None, {input_name: sample}), iterations, warmup)
    return {
        "backend": "onnxruntime",
        "device": "cpu",
        "image_size": image_size,
        "warmup": warmup,
        "pure_inference": summarize_latencies(latencies),
    }


def benchmark(
    model_path: Path,
    output_dir: Path,
    image_size: int = 640,
    iterations: int = 100,
    warmup: int = 10,
    backend: str = "pytorch",
    device: str = "cpu",
    include_annotation: bool = True,
) -> Path:
    if backend == "pytorch":
        payload = benchmark_pytorch(model_path, image_size, iterations, warmup, device, include_annotation)
    elif backend == "onnx":
        payload = benchmark_onnx(model_path, image_size, iterations, warmup)
    else:
        raise ValueError(f"Backend tidak didukung: {backend}")
    payload["environment"] = environment_snapshot()
    payload["model_path"] = str(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", payload)
    (output_dir / "latency_summary.pretty.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SiBiSee model latency.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmarks"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--backend", choices=["pytorch", "onnx"], default="pytorch")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-annotation", action="store_true")
    args = parser.parse_args()
    output_dir = benchmark(
        args.model,
        args.output_dir,
        args.image_size,
        args.iterations,
        args.warmup,
        args.backend,
        args.device,
        not args.no_annotation,
    )
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
