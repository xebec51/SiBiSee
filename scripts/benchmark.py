from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from mlops_utils import write_json


def benchmark(
    model_path: Path, output_dir: Path, image_size: int = 640, iterations: int = 100, warmup: int = 10
) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    sample = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    for _ in range(warmup):
        model.predict(sample, verbose=False)
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        model.predict(sample, verbose=False)
        latencies.append((time.perf_counter() - start) * 1000)
    payload = {
        "iterations": iterations,
        "warmup": warmup,
        "image_size": image_size,
        "mean_ms": statistics.mean(latencies),
        "median_ms": statistics.median(latencies),
        "p95_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies),
        "fps_mean": 1000 / statistics.mean(latencies),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", payload)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SiBiSee model latency.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmarks"))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    output_dir = benchmark(args.model, args.output_dir, args.image_size, args.iterations, args.warmup)
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
