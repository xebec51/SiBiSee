# Benchmark

Benchmark deployment dijalankan dengan warm-up, batch size 1, input size produksi, dan minimal 100 measured iterations. Script memisahkan pure inference dari annotation time.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-benchmark.txt

.\.venv\Scripts\python.exe scripts\benchmark.py `
  --model path\to\best.pt `
  --backend pytorch `
  --device 0 `
  --iterations 100 `
  --warmup 10 `
  --output-dir artifacts\benchmarks\gpu

.\.venv\Scripts\python.exe scripts\benchmark.py `
  --model path\to\best.pt `
  --backend pytorch `
  --device cpu `
  --iterations 100 `
  --warmup 10 `
  --output-dir artifacts\benchmarks\cpu

.\.venv\Scripts\python.exe scripts\export_model.py `
  --model path\to\best.pt `
  --format onnx `
  --output-dir artifacts\models\onnx

.\.venv\Scripts\python.exe scripts\benchmark.py `
  --model artifacts\models\onnx\best.onnx `
  --backend onnx `
  --iterations 100 `
  --warmup 10 `
  --no-annotation `
  --output-dir artifacts\benchmarks\onnx
```

Setiap `summary.json` mencatat mean, median, p95, FPS, environment snapshot, dan peak CUDA memory bila tersedia.

## Screening Benchmark

Screening benchmark was run on July 29, 2026 with PyTorch, batch size 1, image size 640, 10 warm-up iterations, and 100 measured iterations. Pure inference is reported separately from annotation.

Environment:

- GPU: NVIDIA GeForce RTX 4060 Ti
- PyTorch: 2.5.1+cu121
- CUDA runtime reported by PyTorch: 12.1
- Ultralytics: 8.3.40

| Model | Device | Mean ms | Median ms | p95 ms | FPS | Artifact size |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8s baseline | RTX 4060 Ti | 6.36 | 6.18 | 8.14 | 157.15 | 22,552,291 bytes |
| YOLOv8s-CBAM | RTX 4060 Ti | 6.51 | 6.40 | 7.58 | 153.57 | 22,621,396 bytes |
| YOLOv8n lightweight | RTX 4060 Ti | 6.45 | 6.26 | 7.98 | 155.15 | 6,264,867 bytes |
| YOLOv8s baseline | CPU | 70.88 | 69.03 | 84.90 | 14.11 | 22,552,291 bytes |
| YOLOv8s-CBAM | CPU | 77.61 | 69.71 | 107.48 | 12.89 | 22,621,396 bytes |
| YOLOv8n lightweight | CPU | 52.40 | 52.96 | 60.04 | 19.09 | 6,264,867 bytes |

ONNX was not benchmarked in this screening pass because export parity validation has not been completed.

Model baru hanya boleh dipromosikan bila memenuhi salah satu:

1. Mean test mAP50-95 naik sekitar 0.3 percentage point tanpa regresi per-class recall yang tidak dapat diterima dan latency tidak memburuk material.
2. Mean test mAP50-95 berada dalam sekitar 0.2 percentage point dari model terbaik, tetapi CPU latency atau ukuran artifact membaik minimal sekitar 20%.
3. Real-world holdout membaik jelas dan trade-off terdokumentasi.

Belum ada benchmark final multi-seed karena final training, internal test evaluation, holdout evaluation, and production promotion have not been completed.
