# Screening Results

This document records the completed screening pass only. It is not a final multi-seed result, not an internal-test result, and not a production model promotion.

## Dataset

- Source: Roboflow `sibi-detection-nftzq/sibi-bieme`, version 2, YOLOv8 export
- Images: 5831
- Labels: 5831 files
- Annotations: 5832 boxes
- Classes: 49
- Download manifest SHA-256: `4e240d4d1673bea6c4ff03594a991dd2bceb177f9dc4d03a7c176b8866df55ad`
- Corrupt images: 0
- Missing labels: 0
- Empty labels: 0
- Invalid/orphan labels: 0
- Duplicate or near-duplicate groups: 365
- Cross-split leakage groups in original Roboflow split: 138

## Split

Leakage-aware split generated with seed 42 and duplicate-cluster grouping:

| Split | Images | Class coverage |
| --- | ---: | ---: |
| train | 3940 | 49/49 |
| val | 1047 | 49/49 |
| test | 844 | 49/49 |

Subject-independent evaluation cannot be proven because signer/person metadata is not present in the export. Some validation and test classes have only one sample after duplicate-cluster preservation; per-class metrics for those classes are high variance.

## Configuration

- Stage: screening
- Seed: 0
- Epochs: 25
- Image size: 640
- Batch size: 16
- Optimizer: SGD
- Learning rate: 0.01
- LR final factor: 0.01
- Patience: 8
- Workers: 4
- Device: RTX 4060 Ti
- Baseline and CBAM pretrained source: `yolov8s.pt`
- Lightweight pretrained source: `yolov8n.pt`

## Validation Metrics

| Model | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Parameters | GFLOPs | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8s baseline | 25 | 0.7644 | 0.7712 | 0.8931 | 0.7695 | 11,166,560 | 28.82 | 1208.08 |
| YOLOv8s-CBAM | 25 | 0.8084 | 0.7640 | 0.9084 | 0.7800 | 11,199,426 | 28.87 | 1214.05 |
| YOLOv8n lightweight | 25 | 0.6637 | 0.7827 | 0.8718 | 0.7417 | 3,157,200 | 8.86 | 720.22 |

All candidates had best mAP50-95 at epoch 25, so the screening budget may be too short for final conclusions.

## Weakest Classes

Lowest validation mAP50-95 classes:

| Model | Classes |
| --- | --- |
| YOLOv8s baseline | Enam, Satu, Sembilan, Q, R, P, B |
| YOLOv8s-CBAM | Sembilan, Enam, Satu, N, Masuk, B, M, F, G |
| YOLOv8n lightweight | Sembilan, Enam, Empat, Masuk, R, F, N, P, S |

Several of these classes have only one validation instance, so they should be reviewed with caution during final test and holdout evaluation.

## Latency

PyTorch benchmark, batch size 1, image size 640, 10 warm-up iterations, 100 measured iterations:

| Model | Device | Mean ms | Median ms | p95 ms | FPS | Artifact size |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8s baseline | RTX 4060 Ti | 6.36 | 6.18 | 8.14 | 157.15 | 22,552,291 bytes |
| YOLOv8s-CBAM | RTX 4060 Ti | 6.51 | 6.40 | 7.58 | 153.57 | 22,621,396 bytes |
| YOLOv8n lightweight | RTX 4060 Ti | 6.45 | 6.26 | 7.98 | 155.15 | 6,264,867 bytes |
| YOLOv8s baseline | CPU | 70.88 | 69.03 | 84.90 | 14.11 | 22,552,291 bytes |
| YOLOv8s-CBAM | CPU | 77.61 | 69.71 | 107.48 | 12.89 | 22,621,396 bytes |
| YOLOv8n lightweight | CPU | 52.40 | 52.96 | 60.04 | 19.09 | 6,264,867 bytes |

## Candidate Decision

Recommended final-stage candidates:

- `cbam`: strongest validation accuracy in screening.
- `light`: much smaller artifact and materially faster CPU inference, while retaining reasonable validation accuracy.

`baseline` is retained as reference but is not Pareto-preferred in this screening result: it is less accurate than CBAM and less efficient than the lightweight model.

Final multi-seed training, internal test evaluation, ONNX parity, real-world holdout, and production artifact promotion are still not complete.

Use the same experiment output root as the validated screening pass when launching final training:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiments.py `
  --stage final `
  --models cbam light `
  --seeds 0 42 1337 `
  --data artifacts\dataset\sibisee_splits.yaml `
  --dataset-manifest artifacts\dataset\manifest.csv `
  --split-manifest artifacts\dataset\split_manifest.csv `
  --output-dir artifacts\experiments-screening-v2 `
  --device 0 `
  --workers 4
```
