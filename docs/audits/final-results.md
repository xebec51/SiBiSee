# Final Training Results

This document records the completed final multi-seed training pass and internal-test evaluation. It does not claim real-world generalization.

## Protocol

- Candidates: `cbam`, `light`
- Seeds: 0, 42, 1337
- Epoch budget: 150
- Early stopping patience: 25
- Image size: 640
- Batch size: 16
- Optimizer: SGD
- Learning rate: 0.01
- LR final factor: 0.01
- Device: NVIDIA GeForce RTX 4060 Ti
- Dataset split: leakage-aware split generated with seed 42
- Selection metric: validation mAP50-95 before any internal test evaluation

## Validation Summary

| Model | Seed | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Parameters | GFLOPs | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CBAM | 0 | 108 | 0.80416 | 0.90791 | 0.94290 | 0.83387 | 11,199,426 | 28.87 | 6417.55 |
| CBAM | 42 | 85 | 0.84806 | 0.86545 | 0.95750 | 0.84158 | 11,199,426 | 28.87 | 5322.99 |
| CBAM | 1337 | 95 | 0.89528 | 0.86318 | 0.94487 | 0.84113 | 11,199,426 | 28.87 | 5800.63 |
| Lightweight | 0 | 108 | 0.81960 | 0.85869 | 0.95223 | 0.84000 | 3,157,200 | 8.86 | 3806.58 |
| Lightweight | 42 | 80 | 0.86472 | 0.84481 | 0.95200 | 0.81857 | 3,157,200 | 8.86 | 3000.39 |
| Lightweight | 1337 | 104 | 0.86864 | 0.85397 | 0.94892 | 0.84053 | 3,157,200 | 8.86 | 3689.89 |

Aggregate validation mAP50-95:

| Model | Mean | Std dev |
| --- | ---: | ---: |
| CBAM | 0.83886 | 0.00433 |
| Lightweight | 0.83303 | 0.01253 |

CBAM was selected before internal test evaluation because it had the best mean validation mAP50-95 and the best individual validation checkpoint. The selected checkpoint is `final-cbam-seed42`.

## Internal Test

Internal test was run once after model selection on `final-cbam-seed42`.

| Metric | Value |
| --- | ---: |
| Precision | 0.94001 |
| Recall | 0.92946 |
| mAP50 | 0.96593 |
| mAP50-95 | 0.84722 |

Weakest internal-test classes by mAP50-95:

| Class | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| Dua | 1.00000 | 0.00000 | 0.08292 | 0.07463 |
| Sembilan | 0.65683 | 0.84615 | 0.80776 | 0.68976 |
| Bodoh | 0.95684 | 1.00000 | 0.99500 | 0.72756 |
| Makan | 0.92380 | 1.00000 | 0.99500 | 0.72849 |
| Minum | 0.98403 | 1.00000 | 0.99500 | 0.74854 |
| Masuk | 0.98855 | 1.00000 | 0.99500 | 0.77139 |
| N | 0.86604 | 0.95455 | 0.87695 | 0.78449 |
| Rumah | 0.84644 | 1.00000 | 0.99500 | 0.79600 |
| Tidur | 0.99349 | 1.00000 | 0.99500 | 0.80465 |
| Saya | 0.97098 | 1.00000 | 0.99500 | 0.80998 |

`Dua`, `Cinta`, `Empat`, `Maaf`, `Rumah`, and `Z` have one internal-test sample each. Those per-class values have high variance and need real-world holdout review.

## Final Benchmark

PyTorch benchmark, batch size 1, image size 640, 10 warm-up iterations, and 100 measured iterations:

| Model | Device | Mean ms | Median ms | p95 ms | FPS | Artifact size |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CBAM seed 42 | RTX 4060 Ti | 10.59 | 11.20 | 12.30 | 94.44 | 22,632,276 bytes |
| CBAM seed 42 | CPU | 73.80 | 68.12 | 102.80 | 13.55 | 22,632,276 bytes |
| Lightweight seed 1337 | RTX 4060 Ti | 10.46 | 10.85 | 11.64 | 95.57 | 6,278,179 bytes |
| Lightweight seed 1337 | CPU | 37.70 | 36.58 | 47.26 | 26.52 | 6,278,179 bytes |

Annotation time was measured separately from pure inference and stayed around 0.32-0.35 ms.

## Decision

CBAM seed 42 is the best accuracy candidate from the completed protocol and is the recommended model for a production-candidate package.

The lightweight model remains valuable for CPU deployment: it is much smaller and materially faster on CPU, but its mean validation mAP50-95 is 0.00583 lower than CBAM. It should only replace CBAM if deployment constraints outweigh that accuracy gap or if a real-world holdout shows a clearer trade-off.

## Release Scope

- Production backend: PyTorch.
- ONNX export/parity: NOT RUN - not required by the selected PyTorch deployment backend.
- Real-world holdout: NOT RUN - intentionally excluded from the current release scope.
- Real-world generalization: NOT CLAIMED.

This release is a portfolio/research application for isolated SIBI sign recognition. It is not a complete sign-language translator and must not be used as a safety-critical or accessibility-critical system.

## Release Gate Status

- Production encrypted artifact: PASS.
- Clean-start compatibility and app startup smoke: PASS.
- Streamlit/deployment secrets update: BLOCKED - requires manual dashboard action.
- Deployment smoke: BLOCKED - no public deployment URL or connected deployment control was available.
- No tag or deployment was created.
