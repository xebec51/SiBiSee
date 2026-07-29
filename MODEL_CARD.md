# Model Card

## Intended Use

Model SiBiSee ditujukan untuk pengenalan gestur SIBI terisolasi pada aplikasi portfolio/research. Output harus diperlakukan sebagai prediksi bantu yang dapat salah.

## Out of Scope

Model tidak boleh diklaim menerjemahkan tata bahasa SIBI lengkap, percakapan kontinu, atau semua variasi signer. Gestur dinamis memerlukan model temporal terpisah. Model ini bukan sistem keselamatan, medis, legal, atau aksesibilitas kritis.

## Artifact

- Production artifact: `models/best.pt.enc`
- SHA-256 encrypted artifact: `04f9fb0f21e42dc01e3832f2e786aa92b281400035bb488d9ebbe5e7b146cd23`
- Metadata: `models/best.metadata.json`
- Artifact size: 30,176,460 bytes
- Backend: PyTorch
- Runtime: Ultralytics YOLO with CPU or CUDA-capable PyTorch environment
- ONNX export/parity: NOT RUN - not required by the selected PyTorch deployment backend
- Real-world holdout: NOT RUN - intentionally excluded from the current release scope

## Evaluation

No production artifact has been replaced.

Final validation, leakage-aware split, seeds 0/42/1337:

| Model | Precision | Recall | mAP50 | mAP50-95 | Parameters | GFLOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8s-CBAM mean | 0.8492 | 0.8788 | 0.9484 | 0.8389 | 11,199,426 | 28.87 |
| YOLOv8n lightweight mean | 0.8510 | 0.8525 | 0.9511 | 0.8330 | 3,157,200 | 8.86 |

Selected production candidate before test evaluation: YOLOv8s-CBAM seed 42. Internal test result for that checkpoint: precision 0.9400, recall 0.9295, mAP50 0.9659, mAP50-95 0.8472.

Production PyTorch benchmark, batch size 1, image size 640, 10 warm-up iterations, and 100 measured iterations:

| Device | Mean ms | Median ms | p95 ms | FPS |
| --- | ---: | ---: | ---: | ---: |
| RTX 4060 Ti | 10.29 | 11.07 | 11.63 | 97.22 |
| CPU | 70.14 | 68.00 | 84.58 | 14.26 |

The lightweight model is the strongest deployment-efficiency candidate because it has much lower CPU latency and a much smaller checkpoint. Production promotion for this release proceeds without real-world holdout evidence by explicit scope decision. The model must not be described as validated for new participants, cameras, lighting, backgrounds, distances, or devices.

## Limitations

- Generalisasi lintas signer belum dapat dibuktikan tanpa split berbasis signer/session.
- The current split is duplicate-cluster-aware, but not signer-independent.
- Several validation classes have only one sample after leakage-aware splitting, making those per-class estimates noisy.
- Kelas dengan bentuk tangan mirip memerlukan per-class analysis pada test set.
- Lighting, blur, occlusion, dan background kompleks dapat menurunkan confidence.
- Internal-test metrics do not guarantee real-world performance.
