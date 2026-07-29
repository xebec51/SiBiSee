# Model Card

## Intended Use

Model SiBiSee ditujukan untuk pengenalan gestur SIBI terisolasi pada aplikasi demo/assistive prediction. Output harus diperlakukan sebagai prediksi bantu yang dapat salah.

## Out of Scope

Model tidak boleh diklaim menerjemahkan tata bahasa SIBI lengkap, percakapan kontinu, atau semua variasi signer. Gestur dinamis memerlukan model temporal terpisah.

## Artifact

- Production artifact: `models/best.pt.enc`
- SHA-256 encrypted artifact: `9f58c1af732e6817efb3776842667d72d98fb37c9a336a049fbd1d5b19da8661`
- Runtime: Ultralytics YOLO with PyTorch CUDA-capable environment

## Evaluation

Current evidence is screening only, not final model selection. No production artifact has been replaced.

Validation screening on the leakage-aware split, seed 0, 25 epochs:

| Model | Precision | Recall | mAP50 | mAP50-95 | Parameters | GFLOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8s baseline | 0.7644 | 0.7712 | 0.8931 | 0.7695 | 11,166,560 | 28.82 |
| YOLOv8s-CBAM | 0.8084 | 0.7640 | 0.9084 | 0.7800 | 11,199,426 | 28.87 |
| YOLOv8n lightweight | 0.6637 | 0.7827 | 0.8718 | 0.7417 | 3,157,200 | 8.86 |

CBAM is the strongest accuracy candidate in this single-seed screening. The lightweight model is the strongest deployment-efficiency candidate because it has much lower CPU latency and a much smaller checkpoint. Final multi-seed training and internal test evaluation are still required before any production promotion.

## Limitations

- Generalisasi lintas signer belum dapat dibuktikan tanpa split berbasis signer/session.
- The current split is duplicate-cluster-aware, but not signer-independent.
- Several validation classes have only one sample after leakage-aware splitting, making those per-class estimates noisy.
- Kelas dengan bentuk tangan mirip memerlukan per-class analysis pada test set.
- Lighting, blur, occlusion, dan background kompleks dapat menurunkan confidence.
