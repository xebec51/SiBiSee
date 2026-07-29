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

No production artifact has been replaced.

Final validation, leakage-aware split, seeds 0/42/1337:

| Model | Precision | Recall | mAP50 | mAP50-95 | Parameters | GFLOPs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8s-CBAM mean | 0.8492 | 0.8788 | 0.9484 | 0.8389 | 11,199,426 | 28.87 |
| YOLOv8n lightweight mean | 0.8510 | 0.8525 | 0.9511 | 0.8330 | 3,157,200 | 8.86 |

Selected production candidate before test evaluation: YOLOv8s-CBAM seed 42. Internal test result for that checkpoint: precision 0.9400, recall 0.9295, mAP50 0.9659, mAP50-95 0.8472.

The lightweight model is the strongest deployment-efficiency candidate because it has much lower CPU latency and a much smaller checkpoint. Production promotion still requires model packaging/encryption, app smoke testing, and real-world holdout review.

## Limitations

- Generalisasi lintas signer belum dapat dibuktikan tanpa split berbasis signer/session.
- The current split is duplicate-cluster-aware, but not signer-independent.
- Several validation classes have only one sample after leakage-aware splitting, making those per-class estimates noisy.
- Kelas dengan bentuk tangan mirip memerlukan per-class analysis pada test set.
- Lighting, blur, occlusion, dan background kompleks dapat menurunkan confidence.
