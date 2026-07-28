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

Metrik test yang direproduksi ulang belum tersedia di workspace ini karena dataset lokal tidak ditemukan. Angka dari notebook lama tidak dipromosikan sebagai hasil final v2.

## Limitations

- Generalisasi lintas signer belum dapat dibuktikan tanpa split berbasis signer/session.
- Kelas dengan bentuk tangan mirip memerlukan per-class analysis pada test set.
- Lighting, blur, occlusion, dan background kompleks dapat menurunkan confidence.
