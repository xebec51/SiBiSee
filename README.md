# SiBiSee

SiBiSee adalah aplikasi Streamlit untuk pengenalan gestur SIBI terisolasi secara real-time menggunakan model YOLO. Aplikasi mendukung live camera WebRTC, upload/camera snapshot, validasi gambar, temporal stabilization, dan transcript builder.

Status saat ini: codebase sudah dimodularisasi dan dites, model produksi masih artifact terenkripsi `models/best.pt.enc`. Training ulang dan klaim peningkatan akurasi belum dijalankan ulang karena dataset lokal tidak ditemukan di workspace ini.

## Kelas

Gambar panduan saat ini memuat 49 kelas:

- Alfabet: A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
- Angka: Satu, Dua, Tiga, Empat, Lima, Enam, Tujuh, Delapan, Sembilan
- Kata: Bodoh, Cinta, Jahat, Kamu, Kasih, Maaf, Makan, Masuk, Minum, Nama, Rumah, Saya, Terima, Tidur, Tolong

Gestur dinamis seperti J dan Z perlu evaluasi temporal terpisah; model single-frame tidak boleh diklaim menangani tata bahasa SIBI lengkap.

## Instalasi

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
```

`requirements-app.txt` dan `requirements.txt` memakai PyTorch CPU wheel untuk deployment. Untuk PC lokal dengan RTX 4060 Ti dan CUDA 12.1 runtime PyTorch, gunakan profile GPU:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

Isi `.streamlit/secrets.toml` dengan key model dan credential Twilio bila dipakai. Jangan commit file secrets.

## Menjalankan Aplikasi

```powershell
.\.venv\Scripts\streamlit.exe run src\app.py
```

Twilio bersifat opsional; jika tidak tersedia, aplikasi memakai fallback STUN publik. Key model wajib untuk memuat `models/best.pt.enc`.

## Development

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip_audit
```

Secret scan:

```powershell
$files = git ls-files | Where-Object { $_ -notmatch '^(assets/|models/best\.pt\.enc|docs/Laporan_Akhir_SiBiSee\.pdf)' }
.\.venv\Scripts\detect-secrets.exe scan --no-verify @files
```

## Dataset

Dataset yang disebut proyek lama berasal dari Roboflow Universe `sibi-detection-nftzq/sibi-bieme` version 2. Dataset tidak disimpan di Git.

Audit dataset:

```powershell
$env:ROBOFLOW_API_KEY = Read-Host "Roboflow API key"
.\.venv\Scripts\python.exe scripts\download_dataset.py --output-dir D:\Datasets\SiBiSee
Remove-Item Env:\ROBOFLOW_API_KEY
$env:SIBISEE_DATASET_DIR = "D:\path\to\dataset"
.\.venv\Scripts\python.exe scripts\build_manifest.py
.\.venv\Scripts\python.exe scripts\create_splits.py
```

Output audit disimpan di `artifacts/dataset/`.

## Training dan Evaluasi

Training reproducible memakai file YAML di `configs/training/`.

```powershell
$env:SIBISEE_DATASET_YAML = "D:\path\to\dataset\data.yaml"
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
.\.venv\Scripts\python.exe scripts\train.py --config configs\training\baseline.yaml
.\.venv\Scripts\python.exe scripts\evaluate.py --model artifacts\runs\baseline-yolov8s-seed0\weights\best.pt --data $env:SIBISEE_DATASET_YAML
.\.venv\Scripts\python.exe scripts\benchmark.py --model artifacts\runs\baseline-yolov8s-seed0\weights\best.pt
```

Model baru hanya boleh dipromosikan setelah test set dan benchmark deployment memenuhi rule di `docs/BENCHMARK.md`.

## Struktur

```text
src/
  app.py
  sibisee/
    config.py
    logging_config.py
    domain/
    inference/
    services/
    ui/
scripts/
configs/training/
tests/
docs/
```

## Security

Notebook lama pernah memuat Roboflow API key plaintext. Notebook aktif sudah disanitasi, tetapi histori publik masih perlu dibersihkan dengan `git-filter-repo` setelah pemilik repository memastikan key lama sudah direvoke. Detail ada di `SECURITY.md`.

## License

MIT License. Lihat `LICENSE`.
