# SiBiSee

SiBiSee adalah aplikasi Streamlit untuk pengenalan gestur SIBI terisolasi secara real-time menggunakan model YOLO. Aplikasi mendukung live camera WebRTC, upload/camera snapshot, validasi gambar, temporal stabilization, dan transcript builder.

Status release: model produksi yang dipilih adalah YOLOv8s-CBAM seed 42, dikemas sebagai artifact terenkripsi `models/best.pt.enc` dengan backend PyTorch. Release ini memakai evidence dataset audit, leakage-aware split, final multi-seed validation, satu kali internal-test evaluation, benchmark PyTorch, packaging, smoke test, QA, CI, dan secret scan.

Real-world holdout: NOT RUN - intentionally excluded from the current release scope. Model belum tervalidasi untuk participant, kamera, pencahayaan, background, jarak, atau device baru. Jangan mengklaim generalisasi dunia nyata dari hasil internal test.

## Kelas

Gambar panduan saat ini memuat 49 kelas:

- Alfabet: A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
- Angka: Satu, Dua, Tiga, Empat, Lima, Enam, Tujuh, Delapan, Sembilan
- Kata: Bodoh, Cinta, Jahat, Kamu, Kasih, Maaf, Makan, Masuk, Minum, Nama, Rumah, Saya, Terima, Tidur, Tolong

Gestur dinamis seperti J dan Z perlu evaluasi temporal terpisah; model single-frame tidak boleh diklaim menangani tata bahasa SIBI lengkap. SiBiSee adalah aplikasi portfolio/research untuk isolated SIBI sign recognition, bukan penerjemah bahasa isyarat lengkap dan bukan sistem keselamatan atau aksesibilitas kritis.

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
Runtime memverifikasi `models/best.pt.enc` memakai checksum di `models/best.metadata.json`.

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
.\.venv\Scripts\python.exe scripts\run_experiments.py --stage screening --models baseline cbam light --seeds 0
.\.venv\Scripts\python.exe scripts\evaluate.py --model artifacts\experiments\screening\runs\screening-baseline-seed0\weights\best.pt --data $env:SIBISEE_DATASET_YAML
.\.venv\Scripts\python.exe scripts\benchmark.py --model artifacts\experiments\screening\runs\screening-baseline-seed0\weights\best.pt
```

Model v2.0.0 dipromosikan berdasarkan model-selection freeze dan internal-test evaluation yang sudah dibuka sekali. Jangan menjalankan tuning tambahan berdasarkan internal test.

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

Notebook lama pernah memuat Roboflow API key plaintext. Key lama telah direvoke dan history repository sudah dibersihkan pada upgrade ini. Detail ada di `SECURITY.md`.

Model production metadata:

- Artifact: `models/best.pt.enc`
- Metadata: `models/best.metadata.json`
- Backend: PyTorch
- ONNX status: NOT RUN - not required by the selected PyTorch deployment backend
- Holdout status: NOT RUN - intentionally excluded from the current release scope

## License

MIT License. Lihat `LICENSE`.
