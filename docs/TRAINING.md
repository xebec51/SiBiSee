# Training

Training dijalankan melalui `scripts/train.py` dan konfigurasi YAML di `configs/training/`.

Sebelum training:

```powershell
@'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
'@ | .\.venv\Scripts\python.exe -
```

Jangan mulai training jika `torch.cuda.is_available()` bernilai `False`.

Dataset lokal harus ditentukan:

```powershell
$env:SIBISEE_DATASET_YAML = "D:\path\to\dataset\data.yaml"
```

For GPU training on the local RTX 4060 Ti PC:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
```

For CPU-only CI or smoke tests:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Baseline dan CBAM harus memakai split, seed, image size, optimizer, scheduler, augmentation, dan training budget yang identik. Test set tidak boleh digunakan untuk memilih hyperparameter.
