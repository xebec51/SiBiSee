# Contributing

Gunakan branch `main` hanya jika memang mengikuti instruksi maintainer repository ini. Untuk perubahan normal, gunakan branch terpisah dan PR.

Checklist lokal:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m pip_audit
```

Jangan commit dataset penuh, run training lengkap, credential, key model, atau artifact intermediate.
