# Deployment

## Local

```powershell
.\.venv\Scripts\streamlit.exe run src\app.py
```

Install CPU deployment dependencies with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
```

Install GPU local dependencies only on a compatible NVIDIA machine:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

## Secrets

Tambahkan `.streamlit/secrets.toml` dari template:

```toml
[model_security]
ENCRYPTION_KEY = "replace-with-fernet-key"

[twilio]
ACCOUNT_SID = "replace-with-twilio-account-sid"
AUTH_TOKEN = "replace-with-twilio-auth-token"
```

Twilio opsional. Model key wajib jika artifact produksi tetap terenkripsi.
For local non-Streamlit smoke tests, set `SIBISEE_MODEL_ENCRYPTION_KEY` instead of passing the key as a CLI argument.

## Production Notes

- Jangan deploy `.env` atau secrets template yang sudah diisi.
- Jangan deploy dataset atau run training.
- Verifikasi checksum `models/best.pt.enc` sebelum startup. Runtime membaca checksum dari `models/best.metadata.json`; `SIBISEE_MODEL_SHA256` tetap tersedia sebagai override eksplisit.
- Gunakan HTTPS untuk WebRTC camera access.
- Backend produksi v2.0.0 adalah PyTorch. ONNX export/parity: NOT RUN - not required.

## Deployment Secret

Masukkan Fernet key produksi ke deployment secrets, bukan ke repository:

```toml
[model_security]
ENCRYPTION_KEY = "paste-key-from-clipboard"
```

Jangan mengirim key melalui chat, issue, commit, atau log. Setelah secret tersimpan dan deployment direstart,
hapus key lokal:

```powershell
Remove-Item Env:\SIBISEE_MODEL_ENCRYPTION_KEY -ErrorAction SilentlyContinue
Set-Clipboard -Value " " -ErrorAction SilentlyContinue
```

## Rollback

Rollback artifact membutuhkan pasangan artifact dan key yang cocok:

1. Kembalikan `models/best.pt.enc` dan `models/best.metadata.json` dari commit/tag release sebelumnya.
2. Kembalikan deployment secret `model_security.ENCRYPTION_KEY` ke Fernet key yang digunakan artifact sebelumnya.
3. Redeploy branch/tag rollback.
4. Jalankan `scripts/check_model_compatibility.py` dengan key rollback sebelum membuka traffic.

## Current Deployment Gate

Deployment smoke for v2.0.0 is BLOCKED until the Fernet key is added to the deployment secret and the deployed app is
restarted from branch `main`. Do not create the `v2.0.0` tag until `docs/audits/deployment-smoke.md` is updated with a
real public URL and smoke-test evidence.
