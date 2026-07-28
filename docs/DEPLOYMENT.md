# Deployment

## Local

```powershell
.\.venv\Scripts\streamlit.exe run src\app.py
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

## Production Notes

- Jangan deploy `.env` atau secrets template yang sudah diisi.
- Jangan deploy dataset atau run training.
- Verifikasi checksum `models/best.pt.enc` sebelum startup.
- Gunakan HTTPS untuk WebRTC camera access.
