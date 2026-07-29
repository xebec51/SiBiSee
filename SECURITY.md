# Security

## Credential Incident

Audit menemukan Roboflow API key plaintext pada versi lama `notebooks/training_experiment.ipynb`. Key lama telah direvoke oleh pemilik repository. History repository sudah dibersihkan dengan backup bundle dan notebook aktif hanya membaca `ROBOFLOW_API_KEY` dari environment variable.

Backup lokal pre-upgrade disimpan di `D:\NALDI\SiBiSee-pre-upgrade.bundle` dan tidak boleh diubah atau dihapus.

## Secret Handling

- Jangan commit `.env`, `.streamlit/secrets.toml`, API key, token, password, private key, atau encryption key.
- `.streamlit/secrets.toml.example` hanya berisi placeholder.
- Model terenkripsi melindungi artifact at rest, tetapi model tetap berada di memory runtime saat inference.
- Model encryption key harus berupa Fernet key khusus model, bukan Roboflow API key.
- Runtime memverifikasi checksum encrypted artifact dari `models/best.metadata.json` sebelum decrypt.
- Error teknis masuk logger; UI menampilkan pesan ringkas tanpa stack trace atau path internal.

## Recommended GitHub Settings

Aktifkan GitHub secret scanning dan push protection pada repository `xebec51/SiBiSee`.
