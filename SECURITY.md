# Security

## Credential Incident

Audit menemukan Roboflow API key plaintext pada `notebooks/training_experiment.ipynb` di histori Git. Working tree saat ini sudah mengganti notebook dengan versi bersih yang membaca `ROBOFLOW_API_KEY` dari environment variable.

Tindakan manual yang wajib dilakukan pemilik repository:

1. Revoke Roboflow API key lama di dashboard Roboflow.
2. Buat key baru jika masih diperlukan.
3. Simpan key baru hanya di environment variable atau secret manager.
4. Setelah revoke dikonfirmasi, lakukan history rewrite dengan backup bundle dan `git-filter-repo`, lalu push menggunakan `--force-with-lease`.

Backup lokal sudah dibuat di parent repository sebagai `SiBiSee-pre-upgrade.bundle`.

## Secret Handling

- Jangan commit `.env`, `.streamlit/secrets.toml`, API key, token, password, private key, atau encryption key.
- `.streamlit/secrets.toml.example` hanya berisi placeholder.
- Model terenkripsi melindungi artifact at rest, tetapi model tetap berada di memory runtime saat inference.
- Error teknis masuk logger; UI menampilkan pesan ringkas tanpa stack trace atau path internal.

## Recommended GitHub Settings

Aktifkan GitHub secret scanning dan push protection pada repository `xebec51/SiBiSee`.
