# Data Card

## Source

Dokumentasi lama menyebut dataset Roboflow Universe `sibi-detection-nftzq/sibi-bieme` version 2. Dataset penuh tidak disimpan di Git.

## Current Audit Status

Dataset lokal tidak ditemukan di workspace saat audit. Script audit tersedia di:

- `scripts/build_manifest.py`
- `scripts/audit_dataset.py`
- `scripts/create_splits.py`

## Split Policy

Prioritas split:

1. Group split berdasarkan signer/person.
2. Group split berdasarkan source video/session.
3. Group split berdasarkan duplicate cluster.
4. Random stratified split hanya sebagai fallback.

Jika metadata signer tidak tersedia, evaluasi tidak boleh diklaim subject-independent.
