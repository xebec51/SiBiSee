# Data Card

## Source

Dataset was downloaded from Roboflow Universe:

- Workspace: `sibi-detection-nftzq`
- Project: `sibi-bieme`
- Version: 2
- Format: YOLOv8
- Downloaded images: 5831
- Downloaded label files: 5831
- Download manifest SHA-256: `4e240d4d1673bea6c4ff03594a991dd2bceb177f9dc4d03a7c176b8866df55ad`

Dataset penuh tidak disimpan di Git.

## Current Audit Status

The local audit found:

- Images: 5831
- Annotations: 5832
- Classes in `data.yaml`: 49
- Observed classes: 49
- Corrupt images: 0
- Missing labels: 0
- Empty labels: 0
- Unknown split images: 0
- Orphan labels: 0
- Exact/near duplicate groups: 365
- Cross-split leakage groups in the original Roboflow split: 138
- Class imbalance ratio: 1.0084

Script audit tersedia di:

- `scripts/download_dataset.py`
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

Signer/person metadata was not available in the downloaded export. The generated split therefore uses duplicate clusters as the grouping key. Subject-independent evaluation cannot be claimed from this split.

Generated leakage-aware split with seed 42:

| Split | Images | Class coverage |
| --- | ---: | ---: |
| train | 3940 | 49/49 |
| val | 1047 | 49/49 |
| test | 844 | 49/49 |
