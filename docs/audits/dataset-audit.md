# Dataset Audit

- Dataset source: Roboflow `sibi-detection-nftzq/sibi-bieme`, version 2, YOLOv8 export
- Dataset root in commands: `<local-dataset-root>`
- Dataset source metadata: `data.yaml`
- Images: 5831
- Annotations: 5832
- Classes in data.yaml: 49
- Corrupt images: 0
- Missing labels: 0
- Empty labels: 0
- Orphan labels: 0
- Unknown split images: 0
- Exact/near duplicate groups: 365
- Cross-split leakage groups: 138
- Class imbalance ratio: 1.0084
- Manifest SHA-256 from download: `4e240d4d1673bea6c4ff03594a991dd2bceb177f9dc4d03a7c176b8866df55ad`

The original Roboflow split contains cross-split duplicate or near-duplicate leakage. A new leakage-aware split was generated with seed 42 using duplicate clusters as the grouping key because signer/person metadata was not available.

New split counts:

| Split | Images | Class coverage |
| --- | ---: | ---: |
| train | 3940 | 49/49 |
| val | 1047 | 49/49 |
| test | 844 | 49/49 |

Subject-independent evaluation cannot be proven because signer/person metadata is not present in the downloaded export. Some validation and test classes have very low support after preserving duplicate clusters, so per-class metrics for those classes have high variance.

Generated artifacts:

- `artifacts/dataset/manifest.csv`
- `artifacts/dataset/class_distribution.csv`
- `artifacts/dataset/split_distribution.csv`
- `artifacts/dataset/duplicate_groups.csv`
- `artifacts/dataset/cross_split_leakage.csv`
- `artifacts/dataset/bbox_statistics.csv`
- `artifacts/dataset/summary.json`
- `artifacts/dataset/split_manifest.csv`
- `artifacts/dataset/train.txt`
- `artifacts/dataset/val.txt`
- `artifacts/dataset/test.txt`
- `artifacts/dataset/sibisee_splits.yaml`
- `artifacts/dataset/split_summary.json`

These generated artifacts are local evidence files and are not committed when they contain local dataset paths.
