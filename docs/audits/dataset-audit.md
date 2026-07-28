# Dataset Audit

Dataset audit could not be completed in this workspace.

Commands attempted:

```powershell
Get-ChildItem -Force -Directory .. | Where-Object { $_.Name -match 'SIBI|sibi|dataset|data|bieme' }
if ($env:SIBISEE_DATASET_DIR) { Write-Output $env:SIBISEE_DATASET_DIR } else { Write-Output 'SIBISEE_DATASET_DIR not set' }
.\.venv\Scripts\python.exe scripts\build_manifest.py
```

Result:

- No sibling dataset directory found.
- `SIBISEE_DATASET_DIR` was not set.
- `scripts/build_manifest.py` stopped with: `Dataset tidak ditemukan. Set SIBISEE_DATASET_DIR atau berikan path eksplisit.`

No manifest, class distribution, duplicate report, split manifest, training, or model comparison metrics were generated in this session.
