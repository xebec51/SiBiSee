# Initial Environment Audit

- Date: 2026-07-28
- OS: Windows
- Repository: `https://github.com/xebec51/SiBiSee.git`
- Branch: `main`
- Initial commit SHA: `ab91bc6e9364dd9c1f237d488e53a4ed4ae04236`
- Backup bundle: `D:\NALDI\SiBiSee-pre-upgrade.bundle`
- Python: `Python 3.10.8`
- pip before venv: `pip 22.2.2`
- Git: `git version 2.50.1.windows.1`
- GPU: `NVIDIA GeForce RTX 4060 Ti`, 16380 MiB
- NVIDIA driver: `591.86`
- CUDA runtime visible to driver: `13.1`
- PyTorch in project venv: `2.5.1+cu121`
- PyTorch CUDA available: `True`
- PyTorch CUDA runtime: `12.1`
- CPU: `12th Gen Intel(R) Core(TM) i7-12700`, 12 cores, 20 logical processors
- RAM: 16 GB
- Approximate repository size before edits: 156,519,831 bytes

## Initial Status

- App status: single-file Streamlit app with model loading at import-time and no test suite.
- Model status: encrypted production artifact exists at `models/best.pt.enc`.
- Model artifact SHA-256: `9f58c1af732e6817efb3776842667d72d98fb37c9a336a049fbd1d5b19da8661`.
- Dependency status: original `requirements.txt` was unpinned.
- Security status: tracked notebook contained a Roboflow API key literal and old output/path logs.
