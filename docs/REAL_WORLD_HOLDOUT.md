# Real-World Holdout Protocol

Status: holdout belum tersedia di repository ini, sehingga generalisasi dunia nyata belum boleh diklaim.

Minimum collection criteria:

- Several different people using pseudonymous participant IDs.
- Several backgrounds.
- Lighting variation: bright, normal indoor, low light, backlit if possible.
- Camera distance variation: near, desk, far.
- Device variation: laptop webcam, phone camera, external webcam where available.
- Include visually difficult classes and classes that are often confused.
- Do not use frames copied from the training, validation, or test dataset.
- Record only metadata needed for evaluation; do not store names, email, phone numbers, addresses, or other unnecessary personal data.

Example capture:

```powershell
.\.venv\Scripts\python.exe scripts\collect_holdout.py `
  --output-dir D:\Datasets\SiBiSee-Holdout `
  --participant-id P001 `
  --session-id S001 `
  --device-label laptop-webcam `
  --background plain-wall `
  --lighting indoor-normal `
  --distance desk `
  --class-name A `
  --class-name B `
  --class-name Saya
```

During capture:

- Press `SPACE` to save a frame.
- Press `n` to move to the next class.
- Press `q` to end the session.

Dry-run smoke:

```powershell
.\.venv\Scripts\python.exe scripts\collect_holdout.py `
  --output-dir artifacts\private\holdout-smoke `
  --participant-id P000 `
  --session-id S000 `
  --device-label dry-run `
  --background dry-run `
  --lighting dry-run `
  --distance dry-run `
  --class-name A `
  --dry-run
```

Evaluation rule:

- Use the real-world holdout only after internal model selection is frozen.
- Report aggregate and per-class metrics separately from internal test metrics.
- Do not tune hyperparameters after viewing holdout results unless a new locked holdout is created.
