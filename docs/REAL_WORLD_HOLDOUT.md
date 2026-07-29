# Real-World Holdout Protocol

Status: NOT RUN - intentionally excluded from the current release scope.

Real-world holdout is not release evidence for v2.0.0. The current release is based on the dataset audit,
leakage-aware split, final multi-seed validation, one internal-test evaluation, PyTorch benchmark, production
packaging, app smoke tests, deployment smoke tests, QA, CI, and secret scanning.

The model has not been validated on new real-world participants, cameras, lighting, backgrounds, distances, or
devices. Internal validation and internal-test metrics do not guarantee real-world performance.

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

Validation gate:

```powershell
.\.venv\Scripts\python.exe scripts\validate_holdout.py `
  --holdout-dir D:\Datasets\SiBiSee-Holdout `
  --dataset-dir D:\Datasets\SiBiSee `
  --data artifacts\dataset\sibisee_splits.yaml `
  --output-dir artifacts\private\holdout-validation
```

The validator writes private, ignored artifacts:

- `summary.json`
- `class-coverage.csv`
- `session-coverage.csv`
- `duplicate-groups.csv`
- `dataset-overlap.csv`
- `image-errors.csv`
- `sensitive-metadata.csv`

The gate blocks evaluation when metadata is missing, paths are unsafe, images are missing/corrupt,
class labels are unknown, private contact-like metadata is detected, exact/near duplicates are present,
or any exact image overlaps the training/validation/test dataset. It also checks that collection covers
multiple participants, sessions, devices, backgrounds, lighting conditions, distances, and every class.

Evaluation after the gate passes:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_holdout.py `
  --model artifacts\experiments-screening-v2\final\runs\final-cbam-seed42\weights\best.pt `
  --holdout-dir D:\Datasets\SiBiSee-Holdout `
  --data artifacts\dataset\sibisee_splits.yaml `
  --output-dir artifacts\private\holdout-evaluation\cbam-seed42 `
  --device 0
```

`scripts/evaluate_holdout.py` uses the production inference threshold and primary-detection strategy by
default: confidence threshold `0.4`, IoU threshold `0.7`, max detections `5`, image size `640`, and primary
selection by confidence. It writes aggregate metrics, per-class precision/recall/F1, confusion pairs,
group metrics by capture condition, private raw predictions, and latency. The evaluation is image-level:
each holdout image contributes one expected class and one primary predicted class or `<no_detection>`.

Evaluation rule:

- Use the real-world holdout only after internal model selection is frozen.
- Report aggregate and per-class metrics separately from internal test metrics.
- Do not tune hyperparameters after viewing holdout results unless a new locked holdout is created.
- Do not package, encrypt, deploy, or tag a new production model until the holdout gate is documented.
- For v2.0.0, the documented holdout status is `NOT RUN - intentionally excluded from the current release scope`.
