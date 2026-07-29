# Model Selection Freeze

- Freeze timestamp UTC: 2026-07-29T08:32:00Z
- Git SHA at selection freeze: `9bdd43a465d69a29bc64317974c21ba1478713e8`
- Selection was made before opening the internal test set.
- Internal test has already been opened once for the selected checkpoint.
- The internal test result must not be used for additional tuning, threshold changes, architecture changes, or model reselection.

## Selected Accuracy Candidate

- Identifier: `final-cbam-seed42`
- Relative checkpoint: `artifacts/experiments-screening-v2/final/runs/final-cbam-seed42/weights/best.pt`
- Checkpoint SHA-256: `B22B581114941B8FB22C9A2BC1492C02786FF8F249D26F516F28717DCC032A7C`
- Model: YOLOv8s-CBAM
- Seed: 42
- Best epoch: 85
- Validation precision: 0.84806
- Validation recall: 0.86545
- Validation mAP50: 0.95750
- Validation mAP50-95: 0.84158
- Parameters: 11,199,426
- GFLOPs: 28.869352

## Efficient Candidate Kept For Deployment Trade-Offs

- Identifier: `final-light-seed1337`
- Relative checkpoint: `artifacts/experiments-screening-v2/final/runs/final-light-seed1337/weights/best.pt`
- Checkpoint SHA-256: `A11EA35ECDC2F15496041FC2122719AD663901BC56DFA51C81A9A16BD0F3F36D`
- Model: YOLOv8n lightweight
- Seed: 1337
- Best epoch: 104
- Validation precision: 0.86864
- Validation recall: 0.85397
- Validation mAP50: 0.94892
- Validation mAP50-95: 0.84053
- Parameters: 3,157,200
- GFLOPs: 8.8575488

## Rationale

CBAM is frozen as the production-candidate model because it had the highest mean validation mAP50-95 across the final seeds and the highest individual validation mAP50-95 checkpoint. The lightweight model remains documented as the efficient deployment candidate because it is materially smaller and faster on CPU.
