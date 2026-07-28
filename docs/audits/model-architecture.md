# Model Architecture Audit

Command:

```powershell
.\.venv\Scripts\python.exe scripts\inspect_architecture.py --output artifacts\models\architecture_smoke.json
```

Result:

| model | config | parameters | GFLOPs @640 | Detect input channels |
| --- | --- | ---: | ---: | --- |
| Baseline YOLOv8s | `configs/models/yolov8s.yaml` | 11,166,560 | 28.8168 | `[128, 256, 512]` |
| YOLOv8s + CBAM | `configs/models/yolov8s-cbam.yaml` | 11,199,426 | 28.8694 | `[128, 256, 512]` |

CBAM insertion:

- One dimension-preserving `CBAM` block is inserted immediately after the final P5 head feature and immediately before `Detect`.
- The `Detect` head uses the CBAM output for P5.
- The `Detect` head receives the same channel dimensions as baseline: `[128, 256, 512]`.
- Parameter delta is `32,866`, primarily from the added CBAM channel/spatial attention module.

Fair pretrained initialization smoke:

| model | pretrained source | matched items | target items | note |
| --- | --- | ---: | ---: | --- |
| Baseline YOLOv8s | `yolov8s.pt` | 355 | 355 | direct name/shape match |
| YOLOv8s + CBAM | `yolov8s.pt` | 355 | 358 | 270 direct matches, 85 Detect head remaps, 3 CBAM items random |

Smoke verification:

- `YOLO(configs/models/yolov8s.yaml).info()` succeeded.
- `YOLO(configs/models/yolov8s-cbam.yaml).info()` succeeded after registering local CBAM.
- Baseline and CBAM models both completed a dummy forward pass.
- No training accuracy, screening result, or benchmark result is claimed by this audit.
