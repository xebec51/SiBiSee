# Architecture

SiBiSee v2 memisahkan UI, domain logic, inference, dan services.

```text
src/app.py
src/sibisee/config.py
src/sibisee/domain/detection.py
src/sibisee/domain/transcript.py
src/sibisee/inference/model_loader.py
src/sibisee/inference/detector.py
src/sibisee/inference/preprocessing.py
src/sibisee/inference/temporal_decoder.py
src/sibisee/services/ice_servers.py
src/sibisee/services/gesture_guide.py
src/sibisee/ui/
```

Model dimuat sekali melalui Streamlit cache. Inference model dilindungi lock agar callback WebRTC tidak mengubah state model secara bersamaan.
