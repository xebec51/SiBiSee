# Dynamic Gestures

Gestur seperti J dan Z harus diperlakukan sebagai temporal task.

Pipeline yang disarankan:

```text
hand crop/detection
-> landmark extraction
-> normalized landmark sequence
-> temporal classifier
```

Baseline penelitian:

- Landmark trajectory rules.
- TCN.
- LSTM/GRU.
- Small Transformer encoder.

Dataset sequence perlu menyimpan signer/session metadata agar split subject-independent dapat dibuat. Jangan mengintegrasikan model temporal dummy ke produksi.
