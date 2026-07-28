# Benchmark

Benchmark deployment dijalankan dengan warm-up, batch size 1, input size produksi, dan banyak iterasi.

```powershell
.\.venv\Scripts\python.exe scripts\benchmark.py --model path\to\best.pt --iterations 100 --warmup 10
```

Model baru hanya boleh dipromosikan bila memenuhi salah satu:

1. Mean test mAP50-95 naik sekitar 0.3 percentage point tanpa regresi per-class recall yang tidak dapat diterima dan latency tidak memburuk material.
2. Mean test mAP50-95 berada dalam sekitar 0.2 percentage point dari model terbaik, tetapi CPU latency atau ukuran artifact membaik minimal sekitar 20%.
3. Real-world holdout membaik jelas dan trade-off terdokumentasi.

Belum ada benchmark v2 final karena dataset dan key model production tidak tersedia untuk evaluasi lengkap di sesi ini.
