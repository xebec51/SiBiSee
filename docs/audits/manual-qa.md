# Manual QA Notes

## Live Recognition Refresh

Implementation status:

- WebRTC processing thread no longer calls Streamlit APIs directly.
- Live status rendering uses `st.fragment(run_every="1s")` when supported by the installed Streamlit version.
- Recognition processor exposes a lock-protected snapshot for token, confidence, and latency.
- Each processed frame performs exactly one model inference; annotation is plotted from the same raw result.

Manual verification status:

- Full live camera verification was not completed in this session because the production model encryption key was not available locally.
- `scripts/check_model_compatibility.py` stopped with `SIBISEE_MODEL_ENCRYPTION_KEY belum dikonfigurasi; compatibility smoke tidak dijalankan.`

Required manual QA after configuring secrets:

1. Run `streamlit run src/app.py`.
2. Start live camera mode.
3. Confirm transcript, confidence, and latency update without pressing another widget.
4. Confirm CPU usage stays stable and no busy loop is created.
5. Confirm transcript undo, clear, and download still work during and after live inference.
