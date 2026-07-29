# Manual QA Notes

## Live Recognition Refresh

Implementation status:

- WebRTC processing thread no longer calls Streamlit APIs directly.
- Live status rendering uses `st.fragment(run_every="1s")` when supported by the installed Streamlit version.
- Recognition processor exposes a lock-protected snapshot for token, confidence, and latency.
- Each processed frame performs exactly one model inference; annotation is plotted from the same raw result.

Manual verification status for release packaging:

| Check | Status | Evidence |
| --- | --- | --- |
| Encrypted artifact compatibility | PASS | `scripts/check_model_compatibility.py` loaded `models/best.pt.enc`, verified checksum/decrypt/load, 49 classes, one inference, and `temporary_model_dirs_remaining: 0`. |
| Streamlit startup | PASS | Headless local server on `127.0.0.1:8510` returned HTTP 200 with `SIBISEE_MODEL_ENCRYPTION_KEY` configured. |
| Static image workflow | PASS | Regression tests cover static primary detection without temporal window and duplicate transcript prevention on rerun. |
| Invalid file rejection | PASS | Regression tests cover upload validation. |
| One inference per static image/live processed frame | PASS | Regression tests cover detector/model call count and live processor single predict per processed frame. |
| Transcript undo/clear/download | PASS | Regression tests cover transcript behavior. |
| Temporal settings reset | PASS | Regression test confirms `TemporalDecoder` is recreated when settings fingerprint changes. |
| Twilio fallback | PASS | Regression tests cover fallback STUN behavior. |
| Live camera browser interaction | NOT RUN | No physical browser/camera interaction was performed in this headless smoke. WebRTC code path remains covered by processor tests, not a live-device manual test. |
