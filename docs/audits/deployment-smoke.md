# Deployment Smoke

Status: BLOCKED

- Timestamp UTC: 2026-07-29T10:23:00Z
- Repository: `xebec51/SiBiSee`
- Branch: `main`
- Production backend: PyTorch
- Artifact: `models/best.pt.enc`
- Metadata: `models/best.metadata.json`
- Encrypted artifact SHA-256: `04f9fb0f21e42dc01e3832f2e786aa92b281400035bb488d9ebbe5e7b146cd23`

## Completed Before Deployment

| Check | Status | Evidence |
| --- | --- | --- |
| Production package | PASS | `scripts/package_model.py` produced encrypted artifact and metadata. |
| Compatibility smoke | PASS | `scripts/check_model_compatibility.py` loaded artifact, verified checksum/decrypt/load, 49 classes, inference, and temporary cleanup. |
| Local Streamlit startup | PASS | Local headless server returned HTTP 200. |
| Clean CPU install | PASS | Fresh clone installed `requirements-app.txt` with CPU PyTorch. |
| Clean-clone compatibility | PASS | Fresh clone loaded encrypted artifact and ran inference. |
| Clean-clone Streamlit startup | PASS | Fresh clone headless server returned HTTP 200. |
| Gitleaks clean clone | PASS | 60 commits scanned, no leaks found. |

## Blocker

Deployment smoke was not run because no existing public deployment URL or connected deployment control was available in
the repository context, and the required Fernet key must be entered manually into deployment secrets.

Required manual action:

1. Paste the Fernet key currently in the local clipboard into the deployment secret:

```toml
[model_security]
ENCRYPTION_KEY = "<paste-key-from-clipboard>"
```

Do not send the key through chat, issue comments, logs, or commits.

After the secret is saved, redeploy/restart the app from branch `main`, then verify build, startup, encrypted artifact
load, static image inference, live camera, confidence/latency refresh, transcript, undo/clear/download, Twilio fallback,
and logs without secret leakage.
