from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def run_command(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()


def git_commit() -> str:
    try:
        return run_command(["git", "rev-parse", "HEAD"])
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def environment_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
    }
    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["torch_cuda"] = torch.version.cuda
        snapshot["cuda_available"] = torch.cuda.is_available()
        snapshot["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        snapshot["torch_error"] = exc.__class__.__name__
    try:
        import ultralytics

        snapshot["ultralytics"] = ultralytics.__version__
    except Exception as exc:
        snapshot["ultralytics_error"] = exc.__class__.__name__
    return snapshot
