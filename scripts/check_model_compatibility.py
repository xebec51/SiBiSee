from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

from PIL import Image


def check_model_compatibility(model_path: Path | None = None) -> None:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from sibisee.config import load_settings
    from sibisee.inference.model_loader import ModelLoadError, load_encrypted_model

    settings = load_settings()
    encrypted_path = model_path or settings.security.encrypted_model_path
    key = os.getenv("SIBISEE_MODEL_ENCRYPTION_KEY")
    if not key:
        raise SystemExit("SIBISEE_MODEL_ENCRYPTION_KEY belum dikonfigurasi; compatibility smoke tidak dijalankan.")

    temp_root = Path(tempfile.gettempdir())
    before = set(temp_root.glob("sibisee-model-*"))
    try:
        model = load_encrypted_model(encrypted_path, key, expected_sha256=settings.security.model_sha256)
    except ModelLoadError as exc:
        raise SystemExit(str(exc)) from exc

    image = Image.new("RGB", (settings.inference.image_size, settings.inference.image_size), "white")
    results = model.predict(image, conf=0.25, verbose=False)
    names = getattr(model, "names", {})
    after = set(temp_root.glob("sibisee-model-*"))
    leaked_temp_dirs = sorted(path for path in after - before if path.exists())

    print(f"model_path: {encrypted_path}")
    print(f"class_count: {len(names)}")
    print(f"class_names_sample: {list(names.values())[:10] if isinstance(names, dict) else names[:10]}")
    print(f"result_count: {len(results)}")
    print(f"temporary_model_dirs_remaining: {len(leaked_temp_dirs)}")
    if leaked_temp_dirs:
        raise SystemExit("Temporary decrypted model directory cleanup failed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test encrypted legacy YOLO model compatibility.")
    parser.add_argument("--model", type=Path, help="Encrypted model path. Defaults to configured production artifact.")
    args = parser.parse_args()
    check_model_compatibility(args.model)


if __name__ == "__main__":
    main()
