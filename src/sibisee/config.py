from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def _env_list(name: str) -> set[str] | None:
    value = os.getenv(name)
    if not value:
        return None
    items = {item.strip() for item in value.split(",") if item.strip()}
    return items or None


@dataclass(frozen=True)
class InferenceSettings:
    image_size: int = 640
    infer_every_n_frames: int = 3
    confidence_threshold: float = 0.4
    iou_threshold: float = 0.7
    max_detections: int = 5
    primary_detection_strategy: str = "confidence"
    class_allowlist: set[str] | None = None


@dataclass(frozen=True)
class TemporalSettings:
    window_size: int = 7
    min_matching_frames: int = 4
    min_average_confidence: float = 0.55
    confirmation_cooldown_seconds: float = 1.0


@dataclass(frozen=True)
class SecuritySettings:
    encrypted_model_path: Path = PROJECT_ROOT / "models" / "best.pt.enc"
    model_sha256: str | None = "9f58c1af732e6817efb3776842667d72d98fb37c9a336a049fbd1d5b19da8661"
    max_upload_mb: int = 8
    max_image_pixels: int = 12_000_000


@dataclass(frozen=True)
class AppSettings:
    project_root: Path = PROJECT_ROOT
    assets_dir: Path = PROJECT_ROOT / "assets"
    guide_dir: Path = PROJECT_ROOT / "assets" / "guide"
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    temporal: TemporalSettings = field(default_factory=TemporalSettings)
    security: SecuritySettings = field(default_factory=SecuritySettings)


def load_settings() -> AppSettings:
    inference = InferenceSettings(
        image_size=_env_int("SIBISEE_IMAGE_SIZE", 640),
        infer_every_n_frames=max(1, _env_int("SIBISEE_INFER_EVERY_N_FRAMES", 3)),
        confidence_threshold=_env_float("SIBISEE_CONFIDENCE_THRESHOLD", 0.4),
        iou_threshold=_env_float("SIBISEE_IOU_THRESHOLD", 0.7),
        max_detections=max(1, _env_int("SIBISEE_MAX_DETECTIONS", 5)),
        primary_detection_strategy=os.getenv("SIBISEE_PRIMARY_DETECTION_STRATEGY", "confidence"),
        class_allowlist=_env_list("SIBISEE_CLASS_ALLOWLIST"),
    )
    temporal = TemporalSettings(
        window_size=max(1, _env_int("SIBISEE_TEMPORAL_WINDOW_SIZE", 7)),
        min_matching_frames=max(1, _env_int("SIBISEE_MIN_MATCHING_FRAMES", 4)),
        min_average_confidence=_env_float("SIBISEE_MIN_AVERAGE_CONFIDENCE", 0.55),
        confirmation_cooldown_seconds=_env_float("SIBISEE_CONFIRMATION_COOLDOWN_SECONDS", 1.0),
    )
    security = SecuritySettings(
        encrypted_model_path=Path(os.getenv("SIBISEE_MODEL_PATH", str(PROJECT_ROOT / "models" / "best.pt.enc"))),
        model_sha256=os.getenv(
            "SIBISEE_MODEL_SHA256",
            "9f58c1af732e6817efb3776842667d72d98fb37c9a336a049fbd1d5b19da8661",
        ),
        max_upload_mb=max(1, _env_int("SIBISEE_MAX_UPLOAD_MB", 8)),
        max_image_pixels=max(1, _env_int("SIBISEE_MAX_IMAGE_PIXELS", 12_000_000)),
    )
    return AppSettings(inference=inference, temporal=temporal, security=security)
