from __future__ import annotations

import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}


class ImageValidationError(ValueError):
    """Raised when an uploaded image is unsafe or invalid."""


@dataclass(frozen=True)
class ImageValidationSettings:
    max_upload_mb: int = 8
    max_image_pixels: int = 12_000_000
    max_side: int = 1280


def validate_image_bytes(
    data: bytes,
    mime_type: str | None,
    settings: ImageValidationSettings | None = None,
) -> Image.Image:
    settings = settings or ImageValidationSettings()
    if not data:
        raise ImageValidationError("File gambar kosong.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise ImageValidationError(f"Ukuran gambar melebihi batas {settings.max_upload_mb} MB.")
    if mime_type and mime_type not in ALLOWED_MIME_TYPES:
        raise ImageValidationError("Format gambar harus JPEG atau PNG.")

    original_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image)
            if image.width * image.height > settings.max_image_pixels:
                raise ImageValidationError("Resolusi gambar terlalu besar.")
            image = image.convert("RGB")
            image.thumbnail((settings.max_side, settings.max_side), Image.Resampling.LANCZOS)
            return image.copy()
    except (Image.DecompressionBombError, ImageValidationError):
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError("File gambar tidak valid atau rusak.") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = original_limit


def preprocess_for_inference(image: Image.Image, image_size: int) -> Image.Image:
    prepared = image.convert("RGB").copy()
    prepared.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
    return prepared
