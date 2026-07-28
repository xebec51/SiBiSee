from __future__ import annotations

import io

import pytest
from PIL import Image

from sibisee.inference.preprocessing import ImageValidationError, ImageValidationSettings, validate_image_bytes


def make_png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_validate_image_bytes_returns_rgb_image() -> None:
    image = validate_image_bytes(make_png(), "image/png")

    assert image.mode == "RGB"
    assert image.size == (32, 24)


def test_validate_image_rejects_wrong_mime() -> None:
    with pytest.raises(ImageValidationError):
        validate_image_bytes(make_png(), "application/octet-stream")


def test_validate_image_rejects_corrupt_payload() -> None:
    with pytest.raises(ImageValidationError):
        validate_image_bytes(b"not an image", "image/png")


def test_validate_image_rejects_large_payload() -> None:
    with pytest.raises(ImageValidationError):
        validate_image_bytes(b"x" * 1024, "image/png", ImageValidationSettings(max_upload_mb=0, max_image_pixels=100))
