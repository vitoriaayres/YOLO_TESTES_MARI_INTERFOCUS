import io

import pytest
from PIL import Image

from app.core.exceptions import ApiError
from app.services.image_codec import decode_upload, encode_jpeg_base64
from tests.fakes import jpeg_bytes


def test_decode_upload_should_return_bgr_image_with_original_size() -> None:
    image = decode_upload(jpeg_bytes(64, 48), max_side=2000)
    assert image.shape == (48, 64, 3)


def test_decode_upload_should_shrink_longest_side_to_max_side() -> None:
    image = decode_upload(jpeg_bytes(400, 200), max_side=100)
    assert image.shape[:2] == (50, 100)


def test_decode_upload_should_apply_exif_rotation() -> None:
    buffer = io.BytesIO()
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: girar 90° (foto de celular em pé)
    Image.new("RGB", (64, 48), "white").save(buffer, "JPEG", exif=exif)
    image = decode_upload(buffer.getvalue(), max_side=2000)
    assert image.shape[:2] == (64, 48)


def test_decode_upload_should_reject_empty_file() -> None:
    with pytest.raises(ApiError) as error:
        decode_upload(b"", max_side=2000)
    assert (error.value.status, error.value.code) == (400, "empty_image")


def test_decode_upload_should_reject_non_image_file() -> None:
    with pytest.raises(ApiError) as error:
        decode_upload(b"isto nao e uma imagem", max_side=2000)
    assert (error.value.status, error.value.code) == (400, "invalid_image")


def test_encode_jpeg_base64_should_return_none_without_image() -> None:
    assert encode_jpeg_base64(None) is None
