"""Conversão entre os bytes enviados pelo cliente e imagens OpenCV (BGR)."""
import base64
import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.exceptions import ApiError


def decode_upload(data: bytes, max_side: int) -> np.ndarray:
    """Lê a imagem em memória, aplica a rotação EXIF e reduz o lado maior a max_side."""
    if not data:
        raise ApiError(400, "empty_image", "O arquivo enviado está vazio.")
    try:
        pil_image = Image.open(io.BytesIO(data))
        pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = pil_image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise ApiError(400, "invalid_image", "Não foi possível ler a imagem. Envie JPEG ou PNG.")

    width, height = pil_image.size
    longest = max(width, height)
    if longest > max_side:
        scale = max_side / longest
        pil_image = pil_image.resize((round(width * scale), round(height * scale)), Image.LANCZOS)

    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def encode_jpeg_base64(image: np.ndarray | None, quality: int = 85) -> str | None:
    if image is None:
        return None
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")
