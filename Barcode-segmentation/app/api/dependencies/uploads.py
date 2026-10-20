"""Validação do arquivo enviado no campo "image"."""
from dataclasses import dataclass

from fastapi import File, Request, UploadFile

from app.core.config import Settings
from app.core.exceptions import ApiError


@dataclass
class ImageUpload:
    data: bytes
    filename: str


async def read_image_upload(request: Request, image: UploadFile | None = File(None)) -> ImageUpload:
    if image is None:
        raise ApiError(400, "missing_image", 'Envie a imagem como multipart/form-data no campo "image".')
    settings: Settings = request.app.state.settings
    data = await image.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise ApiError(413, "image_too_large", f"A imagem excede o limite de {settings.max_upload_mb} MB.")
    return ImageUpload(data=data, filename=image.filename or "upload")
