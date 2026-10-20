"""Regra de negócio do POST /api/v1/decode."""
from time import perf_counter

from app.core.config import Settings
from app.schemas.decode import Barcode, DecodeDebug, DecodeResponse, ImageSize
from app.services.barcode_reader import ACTIVE_DETECTORS, BarcodeReader
from app.services.image_codec import decode_upload, encode_jpeg_base64


class DecodeService:
    def __init__(self, reader: BarcodeReader, settings: Settings) -> None:
        self._reader = reader
        self._settings = settings

    @property
    def detector_backend(self) -> str:
        return self._reader.detector_backend

    @property
    def is_healthy(self) -> bool:
        return self._reader.detector_backend in ACTIVE_DETECTORS

    def decode(self, data: bytes, image_name: str, debug: bool) -> DecodeResponse:
        """Lê os códigos da foto. Bloqueante (CPU): chame fora do event loop."""
        image = decode_upload(data, self._settings.max_image_side)

        start = perf_counter()
        reading = self._reader.read(image, image_name)
        processing_ms = round((perf_counter() - start) * 1000)

        height, width = image.shape[:2]
        response = DecodeResponse(
            barcodes=unique_barcodes(reading.barcodes),
            image_size=ImageSize(width=int(width), height=int(height)),
            processing_ms=processing_ms,
        )
        if debug:
            response.debug = DecodeDebug(
                annotated_jpeg_base64=encode_jpeg_base64(reading.annotated),
                visual_stack_jpeg_base64=encode_jpeg_base64(reading.visual_stack),
            )
        return response


def unique_barcodes(barcodes: list[Barcode]) -> list[Barcode]:
    """Cada valor uma vez, com a leitura de maior confiança, na ordem do pipeline.

    Duas caixas podem ler o mesmo código (ex.: uma caixa grande e a metade
    dela após a divisão).
    """
    unique: dict[str, Barcode] = {}
    for barcode in barcodes:
        kept = unique.get(barcode.value)
        if kept is None or (barcode.confidence or 0) > (kept.confidence or 0):
            unique[barcode.value] = barcode
    return sorted(unique.values(), key=lambda b: b.index)
