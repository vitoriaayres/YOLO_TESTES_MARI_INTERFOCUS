"""Regra de negócio do POST /api/v1/decode."""
from time import perf_counter

from app.core.config import Settings
from app.schemas.decode import Barcode, DecodeDebug, DecodeResponse, ImageSize, UnreadBox
from app.services.barcode_reader import ACTIVE_DETECTORS, BarcodeReader, Detection
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
        barcodes = unique_barcodes(reading.barcodes)
        response = DecodeResponse(
            barcodes=barcodes,
            unread=unread_boxes(reading.detections, barcodes, self._settings.unread_min_confidence),
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


# Fração da caixa do detector que precisa estar coberta por um código lido
# para ela contar como lida (a caixa pode ser maior que o código, ou englobar
# dois códigos empilhados que a divisão separou).
READ_OVERLAP = 0.3


def unread_boxes(detections: list[Detection], barcodes: list[Barcode], min_confidence: float) -> list[UnreadBox]:
    """Caixas do detector que não coincidem com nenhum código lido, de cima para baixo."""
    unread = [
        UnreadBox(confidence=d.confidence, bbox=d.bbox)
        for d in detections
        if d.confidence >= min_confidence
        and not any(_overlap(d.bbox, b.bbox) >= READ_OVERLAP for b in barcodes)
    ]
    return sorted(unread, key=lambda box: (box.bbox[1], box.bbox[0]))


def _overlap(a: list[int], b: list[int]) -> float:
    """Interseção sobre a área da menor caixa ([x1, y1, x2, y2])."""
    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])
    if width <= 0 or height <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return width * height / max(smaller, 1)
