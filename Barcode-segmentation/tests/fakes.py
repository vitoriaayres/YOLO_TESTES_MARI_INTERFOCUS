"""Dublês compartilhados pelos testes (sem carregar o modelo)."""
import io

import numpy as np
from PIL import Image

from app.core.config import Settings
from app.schemas.decode import Barcode
from app.services.barcode_reader import Detection, PipelineReading


def make_settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)


def make_barcode(index: int, value: str, confidence: float | None = 0.9) -> Barcode:
    return Barcode(
        index=index, value=value, format="CODE128", confidence=confidence,
        bbox=[0, 0, 10, 10], decoder="zxing",
    )


def jpeg_bytes(width: int = 64, height: int = 48) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, "JPEG")
    return buffer.getvalue()


class FakeReader:
    def __init__(
        self,
        barcodes: list[Barcode] | None = None,
        backend: str = "rfdetr",
        detections: list[Detection] | None = None,
    ) -> None:
        self.barcodes = barcodes or []
        self.backend = backend
        self.detections = detections or []
        self.images: list[np.ndarray] = []

    @property
    def detector_backend(self) -> str:
        return self.backend

    def read(self, image: np.ndarray, image_name: str) -> PipelineReading:
        self.images.append(image)
        return PipelineReading(self.barcodes, annotated=image, visual_stack=None, detections=self.detections)
