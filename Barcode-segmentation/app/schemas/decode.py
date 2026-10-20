"""Contrato de resposta do POST /api/v1/decode."""
from pydantic import BaseModel, Field


class Barcode(BaseModel):
    index: int
    value: str
    format: str | None = Field(description="Formato canônico, ex.: CODE128.")
    confidence: float | None = Field(description="Confiança do detector para a caixa (0 a 1).")
    bbox: list[int] = Field(description="[x1, y1, x2, y2] em pixels de image_size.", min_length=4, max_length=4)
    decoder: str | None


class UnreadBox(BaseModel):
    confidence: float = Field(description="Confiança do detector para a caixa (0 a 1).")
    bbox: list[int] = Field(description="[x1, y1, x2, y2] em pixels de image_size.", min_length=4, max_length=4)


class ImageSize(BaseModel):
    width: int
    height: int


class DecodeDebug(BaseModel):
    annotated_jpeg_base64: str | None
    visual_stack_jpeg_base64: str | None


class DecodeResponse(BaseModel):
    barcodes: list[Barcode] = Field(description="Cada valor uma vez, de cima para baixo.")
    unread: list[UnreadBox] = Field(
        description="Códigos que o detector achou mas não conseguiu ler. Não vazio = vale pedir outra foto, mais de perto."
    )
    # As bboxes são relativas a esta imagem (já rotacionada e reduzida).
    image_size: ImageSize
    processing_ms: int
    debug: DecodeDebug | None = Field(default=None, description="Só com ?debug=true.")
