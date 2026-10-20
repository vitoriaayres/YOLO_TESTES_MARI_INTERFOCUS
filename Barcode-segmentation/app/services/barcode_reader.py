"""Leitura dos códigos numa imagem, isolando o barcode_pipeline do resto da API."""
import importlib
import logging
import threading
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Protocol

import numpy as np

from app.core.config import Settings
from app.schemas.decode import Barcode

logger = logging.getLogger(__name__)

ACTIVE_DETECTORS = {"rfdetr", "local", "hosted"}


@dataclass
class PipelineReading:
    barcodes: list[Barcode]
    annotated: np.ndarray | None
    visual_stack: np.ndarray | None


class BarcodeReader(Protocol):
    @property
    def detector_backend(self) -> str: ...

    def read(self, image: np.ndarray, image_name: str) -> PipelineReading: ...


class PipelineBarcodeReader:
    """Detector RF-DETR + decodificadores do barcode_pipeline.

    O pipeline carrega o modelo ao ser importado, por isso o import fica aqui
    (chamado no startup da API) e não no topo do arquivo.
    """

    def __init__(self, settings: Settings) -> None:
        self._pipeline = importlib.import_module("barcode_pipeline")
        configure_pipeline(self._pipeline, settings)
        # O pipeline compartilha o detector e o CONFIG globais e não foi escrito
        # pensando em threads: uma análise por vez em cada processo. Para atender
        # mais requisições em paralelo, suba mais workers (processos).
        self._lock = threading.Lock()
        logger.info("Detector ativo: %s", self.detector_backend)

    @property
    def detector_backend(self) -> str:
        return str(self._pipeline.ROBOFLOW_BACKEND)

    def read(self, image: np.ndarray, image_name: str) -> PipelineReading:
        with self._lock:
            regions, annotated, _table, visual_stack = self._pipeline.analyze_barcode_image(
                image, image_name=image_name, show_debug=False, show_output=False
            )
        barcodes = [self._to_barcode(region, index) for index, region in enumerate(regions, 1)]
        return PipelineReading(barcodes, annotated, visual_stack)

    def _to_barcode(self, region: dict[str, Any], fallback_index: int) -> Barcode:
        x, y, w, h = region["bbox"]
        confidence = region.get("roboflow_confidence")
        if confidence is None:
            confidence = region.get("signature", {}).get("score")
        return Barcode(
            index=region.get("barcode_id", fallback_index),
            value=str(region["value"]),
            format=self._pipeline.canonical_barcode_type(region.get("type", "")) or None,
            confidence=round(float(confidence), 4) if confidence is not None else None,
            bbox=[int(x), int(y), int(x + w), int(y + h)],
            decoder=region.get("decoder") or None,
        )


def configure_pipeline(pipeline: ModuleType, settings: Settings) -> None:
    """Aplica no CONFIG global do pipeline os filtros e limites da API."""
    split = settings.split_unresolved_boxes
    if split is None:
        split = pipeline.ROBOFLOW_BACKEND == "rfdetr"
    pipeline.CONFIG["allowed_barcode_types"] = settings.allowed_format_set or None
    pipeline.CONFIG["allowed_value_pattern"] = settings.value_pattern
    pipeline.CONFIG["second_pass_time_budget_s"] = settings.second_pass_budget_s
    pipeline.CONFIG["roboflow_split_unresolved_boxes"] = split
