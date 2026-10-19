"""Execução do DecodeService fora do event loop da API.

Em produção as fotos vão para um pool de processos: cada processo carrega o
seu modelo e lê uma foto por vez, e as fotos esperam numa fila única até o
primeiro processo livre. Com vários workers do uvicorn isso não acontece: um
worker assíncrono aceita novos pedidos mesmo ocupado, e eles ficam presos
atrás da foto em andamento enquanto outro worker está parado.
"""
import asyncio
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Protocol

from fastapi.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.exceptions import ApiError
from app.schemas.decode import DecodeResponse
from app.services.barcode_reader import ACTIVE_DETECTORS, PipelineBarcodeReader
from app.services.decode_service import DecodeService

logger = logging.getLogger(__name__)


class DecodeRunner(Protocol):
    @property
    def detector_backend(self) -> str: ...

    @property
    def is_healthy(self) -> bool: ...

    async def decode(self, data: bytes, image_name: str, debug: bool) -> DecodeResponse: ...

    def close(self) -> None: ...


class ThreadDecodeRunner:
    """Roda o serviço numa thread do próprio processo (testes e desenvolvimento)."""

    def __init__(self, service: DecodeService) -> None:
        self._service = service

    @property
    def detector_backend(self) -> str:
        return self._service.detector_backend

    @property
    def is_healthy(self) -> bool:
        return self._service.is_healthy

    async def decode(self, data: bytes, image_name: str, debug: bool) -> DecodeResponse:
        return await run_in_threadpool(self._service.decode, data, image_name, debug)

    def close(self) -> None:
        return None


# Estado de cada processo de leitura (criado pelo initializer do pool).
_worker_service: DecodeService | None = None


def _init_worker(settings: Settings) -> None:
    global _worker_service
    _worker_service = DecodeService(PipelineBarcodeReader(settings), settings)


def _worker_detector_backend() -> str:
    return _worker_service.detector_backend


def _worker_decode(data: bytes, image_name: str, debug: bool) -> DecodeResponse:
    return _worker_service.decode(data, image_name, debug)


class ProcessPoolDecodeRunner:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._restart_lock = asyncio.Lock()
        self._detector_backend = "starting"
        self._executor = self._start_pool()

    @property
    def detector_backend(self) -> str:
        return self._detector_backend

    @property
    def is_healthy(self) -> bool:
        return self._detector_backend in ACTIVE_DETECTORS

    async def decode(self, data: bytes, image_name: str, debug: bool) -> DecodeResponse:
        executor = self._executor
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(executor, _worker_decode, data, image_name, debug)
        except BrokenProcessPool:
            await self._restart_pool(executor)
            raise ApiError(503, "worker_restarted", "O processo de leitura foi reiniciado. Tente novamente.")

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _start_pool(self) -> ProcessPoolExecutor:
        workers = self._settings.workers
        executor = ProcessPoolExecutor(
            max_workers=workers,
            # spawn: o processo novo não herda as threads do uvicorn.
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(self._settings,),
        )
        # Um pedido por processo faz o pool subir todos já no startup, cada um
        # carregando o seu modelo, em vez de na primeira foto.
        futures = [executor.submit(_worker_detector_backend) for _ in range(workers)]
        self._detector_backend = futures[0].result()
        for future in futures[1:]:
            future.result()
        logger.info("Pool de leitura pronto: %d processo(s), detector %s", workers, self._detector_backend)
        return executor

    async def _restart_pool(self, broken: ProcessPoolExecutor) -> None:
        async with self._restart_lock:
            if self._executor is not broken:
                return  # outro pedido já recriou o pool
            logger.error("Um processo de leitura caiu; recriando o pool.")
            broken.shutdown(wait=False, cancel_futures=True)
            self._executor = await asyncio.to_thread(self._start_pool)
