"""API de identificação e decodificação de códigos de barras (FastAPI).

Rodar localmente:
  uvicorn app.main:app --port 5001

Contrato completo em API.md; documentação interativa em /docs.
"""
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import decode, health
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.services.decode_runner import ProcessPoolDecodeRunner

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Os modelos são carregados aqui, um por processo de leitura, e não no import.
    settings: Settings = app.state.settings
    runner = ProcessPoolDecodeRunner(settings)
    app.state.decode_runner = runner
    logger.info("API pronta (chave %s)", "exigida" if settings.key else "desligada")
    yield
    runner.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    app = FastAPI(title="barcode-api", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(decode.router)
    return app


app = create_app()
