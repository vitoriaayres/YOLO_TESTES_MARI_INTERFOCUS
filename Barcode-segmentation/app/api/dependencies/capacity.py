"""Limite de fotos em andamento (sendo lidas + esperando na fila)."""
from collections.abc import AsyncIterator

from fastapi import Request

from app.core.config import Settings
from app.core.exceptions import ApiError


async def reserve_decode_slot(request: Request) -> AsyncIterator[None]:
    # Um único processo atende os pedidos (ver decode_runner), então o
    # contador no app.state vale para a API inteira.
    state = request.app.state
    settings: Settings = state.settings
    if state.pending_decodes >= settings.max_pending:
        raise ApiError(503, "busy", "A fila de leitura está cheia. Tente de novo em alguns segundos.")
    state.pending_decodes += 1
    try:
        yield
    finally:
        state.pending_decodes -= 1
