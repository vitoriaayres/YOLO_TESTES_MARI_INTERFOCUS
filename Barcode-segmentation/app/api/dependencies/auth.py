"""Chave opcional no cabeçalho X-API-Key."""
import hmac

from fastapi import Request, Security
from fastapi.security import APIKeyHeader

from app.core.config import Settings
from app.core.exceptions import ApiError

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(request: Request, sent: str | None = Security(_api_key_header)) -> None:
    settings: Settings = request.app.state.settings
    if settings.key is None:
        return
    if not hmac.compare_digest((sent or "").encode(), settings.key.encode()):
        raise ApiError(401, "unauthorized", "Cabeçalho X-API-Key ausente ou inválido.")
