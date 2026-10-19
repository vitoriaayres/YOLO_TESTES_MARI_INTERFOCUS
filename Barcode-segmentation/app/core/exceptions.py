"""Erros da API e conversão de todos eles para {"error": {"code", "message"}}."""
import logging
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """Erro previsto, devolvido ao cliente com status e código próprios."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def __reduce__(self) -> tuple[type["ApiError"], tuple[int, str, str]]:
        # Volta inteiro do processo de leitura para o processo da API.
        return ApiError, (self.status, self.code, self.message)


def error_response(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status)


async def _handle_api_error(_request: Request, error: ApiError) -> JSONResponse:
    return error_response(error.status, error.code, error.message)


async def _handle_http_error(_request: Request, error: StarletteHTTPException) -> JSONResponse:
    code = HTTPStatus(error.status_code).phrase.lower().replace(" ", "_")
    return error_response(error.status_code, code, str(error.detail))


async def _handle_validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
    first = error.errors()[0] if error.errors() else {}
    field = ".".join(str(part) for part in first.get("loc", ()))
    return error_response(422, "validation_error", f"{field}: {first.get('msg', 'inválido')}")


async def _handle_unexpected_error(request: Request, _error: Exception) -> JSONResponse:
    logger.exception("Falha inesperada em %s %s", request.method, request.url.path)
    return error_response(500, "internal_error", "Falha inesperada ao processar a requisição.")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ApiError, _handle_api_error)
    app.add_exception_handler(StarletteHTTPException, _handle_http_error)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)
    app.add_exception_handler(Exception, _handle_unexpected_error)
