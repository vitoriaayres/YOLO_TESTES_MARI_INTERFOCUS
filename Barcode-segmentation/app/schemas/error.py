"""Formato de erro de todas as rotas (documentado no OpenAPI)."""
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
