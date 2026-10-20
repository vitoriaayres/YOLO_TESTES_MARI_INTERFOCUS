"""Contratos das rotas de status."""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    detector_backend: str
    version: str


class ServiceInfo(BaseModel):
    service: str
    endpoints: dict[str, str]
