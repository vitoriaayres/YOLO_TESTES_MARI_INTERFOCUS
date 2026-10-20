"""Rotas de status (sem chave)."""
from fastapi import APIRouter, Depends, Response

from app.api.dependencies.services import get_decode_runner
from app.schemas.health import HealthResponse, ServiceInfo
from app.services.decode_runner import DecodeRunner

router = APIRouter()


@router.get("/", response_model=ServiceInfo)
async def root() -> ServiceInfo:
    return ServiceInfo(
        service="barcode-api",
        endpoints={
            "GET /api/v1/health": "status do serviço",
            "POST /api/v1/decode": 'multipart/form-data, campo "image" (?debug=true opcional)',
            "GET /docs": "documentação interativa (OpenAPI)",
        },
    )


@router.get(
    "/api/v1/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Detector fora do ar"}},
)
async def health(response: Response, runner: DecodeRunner = Depends(get_decode_runner)) -> HealthResponse:
    # 503 quando degradado: Docker e load balancer olham só o status code.
    healthy = runner.is_healthy
    if not healthy:
        response.status_code = 503
    return HealthResponse(
        status="ok" if healthy else "degraded",
        detector_backend=runner.detector_backend,
    )
