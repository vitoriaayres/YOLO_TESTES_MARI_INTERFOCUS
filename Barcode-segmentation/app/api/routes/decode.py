"""POST /api/v1/decode: foto -> códigos de barras."""
from fastapi import APIRouter, Depends, Query

from app.api.dependencies.auth import require_api_key
from app.api.dependencies.capacity import reserve_decode_slot
from app.api.dependencies.services import get_decode_runner
from app.api.dependencies.uploads import ImageUpload, read_image_upload
from app.schemas.decode import DecodeResponse
from app.schemas.error import ErrorResponse
from app.services.decode_runner import DecodeRunner

router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])


@router.post(
    "/decode",
    response_model=DecodeResponse,
    dependencies=[Depends(reserve_decode_slot)],
    # Sem ?debug=true o campo debug nem aparece na resposta.
    response_model_exclude_unset=True,
    responses={code: {"model": ErrorResponse} for code in (400, 401, 413, 500, 503)},
)
async def decode(
    upload: ImageUpload = Depends(read_image_upload),
    debug: bool = Query(False, description="Devolve a foto anotada em base64."),
    runner: DecodeRunner = Depends(get_decode_runner),
) -> DecodeResponse:
    # A leitura roda fora do event loop: o /health continua respondendo
    # enquanto as fotos são processadas.
    return await runner.decode(upload.data, upload.filename, debug)
