"""Recusa uploads grandes pelo Content-Length, antes de receber o corpo.

Sem isso o Starlette grava o arquivo inteiro em disco antes de a rota
checar o tamanho. Pedidos sem Content-Length (chunked) continuam chegando
até a checagem da rota; o limite no proxy cobre esse caso.
"""
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.exceptions import error_response

# Folga para os cabeçalhos do multipart em volta da foto.
MULTIPART_OVERHEAD_BYTES = 64 * 1024


class UploadLimitMiddleware:
    def __init__(self, app: ASGIApp, max_upload_bytes: int, max_upload_mb: int) -> None:
        self.app = app
        self.max_body_bytes = max_upload_bytes + MULTIPART_OVERHEAD_BYTES
        self.max_upload_mb = max_upload_mb

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http":
            length = dict(scope["headers"]).get(b"content-length", b"")
            if length.isdigit() and int(length) > self.max_body_bytes:
                response = error_response(
                    413, "image_too_large", f"A imagem excede o limite de {self.max_upload_mb} MB."
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
