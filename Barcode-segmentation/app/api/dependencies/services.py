"""Serviços criados no startup da API (ver app.main.lifespan)."""
from fastapi import Request

from app.services.decode_runner import DecodeRunner


def get_decode_runner(request: Request) -> DecodeRunner:
    return request.app.state.decode_runner
