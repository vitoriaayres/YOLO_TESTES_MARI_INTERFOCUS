"""Formato único de log para a API."""
import logging


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s [%(process)d] %(name)s: %(message)s",
    )
