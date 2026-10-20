"""Configuração da API, lida das variáveis de ambiente (ou do .env)."""
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Todas as variáveis começam com API_ (ex.: API_KEY, API_MAX_UPLOAD_MB).

    O detector é escolhido pelo próprio pipeline, com DETECTOR_BACKEND e
    RFDETR_ONNX_PATH (ver barcode_pipeline.CONFIG).
    """

    model_config = SettingsConfigDict(env_prefix="API_", env_file=".env", extra="ignore")

    # Se definida, toda rota (menos / e /health) exige o cabeçalho X-API-Key.
    # Vazia = sem autenticação (quem hospeda pode proteger num gateway).
    key: str | None = None
    # Processos de leitura: cada um carrega o modelo (~650 MB) e lê uma foto
    # por vez, usando ~2 núcleos. Regra prática: 1 para cada 2 vCPUs.
    workers: int = Field(default=2, ge=1)
    # Fotos sendo lidas + esperando na fila. Acima disso o /decode responde 503
    # na hora, em vez de deixar a foto esperar além do timeout de quem chamou.
    # Com 2 workers, 8 pedidos levam ~30 s para esvaziar no pior caso.
    max_pending: int = Field(default=8, ge=1)
    max_upload_mb: int = 10
    # Imagens maiores que isso são reduzidas antes do pipeline; fotos de
    # celular (12+ MP) não ajudam a leitura e deixam o processamento mais lento.
    max_image_side: int = 2000
    # Filtros pensados para etiquetas de equipamentos (serial/MAC em Code 128/39).
    # Códigos EAN/UPC lidos nessas etiquetas são fragmentos falsos de um Code 128.
    # Vazios = aceitar todos os formatos / valores.
    allowed_formats: str = "CODE128,CODE39,CODE93"
    value_pattern: str | None = r"^[A-Z0-9]{6,}$"
    # Teto de tempo (s) para a segunda passagem nas caixas que não foram lidas.
    second_pass_budget_s: float = 6.0
    # Divide caixas que englobam códigos empilhados. None = ligado só com o
    # RF-DETR (leva o v2 de 179 para 180/197 no gabarito).
    split_unresolved_boxes: bool | None = None
    # Confiança mínima de uma caixa do detector para entrar em "unread"
    # (código achado mas não lido). No gabarito, com 0,5: avisa em 11 das 12
    # fotos com código faltando e dá alarme falso em 6 das 59 completas (EAN
    # ignorado de propósito, grade de ventilação). Abaixo de 0,3 os alarmes
    # falsos passam de 12.
    unread_min_confidence: float = Field(default=0.5, ge=0, le=1)
    log_level: str = "INFO"

    @field_validator("key", "value_pattern", "split_unresolved_boxes", mode="before")
    @classmethod
    def _empty_as_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def allowed_format_set(self) -> set[str]:
        return {f.strip().upper() for f in self.allowed_formats.split(",") if f.strip()}

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
