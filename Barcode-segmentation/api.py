"""
API JSON de identificação e decodificação de códigos de barras.

Diferenças em relação ao main.py (interface web):
  - Responde sempre JSON (inclusive erros), sem templates.
  - Processa a imagem em memória: nada é gravado em disco, então
    requisições simultâneas não sobrescrevem o resultado umas das outras.
  - Corrige a orientação EXIF das fotos tiradas pelo celular.
  - Imagens de debug só são devolvidas quando pedidas (?debug=true).

Endpoints:
  GET  /api/v1/health   -> status do serviço e do detector
  POST /api/v1/decode   -> multipart/form-data com o campo "image"

Contrato completo em API.md.

Rodar localmente:
  python api.py
"""
import base64
import hmac
import io
import os
import threading
import traceback
from time import perf_counter

import cv2
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.exceptions import HTTPException

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
except ImportError:
    pass

import barcode_pipeline as bp

MAX_UPLOAD_MB = int(os.environ.get("API_MAX_UPLOAD_MB", "10"))
# Imagens maiores que isso são reduzidas antes do pipeline; fotos de celular
# (12+ MP) não ajudam a leitura e deixam o processamento bem mais lento.
MAX_IMAGE_SIDE = int(os.environ.get("API_MAX_IMAGE_SIDE", "2000"))

def _env_flag(name, default):
    value = os.environ.get(name, "").strip().lower()
    return default if not value else value in {"1", "true", "yes", "on"}


# Filtros pensados para etiquetas de equipamentos (serial/MAC em Code 128/39).
# Códigos EAN/UPC lidos nessas etiquetas são fragmentos falsos de um Code 128.
# Para aceitar todos os formatos, defina API_ALLOWED_FORMATS e/ou
# API_VALUE_PATTERN como vazio no .env.
ALLOWED_FORMATS = {
    f.strip().upper()
    for f in os.environ.get("API_ALLOWED_FORMATS", "CODE128,CODE39,CODE93").split(",")
    if f.strip()
}
VALUE_PATTERN = os.environ.get("API_VALUE_PATTERN", r"^[A-Z0-9]{6,}$") or None
# Teto de tempo (s) para a segunda passagem nas caixas que não foram lidas.
SECOND_PASS_BUDGET_S = float(os.environ.get("API_SECOND_PASS_BUDGET_S", "6"))

bp.CONFIG["allowed_barcode_types"] = ALLOWED_FORMATS or None
bp.CONFIG["allowed_value_pattern"] = VALUE_PATTERN
bp.CONFIG["second_pass_time_budget_s"] = SECOND_PASS_BUDGET_S
# Divide caixas que englobam códigos empilhados. Com o RF-DETR isso leva o
# v2 de 179 para 180/197 no gabarito; com o Roboflow fica desligado.
bp.CONFIG["roboflow_split_unresolved_boxes"] = _env_flag(
    "API_SPLIT_UNRESOLVED_BOXES", bp.ROBOFLOW_BACKEND == "rfdetr"
)

# Se definida, toda rota (menos / e /health) exige o cabeçalho X-API-Key.
# Vazia = sem autenticação (quem hospeda pode proteger num gateway).
API_KEY = os.environ.get("API_KEY", "").strip() or None

# Consulta ao banco de equipamentos (SQLite de exemplo). A identificação do
# equipamento fica com quem consome a API, por isso vem desligada.
EQUIPMENT_LOOKUP = _env_flag("API_EQUIPMENT_LOOKUP", False)
if EQUIPMENT_LOOKUP:
    import equipment_repository as equipment_repo

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024

# O pipeline compartilha o detector e o CONFIG globais, e não foi
# escrito pensando em threads. Serializamos as análises dentro do processo;
# para atender mais requisições em paralelo, suba mais workers (processos).
_pipeline_lock = threading.Lock()


class ApiError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def _error_response(status, code, message):
    return jsonify({"error": {"code": code, "message": message}}), status


@app.before_request
def _check_api_key():
    if API_KEY is None or request.path in {"/", "/api/v1/health"}:
        return None
    sent = request.headers.get("X-API-Key", "")
    if not hmac.compare_digest(sent.encode(), API_KEY.encode()):
        raise ApiError(401, "unauthorized", "Cabeçalho X-API-Key ausente ou inválido.")
    return None


@app.errorhandler(ApiError)
def _handle_api_error(error):
    return _error_response(error.status, error.code, error.message)


@app.errorhandler(HTTPException)
def _handle_http_error(error):
    if error.code == 413:
        return _error_response(
            413, "image_too_large",
            f"A imagem excede o limite de {MAX_UPLOAD_MB} MB.",
        )
    return _error_response(
        error.code, error.name.lower().replace(" ", "_"), error.description
    )


@app.errorhandler(Exception)
def _handle_unexpected_error(error):
    traceback.print_exc()
    return _error_response(
        500, "internal_error", "Falha inesperada ao processar a requisição."
    )


def _read_image(file_storage):
    """Lê o upload em memória, aplica a rotação EXIF e devolve BGR (OpenCV)."""
    data = file_storage.read()
    if not data:
        raise ApiError(400, "empty_image", "O arquivo enviado está vazio.")
    try:
        pil_image = Image.open(io.BytesIO(data))
        pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = pil_image.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise ApiError(
            400, "invalid_image",
            "Não foi possível ler a imagem. Envie JPEG ou PNG.",
        )

    width, height = pil_image.size
    longest = max(width, height)
    if longest > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / longest
        pil_image = pil_image.resize(
            (round(width * scale), round(height * scale)), Image.LANCZOS
        )

    return cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


def _encode_jpeg_base64(image, quality=85):
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _region_to_json(region, fallback_index):
    x, y, w, h = region["bbox"]
    confidence = region.get("roboflow_confidence")
    if confidence is None:
        confidence = region.get("signature", {}).get("score")

    return {
        "index": region.get("barcode_id", fallback_index),
        "value": str(region["value"]),
        "format": bp.canonical_barcode_type(region.get("type", "")) or None,
        "confidence": round(float(confidence), 4) if confidence is not None else None,
        "bbox": [int(x), int(y), int(x + w), int(y + h)],
        "decoder": region.get("decoder") or None,
    }


def _lookup_equipment(codes):
    try:
        return equipment_repo.find_by_codes(codes)
    except Exception:
        traceback.print_exc()
        raise ApiError(
            503, "database_unavailable",
            "Não foi possível consultar o banco de equipamentos.",
        )


def _is_truthy(value):
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@app.get("/")
def root():
    endpoints = {
        "GET /api/v1/health": "status do serviço",
        "POST /api/v1/decode": 'multipart/form-data, campo "image" (?debug=true opcional)',
    }
    if EQUIPMENT_LOOKUP:
        endpoints["GET /api/v1/equipment/<codigo>"] = "consulta um código já lido no celular"
    return jsonify({"service": "barcode-api", "endpoints": endpoints})


@app.get("/api/v1/health")
def health():
    detector_ok = bp.ROBOFLOW_BACKEND in {"rfdetr", "local", "hosted"}
    response = {"detector_backend": bp.ROBOFLOW_BACKEND}
    database_ok = True
    if EQUIPMENT_LOOKUP:
        database_ok = equipment_repo.is_available()
        response["database"] = "ok" if database_ok else "indisponivel"
    response["status"] = "ok" if detector_ok and database_ok else "degraded"
    return jsonify(response)


if EQUIPMENT_LOOKUP:
    @app.get("/api/v1/equipment/<code>")
    def equipment(code):
        normalized = equipment_repo.normalize_code(code)
        found = _lookup_equipment([normalized])
        if normalized not in found:
            raise ApiError(
                404, "equipment_not_found",
                f"Nenhum equipamento cadastrado com o código {normalized}.",
            )
        return jsonify({"code": normalized, "equipment": found[normalized]})


@app.post("/api/v1/decode")
def decode():
    file_storage = request.files.get("image")
    if file_storage is None:
        raise ApiError(
            400, "missing_image",
            'Envie a imagem como multipart/form-data no campo "image".',
        )

    image = _read_image(file_storage)
    debug = _is_truthy(request.args.get("debug", "false"))

    start = perf_counter()
    with _pipeline_lock:
        regions, annotated, _table, visual_stack = bp.analyze_barcode_image(
            image, image_name=file_storage.filename or "upload",
            show_debug=False, show_output=False,
        )
    processing_ms = round((perf_counter() - start) * 1000)

    # Duas caixas podem ler o mesmo código (ex.: uma caixa grande e a
    # metade dela após a divisão); o app recebe cada valor uma vez só,
    # com a leitura de maior confiança.
    unique = {}
    for index, region in enumerate(regions, 1):
        barcode = _region_to_json(region, index)
        kept = unique.get(barcode["value"])
        if kept is None or (barcode["confidence"] or 0) > (kept["confidence"] or 0):
            unique[barcode["value"]] = barcode
    barcodes = sorted(unique.values(), key=lambda b: b["index"])

    height, width = image.shape[:2]
    response = {
        "barcodes": barcodes,
        # As bboxes são relativas a esta imagem (já rotacionada e reduzida).
        "image_size": {"width": int(width), "height": int(height)},
        "processing_ms": processing_ms,
    }

    if EQUIPMENT_LOOKUP:
        # O banco é o filtro final: leituras falsas simplesmente não existem nele.
        found = _lookup_equipment([b["value"] for b in barcodes])
        identified = {}
        for barcode in barcodes:
            match = found.get(equipment_repo.normalize_code(barcode["value"]))
            barcode["found"] = match is not None
            barcode["equipment_id"] = match["id"] if match else None
            if match:
                identified.setdefault(match["id"], match)
        # Equipamentos encontrados (sem repetição, mesmo que serial e MAC
        # da mesma etiqueta tenham sido lidos). Vazio = não cadastrado.
        response["equipment"] = list(identified.values())

    if debug:
        response["debug"] = {
            "annotated_jpeg_base64": _encode_jpeg_base64(annotated),
            "visual_stack_jpeg_base64": (
                _encode_jpeg_base64(visual_stack)
                if visual_stack is not None else None
            ),
        }

    return jsonify(response)


if __name__ == "__main__":
    print(f"Detector ativo: {bp.ROBOFLOW_BACKEND}")
    app.run(
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "5001")),
        debug=False,
        threaded=True,
    )
