from fastapi.testclient import TestClient

from app.main import create_app
from app.services.decode_runner import ThreadDecodeRunner
from app.services.decode_service import DecodeService
from tests.fakes import FakeReader, jpeg_bytes, make_barcode, make_settings


def make_client(reader: FakeReader | None = None, **settings: object) -> TestClient:
    config = make_settings(**settings)
    app = create_app(config)
    # Sem o "with", o TestClient não roda o lifespan: o modelo não é carregado.
    app.state.decode_runner = ThreadDecodeRunner(DecodeService(reader or FakeReader(), config))
    return TestClient(app)


def post_image(client: TestClient, data: bytes | None = None, **kwargs: object):
    files = {"image": ("foto.jpg", data if data is not None else jpeg_bytes(), "image/jpeg")}
    return client.post("/api/v1/decode", files=files, **kwargs)


def test_health_should_report_detector_backend() -> None:
    response = make_client().get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "detector_backend": "rfdetr"}


def test_decode_should_return_barcodes_without_debug_field() -> None:
    response = post_image(make_client(FakeReader([make_barcode(1, "ABC123456")])))
    assert response.status_code == 200
    body = response.json()
    assert body["barcodes"] == [{
        "index": 1, "value": "ABC123456", "format": "CODE128",
        "confidence": 0.9, "bbox": [0, 0, 10, 10], "decoder": "zxing",
    }]
    assert body["image_size"] == {"width": 64, "height": 48}
    assert "debug" not in body


def test_decode_should_include_debug_images_when_requested() -> None:
    response = post_image(make_client(), params={"debug": "true"})
    assert response.status_code == 200
    assert response.json()["debug"]["annotated_jpeg_base64"]


def test_decode_should_return_400_when_image_field_is_missing() -> None:
    response = make_client().post("/api/v1/decode", data={"other": "x"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_image"


def test_decode_should_return_400_when_file_is_not_an_image() -> None:
    response = post_image(make_client(), b"texto")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_image"


def test_decode_should_return_413_when_image_exceeds_upload_limit() -> None:
    response = post_image(make_client(max_upload_mb=0))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "image_too_large"


def test_decode_should_require_api_key_when_configured() -> None:
    client = make_client(key="segredo")
    assert post_image(client).status_code == 401
    assert post_image(client, headers={"X-API-Key": "errada"}).status_code == 401
    assert post_image(client, headers={"X-API-Key": "segredo"}).status_code == 200


def test_health_should_not_require_api_key() -> None:
    assert make_client(key="segredo").get("/api/v1/health").status_code == 200


def test_unknown_route_should_return_error_in_api_format() -> None:
    response = make_client().get("/nao-existe")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
