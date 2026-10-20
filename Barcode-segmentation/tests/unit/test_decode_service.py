import base64
import pickle

from app.core.exceptions import ApiError
from app.services.barcode_reader import Detection
from app.services.decode_service import DecodeService, unique_barcodes, unread_boxes
from tests.fakes import FakeReader, jpeg_bytes, make_barcode, make_settings


def test_unique_barcodes_should_keep_highest_confidence_reading_of_each_value() -> None:
    weak = make_barcode(1, "ABC123456", confidence=0.4)
    strong = make_barcode(3, "ABC123456", confidence=0.8)
    result = unique_barcodes([weak, make_barcode(2, "XYZ987654"), strong])
    assert [(b.index, b.value) for b in result] == [(2, "XYZ987654"), (3, "ABC123456")]


def test_unique_barcodes_should_treat_missing_confidence_as_zero() -> None:
    first = make_barcode(1, "ABC123456", confidence=None)
    second = make_barcode(2, "ABC123456", confidence=0.1)
    assert unique_barcodes([first, second]) == [second]


def test_decode_should_report_image_size_and_barcodes() -> None:
    reader = FakeReader([make_barcode(1, "ABC123456")])
    service = DecodeService(reader, make_settings())
    response = service.decode(jpeg_bytes(64, 48), "foto.jpg", debug=False)
    assert [b.value for b in response.barcodes] == ["ABC123456"]
    assert (response.image_size.width, response.image_size.height) == (64, 48)
    assert response.processing_ms >= 0
    assert "debug" not in response.model_fields_set


def test_decode_should_shrink_image_before_reading() -> None:
    reader = FakeReader()
    DecodeService(reader, make_settings(max_image_side=32)).decode(jpeg_bytes(64, 48), "foto.jpg", debug=False)
    assert reader.images[0].shape[:2] == (24, 32)


def test_decode_should_return_annotated_jpeg_when_debug_is_requested() -> None:
    service = DecodeService(FakeReader(), make_settings())
    response = service.decode(jpeg_bytes(), "foto.jpg", debug=True)
    assert base64.b64decode(response.debug.annotated_jpeg_base64)[:2] == b"\xff\xd8"
    assert response.debug.visual_stack_jpeg_base64 is None


def test_unread_boxes_should_skip_boxes_covered_by_a_reading() -> None:
    read = make_barcode(1, "ABC123456")  # bbox [0, 0, 10, 10]
    detections = [
        Detection(bbox=[0, 0, 20, 10], confidence=0.9),  # metade coberta pela leitura
        Detection(bbox=[0, 50, 10, 60], confidence=0.8),  # longe de qualquer leitura
        Detection(bbox=[0, 30, 10, 40], confidence=0.7),
    ]
    unread = unread_boxes(detections, [read], min_confidence=0.0)
    assert [box.bbox for box in unread] == [[0, 30, 10, 40], [0, 50, 10, 60]]


def test_unread_boxes_should_ignore_weak_detections() -> None:
    detections = [Detection(bbox=[0, 0, 10, 10], confidence=0.2), Detection(bbox=[0, 20, 10, 30], confidence=0.6)]
    unread = unread_boxes(detections, [], min_confidence=0.5)
    assert [box.confidence for box in unread] == [0.6]


def test_decode_should_report_unread_boxes() -> None:
    reader = FakeReader(detections=[Detection(bbox=[1, 2, 30, 20], confidence=0.8)])
    response = DecodeService(reader, make_settings()).decode(jpeg_bytes(), "foto.jpg", debug=False)
    assert response.barcodes == []
    assert [(box.bbox, box.confidence) for box in response.unread] == [([1, 2, 30, 20], 0.8)]


def test_is_healthy_should_be_false_when_detector_is_not_loaded() -> None:
    assert not DecodeService(FakeReader(backend="disabled"), make_settings()).is_healthy


def test_api_error_should_survive_pickling_between_processes() -> None:
    error = pickle.loads(pickle.dumps(ApiError(400, "invalid_image", "ilegível")))
    assert (error.status, error.code, error.message) == (400, "invalid_image", "ilegível")


def test_decode_response_should_survive_pickling_between_processes() -> None:
    response = DecodeService(FakeReader([make_barcode(1, "ABC123456")]), make_settings()).decode(
        jpeg_bytes(), "foto.jpg", debug=False
    )
    restored = pickle.loads(pickle.dumps(response))
    assert restored == response
    assert "debug" not in restored.model_fields_set
