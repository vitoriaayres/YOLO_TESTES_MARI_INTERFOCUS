import tomllib
from pathlib import Path

from app import __version__
from tests.fakes import make_settings


def test_version_should_match_pyproject() -> None:
    pyproject = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == __version__


def test_settings_should_treat_empty_variables_as_disabled(monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "")
    monkeypatch.setenv("API_VALUE_PATTERN", "")
    monkeypatch.setenv("API_ALLOWED_FORMATS", "")
    monkeypatch.setenv("API_SPLIT_UNRESOLVED_BOXES", "")
    settings = make_settings()
    assert settings.key is None
    assert settings.value_pattern is None
    assert settings.allowed_format_set == set()
    assert settings.split_unresolved_boxes is None


def test_settings_should_parse_allowed_formats_list(monkeypatch) -> None:
    monkeypatch.setenv("API_ALLOWED_FORMATS", "code128, ean13")
    assert make_settings().allowed_format_set == {"CODE128", "EAN13"}
