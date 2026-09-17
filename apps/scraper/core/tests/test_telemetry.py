from pydantic import SecretStr
from scraper.core.settings import Telemetry
from scraper.core.telemetry import PROJECT_NAME, exporter, phoenix_target, resource


def test_phoenix_target_joins_the_otlp_path_without_a_header() -> None:
    cfg = Telemetry(phoenix_url="http://phoenix:6006/")

    target = phoenix_target(cfg)

    assert target is not None
    assert target.endpoint == "http://phoenix:6006/v1/traces"
    assert target.headers == {}


def test_an_api_key_becomes_a_bearer_header() -> None:
    cfg = Telemetry(phoenix_url="http://phoenix:6006", phoenix_api_key=SecretStr("px_x"))

    target = phoenix_target(cfg)

    assert target is not None
    assert target.headers == {"Authorization": "Bearer px_x"}


def test_exporter_uses_target_endpoint_and_headers() -> None:
    cfg = Telemetry(phoenix_url="http://phoenix:6006", phoenix_api_key=SecretStr("px_x"))
    target = phoenix_target(cfg)
    assert target is not None

    exp = exporter(target)

    assert exp._endpoint == "http://phoenix:6006/v1/traces"
    assert exp._headers["Authorization"] == "Bearer px_x"


def test_an_empty_phoenix_url_disables_export() -> None:
    assert phoenix_target(Telemetry()) is None
    assert phoenix_target(Telemetry(phoenix_url=" ")) is None


def test_the_api_key_is_not_in_the_target_repr() -> None:
    cfg = Telemetry(phoenix_url="http://phoenix:6006", phoenix_api_key=SecretStr("px_secret"))

    assert "px_secret" not in repr(phoenix_target(cfg))


def test_the_resource_names_the_service_and_the_phoenix_project() -> None:
    attrs = resource(Telemetry(project_name="sparky-test")).attributes

    assert attrs["service.name"] == "scraper"
    assert attrs[PROJECT_NAME] == "sparky-test"


def test_unused_telemetry_keys_from_sparky_toml_are_ignored() -> None:
    cfg = Telemetry.model_validate({"provider_name": "llama.cpp", "sample_ratio": 1.0})

    assert cfg.project_name == "sparky"
