from pydantic import SecretStr
from scraper.core.settings import Telemetry
from scraper.core.telemetry import export_target, export_targets, exporter, phoenix_target


def test_target_joins_host_and_path_with_bearer_header() -> None:
    cfg = Telemetry(host="http://posthog:8010/", project_token=SecretStr("phc_x"))

    target = export_target(cfg)

    assert target is not None
    assert target.endpoint == "http://posthog:8010/i/v1/traces"
    assert target.headers == {"Authorization": "Bearer phc_x"}


def test_exporter_uses_target_endpoint_and_headers() -> None:
    cfg = Telemetry(host="http://posthog:8010", project_token=SecretStr("phc_x"))
    target = export_target(cfg)
    assert target is not None

    exp = exporter(target)

    assert exp._endpoint == "http://posthog:8010/i/v1/traces"
    assert exp._headers["Authorization"] == "Bearer phc_x"


def test_empty_token_or_host_disables_export() -> None:
    assert export_target(Telemetry(project_token=SecretStr(""))) is None
    assert export_target(Telemetry(host=" ", project_token=SecretStr("phc_x"))) is None


def test_token_is_not_in_the_target_repr() -> None:
    target = export_target(Telemetry(project_token=SecretStr("phc_secret")))

    assert "phc_secret" not in repr(target)


def test_phoenix_target_joins_the_otlp_path_without_a_header() -> None:
    cfg = Telemetry(phoenix_url="http://phoenix:6006/")

    target = phoenix_target(cfg)

    assert target is not None
    assert target.endpoint == "http://phoenix:6006/v1/traces"
    assert target.headers == {}


def test_every_configured_destination_is_exported_to() -> None:
    both = Telemetry(project_token=SecretStr("phc_x"), phoenix_url="http://phoenix:6006")
    assert [t.endpoint for t in export_targets(both)] == [
        "http://localhost:8010/i/v1/traces",
        "http://phoenix:6006/v1/traces",
    ]

    phoenix_only = Telemetry(host="", phoenix_url="http://phoenix:6006")
    assert [t.endpoint for t in export_targets(phoenix_only)] == ["http://phoenix:6006/v1/traces"]

    assert export_targets(Telemetry(host="", phoenix_url=" ")) == []


def test_unused_telemetry_keys_from_sparky_toml_are_ignored() -> None:
    cfg = Telemetry.model_validate(
        {"ai_path": "/i/v0/ai/otel", "provider_name": "llama.cpp", "sample_ratio": 1.0}
    )

    assert cfg.traces_path == "/i/v1/traces"
