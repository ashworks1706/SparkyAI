"""OpenTelemetry export over OTLP/HTTP to Phoenix. One span per source run or query."""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from scraper.core.settings import Telemetry, settings

log = structlog.get_logger()

# The OTLP/HTTP traces path, fixed by the protocol. Phoenix serves it under phoenix_url.
OTLP_TRACES_PATH = "/v1/traces"

# Resource attribute naming the Phoenix project a span belongs to.
PROJECT_NAME = "openinference.project.name"

_provider: TracerProvider | None = None


@dataclass(frozen=True)
class ExportTarget:
    """Where spans are sent and the headers sent with them."""

    endpoint: str
    headers: dict[str, str] = field(repr=False)
    timeout_secs: float


def phoenix_target(cfg: Telemetry) -> ExportTarget | None:
    """The Phoenix traces endpoint and its headers, or None when phoenix_url is empty."""
    url = cfg.phoenix_url.strip().rstrip("/")
    if not url:
        return None
    key = cfg.phoenix_api_key.get_secret_value().strip()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return ExportTarget(
        endpoint=url + OTLP_TRACES_PATH, headers=headers, timeout_secs=cfg.export_timeout_secs
    )


def resource(cfg: Telemetry) -> Resource:
    """The resource every exported span carries: service name and Phoenix project."""
    return Resource.create({"service.name": "scraper", PROJECT_NAME: cfg.project_name})


def exporter(target: ExportTarget) -> OTLPSpanExporter:
    """The OTLP/HTTP protobuf span exporter for one target."""
    return OTLPSpanExporter(
        endpoint=target.endpoint, headers=target.headers, timeout=target.timeout_secs
    )


def init() -> None:
    """Installs the tracer provider once. Safe to call when export is disabled."""
    global _provider
    if _provider is not None:
        return
    cfg = settings().telemetry
    target = phoenix_target(cfg)
    if target is None:
        log.warning("telemetry.disabled", reason="telemetry phoenix url is empty")
        return
    _provider = TracerProvider(resource=resource(cfg))
    _provider.add_span_processor(BatchSpanProcessor(exporter(target)))
    trace.set_tracer_provider(_provider)
    atexit.register(shutdown)


def shutdown() -> None:
    """Flushes pending spans."""
    global _provider
    if _provider is not None:
        _provider.shutdown()
        _provider = None


def tracer() -> trace.Tracer:
    """The scraper tracer."""
    return trace.get_tracer("scraper")
