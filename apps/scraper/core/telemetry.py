"""OpenTelemetry export over OTLP/HTTP protobuf to the PostHog traces endpoint. One span per
source run and per live query, with sparky attributes. An empty host or project token disables
export."""

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

_provider: TracerProvider | None = None


@dataclass(frozen=True)
class ExportTarget:
    """Where spans are sent and the headers sent with them."""

    endpoint: str
    headers: dict[str, str] = field(repr=False)
    timeout_secs: float


def export_target(cfg: Telemetry) -> ExportTarget | None:
    """The traces endpoint and bearer header, or None when host or token is empty."""
    host = cfg.host.strip().rstrip("/")
    token = cfg.project_token.get_secret_value().strip()
    if not host or not token:
        return None
    path = "/" + cfg.traces_path.strip().lstrip("/")
    return ExportTarget(
        endpoint=host + path,
        headers={"Authorization": f"Bearer {token}"},
        timeout_secs=cfg.export_timeout_secs,
    )


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
    target = export_target(settings().telemetry)
    if target is None:
        log.warning("telemetry.disabled", reason="telemetry host or project token is empty")
        return
    _provider = TracerProvider(resource=Resource.create({"service.name": "scraper"}))
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
    return trace.get_tracer("scraper")
