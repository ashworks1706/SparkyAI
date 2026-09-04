"""OpenTelemetry export over OTLP/gRPC to Phoenix (or any OTLP collector). One span per source
run, with OpenInference attributes so it reads like the engine's spans. Empty endpoint disables."""

from __future__ import annotations

import atexit

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from scraper.core.settings import settings

_provider: TracerProvider | None = None


def init() -> None:
    """Installs the tracer provider once. Safe to call when export is disabled."""
    global _provider
    endpoint = settings().telemetry.otlp_endpoint.strip()
    if not endpoint or _provider is not None:
        return
    _provider = TracerProvider(resource=Resource.create({"service.name": "scraper"}))
    _provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    )
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
