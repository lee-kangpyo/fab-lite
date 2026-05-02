from __future__ import annotations

import os
import uuid

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter


def setup_telemetry(service_name: str) -> TracerProvider | None:
    if os.environ.get("TESTING"):
        return None

    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)

    from app.config import settings

    if settings.otel_enabled:
        exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            insecure=True,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))

    trace.set_tracer_provider(provider)
    return provider


def get_tracer(name: str = "fabrix-lite"):
    return trace.get_tracer(name)


def get_correlation_id(request) -> str:
    cid = request.headers.get("x-correlation-id", "")
    if cid:
        return cid
    return str(uuid.uuid4())


def get_langfuse_callback_handler():
    from langfuse.langchain import CallbackHandler
    from app.config import settings

    return CallbackHandler(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def create_langfuse_trace(langfuse_client, correlation_id: str, session_id: str):
    return langfuse_client.trace(
        name="chat_request",
        id=correlation_id,
        metadata={"session_id": session_id},
    )


def get_langfuse_client():
    from langfuse import Langfuse
    from app.config import settings

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
