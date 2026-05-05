from __future__ import annotations

import os
import uuid

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource



def setup_telemetry(service_name: str) -> TracerProvider | None:
    if os.environ.get("TESTING"):
        return None

    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)

    from app.config import settings

    if settings.otel_enabled:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        headers = {}
        if settings.otel_exporter_otlp_headers:
            for pair in settings.otel_exporter_otlp_headers.split(","):
                key, _, value = pair.partition("=")
                key = key.strip()
                if not key:
                    continue
                headers[key] = value.strip()

        exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
            headers=headers,
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


def get_langfuse_callback_handler(session_id: str = None, trace_id: str = None):
    import os
    from langfuse.langchain import CallbackHandler
    from app.config import settings

    # SDK 내부 클라이언트가 참조할 수 있도록 환경변수 세팅
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    os.environ["LANGFUSE_HOST"] = settings.langfuse_host

    # v4.5.1 버전에서는 trace_context를 통해 ID를 전달해야 합니다.
    return CallbackHandler(
        trace_context={
            "trace_id": trace_id,
            "session_id": session_id
        }
    )



def get_langfuse_client():
    from langfuse import Langfuse
    from app.config import settings

    return Langfuse(
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
