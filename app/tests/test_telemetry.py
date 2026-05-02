import os
import pytest
from unittest.mock import MagicMock

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter
from opentelemetry.sdk.trace import ReadableSpan


class InMemorySpanExporter(SpanExporter):
    def __init__(self):
        self._spans = []

    def export(self, spans):
        self._spans.extend(spans)
        from opentelemetry.sdk.trace.export import SpanExportResult
        return SpanExportResult.SUCCESS

    def get_finished_spans(self):
        return self._spans


@pytest.mark.asyncio
async def test_otel_초기화_테스트_환경_비활성화():
    os.environ["TESTING"] = "1"
    from app.core.telemetry import setup_telemetry
    result = setup_telemetry("fabrix-lite-test")
    assert result is None


@pytest.mark.asyncio
async def test_otel_span_생성():
    from opentelemetry import trace

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span("test-span") as span:
        span.set_attribute("key", "value")

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test-span"
    assert spans[0].attributes["key"] == "value"


@pytest.mark.asyncio
async def test_langfuse_콜백_핸들러_생성():
    from unittest.mock import patch, MagicMock
    from app.core.telemetry import get_langfuse_callback_handler

    try:
        from langfuse.langchain import CallbackHandler
        handler = get_langfuse_callback_handler()
        assert handler is not None
    except ModuleNotFoundError:
        pytest.skip("langchain not installed, skipping CallbackHandler test")


@pytest.mark.asyncio
async def test_correlation_id_추출():
    from app.core.telemetry import get_correlation_id

    request = MagicMock()
    request.headers = {"x-correlation-id": "test-correlation-123"}
    cid = get_correlation_id(request)
    assert cid == "test-correlation-123"


@pytest.mark.asyncio
async def test_correlation_id_없으면_생성():
    from app.core.telemetry import get_correlation_id

    request = MagicMock()
    request.headers = {}
    cid = get_correlation_id(request)
    assert cid is not None
    assert len(cid) > 0


@pytest.mark.asyncio
async def test_langfuse_trace_생성():
    from app.core.telemetry import create_langfuse_trace

    mock_langfuse = MagicMock()
    mock_trace = MagicMock()
    mock_langfuse.trace.return_value = mock_trace

    trace_obj = create_langfuse_trace(mock_langfuse, "test-corr-id", "test-session")
    mock_langfuse.trace.assert_called_once_with(
        name="chat_request",
        id="test-corr-id",
        metadata={"session_id": "test-session"},
    )
    assert trace_obj == mock_trace
