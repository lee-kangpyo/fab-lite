from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_graph
from app.config import settings
from app.schemas.chat import (
    ChatMessageRequest,
    ChatMessageResponse,
    SessionCreateResponse,
    SessionHistoryResponse,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])

_sessions: dict[str, any] = {}


def _get_redis_for_cost_guard():
    if os.environ.get("TESTING"):
        import fakeredis.aioredis
        return fakeredis.aioredis.FakeRedis()
    from redis.asyncio import from_url
    return from_url(settings.redis_url_list[0])


@router.post("/sessions", response_model=SessionCreateResponse, status_code=201)
async def create_session(request: Request):
    session_id = str(uuid.uuid4())
    if os.environ.get("TESTING"):
        _sessions[session_id] = MemorySaver()
    else:
        _sessions[session_id] = request.app.state.saver

    return SessionCreateResponse(session_id=session_id)


@router.post("/sessions/{session_id}/message", response_model=ChatMessageResponse)
async def send_message(session_id: str, body: ChatMessageRequest, request: Request):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    redis = _get_redis_for_cost_guard()

    from app.core.cost_guard import check_token_budget
    if not await check_token_budget(redis, session_id, limit=settings.token_limit_per_session):
        await redis.aclose()
        raise HTTPException(
            status_code=429,
            detail=f"세션 토큰 한도 초과 (한도: {settings.token_limit_per_session})",
        )

    checkpointer = _sessions[session_id]
    graph = build_graph(checkpointer=checkpointer)

    config: dict = {"configurable": {"thread_id": session_id}}

    if not os.environ.get("TESTING"):
        from app.core.telemetry import (
            get_correlation_id,
            get_langfuse_callback_handler,
            get_langfuse_client,
            create_langfuse_trace,
        )
        from app.core.cost_guard import TokenCountingCallback

        correlation_id = get_correlation_id(request)

        langfuse_client = get_langfuse_client()
        langfuse_trace = create_langfuse_trace(
            langfuse_client, correlation_id, session_id
        )

        langfuse_handler = get_langfuse_callback_handler()
        token_counter = TokenCountingCallback(redis, session_id)

        config["callbacks"] = [langfuse_handler, token_counter]

        from opentelemetry import trace
        tracer = trace.get_tracer("fabrix-lite")
        with tracer.start_as_current_span("chat.send_message") as span:
            span.set_attribute("session_id", session_id)
            span.set_attribute("correlation_id", correlation_id)

            try:
                result = await graph.ainvoke(
                    {"messages": [HumanMessage(content=body.message)]},
                    config,
                )
            except Exception as e:
                span.set_attribute("error", str(e))
                raise
    else:
        try:
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=body.message)]},
                config,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if not os.environ.get("TESTING"):
        await redis.aclose()

    last_ai_message = None
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            last_ai_message = msg.content
            break

    return ChatMessageResponse(
        reply=last_ai_message or "응답을 생성하지 못했습니다.",
        intent=result.get("intent"),
        needs_confirmation=result.get("needs_confirmation", False),
    )


@router.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
async def get_history(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    checkpointer = _sessions[session_id]
    graph = build_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": session_id}}

    snapshot = await graph.aget_state(config)
    messages = []
    for msg in snapshot.values.get("messages", []):
        messages.append({
            "role": msg.type,
            "content": msg.content,
        })

    return SessionHistoryResponse(
        session_id=session_id,
        messages=messages,
    )