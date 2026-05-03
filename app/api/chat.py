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
    print(f"\n[Debug] Creating session...")
    session_id = str(uuid.uuid4())
    
    try:
        if os.environ.get("TESTING"):
            print("[Debug] Using MemorySaver (Testing mode)")
            _sessions[session_id] = MemorySaver()
        else:
            print(f"[Debug] Using app.state.saver: {getattr(request.app.state, 'saver', 'MISSING')}")
            _sessions[session_id] = request.app.state.saver
        
        print(f"[Debug] Session created: {session_id}")
    except Exception as e:
        print(f"[Debug] Error in create_session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
        )
        from app.core.cost_guard import TokenCountingCallback

        correlation_id = get_correlation_id(request)

        from opentelemetry import trace
        from langfuse import propagate_attributes

        tracer = trace.get_tracer("fabrix-lite")
        with tracer.start_as_current_span("chat.send_message") as span:
            span.set_attribute("session_id", session_id)
            span.set_attribute("correlation_id", correlation_id)

            # 1. OTel이 방금 생성한 고유 명찰(Trace ID)을 가져옵니다.
            otel_trace_id = f"{span.get_span_context().trace_id:032x}"

            # 2. 이 명찰을 Langfuse SDK에게도 똑같이 쓰라고 강제합니다!
            langfuse_handler = get_langfuse_callback_handler(
                session_id=session_id, trace_id=otel_trace_id
            )
            token_counter = TokenCountingCallback(
                redis, session_id, reset_period=settings.token_limit_reset_period
            )

            config["callbacks"] = [langfuse_handler, token_counter]


            try:
                # 랭퓨즈 세션 추적 활성화
                with propagate_attributes(session_id=session_id):
                    result = await graph.ainvoke(
                        {"messages": [HumanMessage(content=body.message)]},
                        config,
                    )
            except Exception as e:
                span.set_attribute("error", str(e))
                await redis.aclose()
                # 에러가 났을 때도 보내기 위해 flush
                from app.core.telemetry import get_langfuse_client
                get_langfuse_client().flush()
                raise

            await redis.aclose()
            await token_counter.await_pending()
            
            # 테스트 스크립트가 너무 빨리 끝나는 것을 막고 데이터를 전송하도록 flush
            from app.core.telemetry import get_langfuse_client
            get_langfuse_client().flush()


    else:
        try:
            result = await graph.ainvoke(
                {"messages": [HumanMessage(content=body.message)]},
                config,
            )
        except Exception as e:
            await redis.aclose()
            raise HTTPException(status_code=500, detail=str(e))

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