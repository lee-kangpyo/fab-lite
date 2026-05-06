from __future__ import annotations

import os
import re
import uuid

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.agent.graph import build_graph
from app.api.chat_utils import generate_and_update_title
from app.config import settings
from app.database import get_db
from app.models.chat import ChatSession
from app.schemas.chat import (
    ChatMessageRequest,
    ChatMessageResponse,
    SessionCreateResponse,
    SessionHistoryResponse,
    SessionListResponse,
)


def _validate_uuid(session_id: str) -> None:
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")


def _clean_content(content: str) -> str:
    if not isinstance(content, str):
        return str(content)
    # Remove <think>...</think> blocks
    cleaned = re.sub(r"<think.*?>.*?</think\s*>", "", content, flags=re.DOTALL)
    return cleaned.strip()


router = APIRouter(prefix="/api/chat", tags=["chat"])


def _get_redis_for_cost_guard():
    if os.environ.get("TESTING"):
        import fakeredis.aioredis
        return fakeredis.aioredis.FakeRedis()
    from redis.asyncio import from_url
    return from_url(settings.redis_url_list[0])


@router.post("/sessions", response_model=SessionCreateResponse, status_code=201)
async def create_session(db: AsyncSession = Depends(get_db)):
    new_session = ChatSession()
    db.add(new_session)
    await db.commit()
    await db.refresh(new_session)
    return SessionCreateResponse(session_id=str(new_session.id))


@router.get("/sessions", response_model=list[SessionListResponse])
async def list_sessions(db: AsyncSession = Depends(get_db)):
    stmt = select(ChatSession).order_by(desc(ChatSession.updated_at))
    result = await db.execute(stmt)
    sessions = result.scalars().all()
    return [
        {"id": str(s.id), "title": s.title, "created_at": s.created_at, "updated_at": s.updated_at}
        for s in sessions
    ]


@router.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
async def get_history(session_id: str, request: Request, db: AsyncSession = Depends(get_db)):
    _validate_uuid(session_id)
    stmt = select(ChatSession).where(ChatSession.id == session_id)
    result = await db.execute(stmt)
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Session not found in database")

    config = {"configurable": {"thread_id": session_id}}
    state = await request.app.state.saver.aget_tuple(config)

    messages = []
    role_mapping = {"human": "user", "ai": "assistant"}
    if state and state.checkpoint and "channel_values" in state.checkpoint:
        channel_values = state.checkpoint["channel_values"]
        if "messages" in channel_values:
            for msg in channel_values["messages"]:
                # Filter out tool messages and messages with empty content (like tool calls)
                if msg.type not in ["human", "ai"]:
                    continue
                
                cleaned_content = _clean_content(msg.content)
                if not cleaned_content:
                    continue

                mapped_role = role_mapping.get(msg.type, msg.type)
                messages.append({
                    "role": mapped_role,
                    "content": cleaned_content,
                })

    return SessionHistoryResponse(
        session_id=session_id,
        messages=messages,
    )


@router.post("/sessions/{session_id}/message", response_model=ChatMessageResponse)
async def send_message(
    session_id: str,
    body: ChatMessageRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    _validate_uuid(session_id)
    stmt = select(ChatSession).where(ChatSession.id == session_id)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found in database")

    redis = _get_redis_for_cost_guard()

    from app.core.cost_guard import check_token_budget
    if not await check_token_budget(redis, session_id, limit=settings.token_limit_per_session):
        await redis.aclose()
        raise HTTPException(
            status_code=429,
            detail=f"세션 토큰 한도 초과 (한도: {settings.token_limit_per_session})",
        )

    checkpointer = request.app.state.saver
    graph = build_graph(checkpointer=checkpointer)

    langfuse_handler = None
    token_counter = None
    otel_trace_id = None
    correlation_id = None

    if not os.environ.get("TESTING"):
        from app.core.telemetry import (
            get_correlation_id,
            get_langfuse_callback_handler,
        )
        from app.core.cost_guard import TokenCountingCallback
        from opentelemetry import trace

        correlation_id = get_correlation_id(request)
        tracer = trace.get_tracer("fabrix-lite")

        span = trace.get_current_span()
        if not span.get_span_context().is_valid:
            span = tracer.start_span("chat.send_message")

        otel_trace_id = f"{span.get_span_context().trace_id:032x}"
        span.set_attribute("session_id", session_id)
        span.set_attribute("correlation_id", correlation_id)

        langfuse_handler = get_langfuse_callback_handler(
            session_id=session_id, trace_id=otel_trace_id
        )
        token_counter = TokenCountingCallback(
            redis, session_id, reset_period=settings.token_limit_reset_period
        )

    config: dict = {
        "configurable": {"thread_id": session_id},
        "metadata": {
            "langfuse_session_id": session_id,
            "langfuse_trace_id": otel_trace_id,
            "correlation_id": correlation_id,
        },
        "callbacks": [h for h in [langfuse_handler, token_counter] if h is not None]
    }

    try:
        from langfuse import propagate_attributes

        # 현재 세션이 인터럽트(승인 대기) 상태인지 확인
        # graph.aget_state()는 tasks 속성이 있는 StateSnapshot을 반환함
        current_graph_state = await graph.aget_state(config)
        is_interrupted = (
            current_graph_state is not None
            and current_graph_state.tasks
            and any(t.interrupts for t in current_graph_state.tasks)
        )

        with propagate_attributes(session_id=session_id):
            if is_interrupted:
                # 인터럽트 지점에서 사용자 답변으로 재개
                result = await graph.ainvoke(
                    Command(resume=body.message),
                    config,
                )
            else:
                # 새 메시지로 그래프 실행
                result = await graph.ainvoke(
                    {"messages": [HumanMessage(content=body.message)]},
                    config,
                )
    except Exception as e:
        if not os.environ.get("TESTING"):
            from app.core.telemetry import get_langfuse_client
            get_langfuse_client().flush()
        await redis.aclose()
        raise
    finally:
        if token_counter:
            await token_counter.await_pending()
        await redis.aclose()

    last_ai_message = None
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            last_ai_message = _clean_content(msg.content)
            if last_ai_message: # Ensure we got a real reply, not just a tool call thought
                break

    messages_count = len(result.get("messages", []))
    if messages_count == 2:
        history_for_title = ""
        for msg in result.get("messages", []):
            role = "사용자" if isinstance(msg, HumanMessage) else "AI"
            history_for_title += f"{role}: {msg.content}\n"
        background_tasks.add_task(generate_and_update_title, session_id, history_for_title)

    from datetime import datetime, timezone
    session.updated_at = datetime.now(timezone.utc)
    await db.commit()

    if not last_ai_message:
        # Check if there are tool calls (likely waiting for interrupt confirmation)
        last_ai_with_tool = next((msg for msg in reversed(result.get("messages", [])) if isinstance(msg, AIMessage) and msg.tool_calls), None)
        if last_ai_with_tool:
            tool_name = last_ai_with_tool.tool_calls[0]["name"]
            if tool_name == "delete_task":
                last_ai_message = "태스크를 삭제하시겠습니까? 삭제하시려면 '응' 또는 '예'라고 답해주세요."
            else:
                last_ai_message = f"'{tool_name}' 작업을 진행하기 위해 승인이 필요합니다. 계속할까요?"
        else:
            last_ai_message = "응답을 생성하지 못했습니다."

    return ChatMessageResponse(
        reply=last_ai_message,
    )
