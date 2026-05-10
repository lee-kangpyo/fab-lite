from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage
from langfuse import propagate_attributes
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from opentelemetry import trace

from app.agent.graph import build_graph
from app.api.chat_utils import generate_and_update_title
from app.config import settings
from app.core.cost_guard import check_token_budget, TokenCountingCallback
from app.core.telemetry import (
    get_correlation_id,
    get_langfuse_callback_handler,
    get_langfuse_client,
)
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


# ==========================================
# [헬퍼 함수 모음]
# SRP(단일 책임 원칙)를 지키기 위해 send_message의 역할들을 분리했습니다.
# ==========================================

async def _verify_session(db: AsyncSession, session_id: str) -> ChatSession:
    """1. 세션 검증 로직: 파라미터 유효성 및 DB 존재 여부 확인"""
    _validate_uuid(session_id)
    stmt = select(ChatSession).where(ChatSession.id == session_id)
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found in database")
    return session


async def _check_budget(redis, session_id: str) -> None:
    """2. 토큰 비용 체크 로직 (Cost Guard)"""
    if not await check_token_budget(redis, session_id, limit=settings.token_limit_per_session):
        await redis.aclose()
        raise HTTPException(
            status_code=429,
            detail=f"세션 토큰 한도 초과 (한도: {settings.token_limit_per_session})",
        )


def _setup_graph_config(request: Request, session_id: str, redis) -> tuple[dict, TokenCountingCallback | None]:
    """3. 모니터링 및 랭그래프 설정 로직"""
    langfuse_handler = None
    token_counter = None
    otel_trace_id = None
    correlation_id = None

    if not os.environ.get("TESTING"):
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
    return config, token_counter


async def _execute_graph(graph, config: dict, session_id: str, user_message: str) -> dict:
    """4. 랭그래프 실제 실행 로직 (인터럽트 처리 포함)"""
    # 현재 세션이 인터럽트(승인 대기) 상태인지 확인
    current_graph_state = await graph.aget_state(config)
    is_interrupted = (
        current_graph_state is not None
        and current_graph_state.tasks
        and any(t.interrupts for t in current_graph_state.tasks)
    )

    with propagate_attributes(session_id=session_id):
        if is_interrupted:
            # 인터럽트 지점에서 사용자 답변으로 재개
            return await graph.ainvoke(Command(resume=user_message), config)
        else:
            # 새 메시지로 그래프 실행
            return await graph.ainvoke({"messages": [HumanMessage(content=user_message)]}, config)


async def _process_response(result: dict, session_id: str, background_tasks: BackgroundTasks, session: ChatSession, db: AsyncSession) -> str:
    """5. AI 응답 후처리 로직 (메시지 추출, 제목 변경 등)"""
    last_ai_message = None
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage):
            last_ai_message = _clean_content(msg.content)
            if last_ai_message:
                break

    messages_count = len(result.get("messages", []))
    if messages_count == 2:
        history_for_title = ""
        for msg in result.get("messages", []):
            role = "사용자" if isinstance(msg, HumanMessage) else "AI"
            history_for_title += f"{role}: {msg.content}\n"
        background_tasks.add_task(generate_and_update_title, session_id, history_for_title)

    session.updated_at = datetime.now(timezone.utc)
    await db.commit()

    if not last_ai_message:
        # 도구 실행 전 승인 대기 메시지 세팅
        last_ai_with_tool = next((msg for msg in reversed(result.get("messages", [])) if isinstance(msg, AIMessage) and msg.tool_calls), None)
        if last_ai_with_tool:
            tool_name = last_ai_with_tool.tool_calls[0]["name"]
            if tool_name == "delete_task":
                last_ai_message = "태스크를 삭제하시겠습니까? 삭제하시려면 '응' 또는 '예'라고 답해주세요."
            else:
                last_ai_message = f"'{tool_name}' 작업을 진행하기 위해 승인이 필요합니다. 계속할까요?"
        else:
            last_ai_message = "응답을 생성하지 못했습니다."
            
    return last_ai_message


# ==========================================
# [메인 API 라우터]
# ==========================================
@router.post("/sessions/{session_id}/message", response_model=ChatMessageResponse)
async def send_message(
    session_id: str,
    body: ChatMessageRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """사용자 메시지를 받아 AI 그래프를 실행하고 응답을 반환합니다."""

    # 1. 세션 검증
    session = await _verify_session(db, session_id)

    # 2. 토큰 비용 체크
    redis = _get_redis_for_cost_guard()
    await _check_budget(redis, session_id)

    # 3. 그래프 실행 준비 (설정 및 모니터링)
    graph = build_graph(checkpointer=request.app.state.saver)
    config, token_counter = _setup_graph_config(request, session_id, redis)

    try:
        # 4. 랭그래프 AI 엔진 실행
        result = await _execute_graph(graph, config, session_id, body.message)
    except Exception:
        if not os.environ.get("TESTING"):
            get_langfuse_client().flush()
        raise
    finally:
        # 리소스 정리 (에러가 나든 안 나든 실행됨)
        if token_counter:
            await token_counter.await_pending()
        await redis.aclose()

    # 5. AI 응답 처리 및 DB 반영
    reply_message = await _process_response(result, session_id, background_tasks, session, db)

    return ChatMessageResponse(reply=reply_message)

