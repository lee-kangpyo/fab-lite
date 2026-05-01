from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import build_graph
from app.schemas.chat import (
    ChatMessageRequest,
    ChatMessageResponse,
    SessionCreateResponse,
    SessionHistoryResponse,
)

router = APIRouter(prefix="/api/chat", tags=["chat"])

_sessions: dict[str, any] = {}
_pool_saver = None


@router.post("/sessions", response_model=SessionCreateResponse, status_code=201)
async def create_session(request: Request):
    session_id = str(uuid.uuid4())
    if os.environ.get("TESTING"):
        _sessions[session_id] = MemorySaver()
    else:
        # FastAPI app.state에 저장된 공용 세이버 사용
        _sessions[session_id] = request.app.state.saver
        
    return SessionCreateResponse(session_id=session_id)


@router.post("/sessions/{session_id}/message", response_model=ChatMessageResponse)
async def send_message(session_id: str, body: ChatMessageRequest):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    checkpointer = _sessions[session_id]
    graph = build_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": session_id}}

    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=body.message)]},
            config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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