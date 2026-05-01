from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.state import AgentState


async def classify_intent(state: AgentState, llm) -> AgentState:
    """사용자 메시지를 분석하여 의도 파악."""
    messages = state.get("messages", [])
    last_human = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content
            break

    system_prompt = """당신은 태스크 관리 에이전트의 의도 분류기입니다.
사용자의 마지막 메시지를 분석하여 다 중 하나의 의도를 반환하세요:
- create: 새 태스크를 생성하려는 의도
- update: 기존 태스크를 수정하려는 의도
- list: 태스크 목록을 조회하려는 의도
- delete: 태스크를 삭제하려는 의도
- unknown: 그 외

응답은 반드시 의도 단어 하나만 반환하세요. 설명이나 추가 텍스트 없이."""

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_human or ""),
    ])
    intent = response.content.strip().lower()

    valid_intents = {"create", "update", "list", "delete", "unknown"}
    if intent not in valid_intents:
        intent = "unknown"

    return {"intent": intent}


async def ask_missing_info(state: AgentState) -> AgentState:
    """누락된 필수 필드가 있으면 사용자에게 질문."""
    missing = state.get("missing_fields", [])
    if not missing:
        return {}

    fields_korean = {
        "title": "제목",
        "priority": "우선순위",
        "description": "설명",
        "task_id": "태스크 ID",
    }
    missing_display = ", ".join(fields_korean.get(f, f) for f in missing)
    return {
        "messages": [AIMessage(content=f"다음 정보가 필요합니다: {missing_display}")],
        "needs_confirmation": False,
    }


async def confirm(state: AgentState) -> AgentState:
    """사용자에게 최종 확인을 요청 (Human-in-the-loop interrupt)."""
    task_data = state.get("task_data", {})
    intent = state.get("intent", "")
    lines = [f"다음 {intent} 작업을 실행하시겠습니까?"]
    for k, v in task_data.items():
        lines.append(f"  - {k}: {v}")
    lines.append("확인하려면 '응' 또는 '네', 취소하려면 '아니요'를 입력하세요.")

    return {
        "messages": [AIMessage(content="\n".join(lines))],
        "needs_confirmation": True,
    }


async def respond(state: AgentState) -> AgentState:
    """unknown 의도 처리."""
    return {
        "messages": [AIMessage(content="죄송합니다, 이해하지 못했습니다. 태스크 생성, 수정, 조회, 삭제 중 하나를 요청해주세요.")],
    }


async def execute_action(state: AgentState, tools_by_name: dict) -> AgentState:
    """확인 후 실제 도구 실행."""
    intent = state.get("intent", "")
    task_data = state.get("task_data", {})

    tool_name_map = {
        "create": "create_task",
        "update": "update_task",
        "delete": "delete_task",
    }
    tool_name = tool_name_map.get(intent)
    if not tool_name or tool_name not in tools_by_name:
        return {"messages": [AIMessage(content="실행할 수 없는 작업입니다.")]}

    tool = tools_by_name[tool_name]
    result = await tool.ainvoke(task_data)
    result_data = json.loads(result)

    if result_data.get("status") == "error":
        return {"messages": [AIMessage(content=result_data["message"])]}

    return {
        "messages": [AIMessage(content=f"완료되었습니다: {result}")],
        "needs_confirmation": False,
        "confirmed": True,
        "error": None,
    }


async def parse_confirmation(state: AgentState) -> AgentState:
    """사용자의 확인/취소 응답을 파악."""
    messages = state.get("messages", [])
    last_human = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content.strip().lower()
            break

    positive = {"응", "네", "예", "yes", "y", "확인", "좋아", "ㅇ", "ㅇㅇ"}
    if last_human in positive:
        return {"confirmed": True}
    return {"confirmed": False, "error": "cancelled"}


def route_intent(state: AgentState) -> str:
    intent = state.get("intent", "unknown")
    if intent == "list":
        return "list_action"
    if intent in ("create", "update", "delete"):
        return "extract_info"
    return "respond"


def has_missing_fields(state: AgentState) -> str:
    missing = state.get("missing_fields", [])
    if missing:
        return "ask_missing_info"
    return "confirm"


def should_execute(state: AgentState) -> str:
    confirmed = state.get("confirmed", False)
    error = state.get("error")
    if confirmed and not error:
        return "execute"
    return "cancel"


def list_action(state: AgentState) -> AgentState:
    return {
        "task_data": {},
        "missing_fields": [],
    }
