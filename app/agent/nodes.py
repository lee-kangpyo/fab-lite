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

    system_prompt = """사용자의 메시지를 분석하여 의도를 분류하세요. 
다음 키워드 중 하나만 선택하여 응답하세요:
- create: 태스크 생성/추가/등록
- update: 태스크 수정/변경/업데이트
- list: 태스크 목록 조회/리스트 확인
- delete: 태스크 삭제/제거
- unknown: 기타 대화

응답은 반드시 'create', 'update', 'list', 'delete', 'unknown' 중 하나의 단어여야 합니다."""

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_human or ""),
    ])
    
    content = response.content.strip().lower()
    
    # 텍스트 내에서 키워드 검색 (유연한 파싱)
    target_intents = ["create", "update", "list", "delete"]
    intent = "unknown"
    for candidate in target_intents:
        if candidate in content:
            intent = candidate
            break

    # 새로운 의도가 파악되면 이전 작업의 찌꺼기를 청소합니다.
    return {
        "intent": intent,
        "task_data": {},
        "missing_fields": [],
        "needs_confirmation": False,
        "confirmed": False
    }


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


async def parse_confirmation(state: AgentState, llm) -> AgentState:
    """LLM을 사용하여 사용자의 확인/취소/새로운 의도를 파악."""
    messages = state.get("messages", [])
    last_human = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content
            break

    if not last_human:
        return {"confirmed": False, "error": "no_input"}

    prompt = f"""사용자의 답변이 이전 작업에 대한 '승인'인지 '취소'인지, 아니면 '새로운 질문'인지 분류하세요.
    
    [사용자 답변]: {last_human}
    
    다음 중 하나만 응답하세요:
    - yes: 승인, 진행, 확인, 응, 그래, 좋아 등 긍정적인 반응
    - no: 취소, 하지마, 관둬, 아니 등 부정적인 반응
    - new_intent: 위의 확인/취소와 상관없는 새로운 질문이나 명령 (예: "목록 보여줘", "다른 거 해줘")
    
    응답:"""

    response = await llm.ainvoke([
        SystemMessage(content="단어 하나로만 응답하세요: yes, no, new_intent"),
        HumanMessage(content=prompt)
    ])
    
    result = response.content.strip().lower()
    
    if "yes" in result:
        return {"confirmed": True, "error": None}
    elif "new_intent" in result:
        # 새로운 질문이면 컨펌 상태를 해제하고 다시 의도 분류로 가도록 유도
        return {"confirmed": False, "needs_confirmation": False, "intent": "unknown", "error": "new_intent"}
    else:
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
