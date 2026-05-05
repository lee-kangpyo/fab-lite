from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.state import AgentState


async def classify_intent(state: AgentState, llm) -> AgentState:
    """사용자 메시지를 분석하여 의도 파악."""
    messages = state.get("messages", [])
    
    # 1. 마지막 human 메시지 위치 찾기
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human_idx = i
            break
    
    last_human = messages[last_human_idx].content if last_human_idx >= 0 else ""
    
    # 2. last_human 이전 메시지 10개를 컨텍스트로 구성 (last_human 제외)
    context_start = max(0, last_human_idx - 10)
    context_messages = messages[context_start:last_human_idx]
    
    context = ""
    for msg in context_messages:
        if isinstance(msg, HumanMessage):
            context += f"사용자: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            context += f"에이전트: {content}\n"

    system_prompt = f"""아래 대화 맥락을 참고하여 사용자의 현재 메시지 의도를 분류하세요.

[대화 맥락]
{context if context else "(이전 대화 없음)"}

[의도별 키워드 사전]
- create (생성): "만들어", "추가해", "등록해", "새로", "생성해"
- update (수정/변경): "수정해", "변경해", "업데이트해", "바꿔", "고쳐"
  - **상태 변경**: "진행중으로", "완료로", "옮겨", "바꿔", "이동해", "상태를"
  - **내용 수정**: "제목을", "설명을", "우선순위를"
- list (조회): "보여줘", "목록", "리스트", "조회", "확인해", "뭐 있어"
- delete (삭제): "삭제해", "지워", "제거해", "없애"
- unknown: 위에 해당하지 않는 일반 대화

[예시]
- "테스트 만들어줘" → create
- "테스트를 진행중으로 옮겨줘" → update (상태 변경)
- "테스트 삭제해" → delete
- "뭐 있어?" → list


응답은 반드시 'create', 'update', 'list', 'delete', 'unknown' 중 하나의 단어여야 합니다."""

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_human)
    ])
    
    content = response.content.strip().lower()
    print(f"[DEBUG] last_human: {last_human}")
    print(f"[DEBUG] LLM response: {content}")
    
    # <think ...>...</think > 블록 제거
    content = re.sub(r'<think.*?>.*?</think\s*>', '', content, flags=re.DOTALL).strip()
    
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
    result_data = json.loads(result) if isinstance(result, str) else result

    if isinstance(result_data, dict) and result_data.get("status") == "error":
        return {"messages": [AIMessage(content=result_data["message"])]}

    message = _format_result(intent, task_data, result_data)

    return {
        "messages": [AIMessage(content=message)],
        "needs_confirmation": False,
        "confirmed": True,
        "error": None,
    }


def _format_result(intent: str, task_data: dict, result_data: dict) -> str:
    """도구 실행 결과를 사람이 읽기 쉬운 메시지로 변환합니다."""
    if intent == "create":
        title = task_data.get("title", "")
        task_id = result_data.get("id", "")
        return f"'{title}' 태스크가 생성되었습니다. (ID: {task_id[:8]}...)"
    elif intent == "update":
        task_id = task_data.get("task_id", "")
        changes = []
        if "status" in task_data:
            status_map = {"todo": "할 일", "in_progress": "진행 중", "done": "완료"}
            changes.append(f"상태 → {status_map.get(task_data['status'], task_data['status'])}")
        if "title" in task_data:
            changes.append(f"제목 → {task_data['title']}")
        if "priority" in task_data:
            changes.append(f"우선순위 → {task_data['priority']}")
        if "description" in task_data:
            changes.append("설명 변경")
        change_str = ", ".join(changes) if changes else "업데이트"
        return f"태스크({task_id[:8]}...) {change_str} 완료!"
    elif intent == "delete":
        return "태스크가 삭제되었습니다."
    return f"작업이 완료되었습니다."


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

    pending_candidates = state.get("pending_candidates", [])
    if pending_candidates:
        select_keywords = ["모두", "전부", "다", "전체", "모든"]
        if any(kw in last_human for kw in select_keywords):
            return {"confirmed": True, "error": None, "select_all": True}

        import re
        num_match = re.search(r'\d+', last_human)
        if num_match:
            idx = int(num_match.group()) - 1
            if 0 <= idx < len(pending_candidates):
                task_data = state.get("task_data", {})
                task_data["task_id"] = pending_candidates[idx]["id"]
                if "task_title" in task_data:
                    del task_data["task_title"]
                return {
                    "confirmed": True,
                    "error": None,
                    "select_all": False,
                    "task_data": task_data,
                    "pending_candidates": [],
                    "pending_action": {},
                }

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
    if error == "new_intent":
        return "classify_intent"
    if confirmed and not error:
        if state.get("select_all"):
            return "execute_all"
        return "execute"
    return "cancel"


def list_action(state: AgentState) -> AgentState:
    return {
        "task_data": {},
        "missing_fields": [],
    }
