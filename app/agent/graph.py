from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from app.agent.state import AgentState
from app.agent.nodes import (
    ask_missing_info,
    classify_intent,
    confirm,
    execute_action,
    list_action,
    parse_confirmation,
    respond,
    route_intent,
    has_missing_fields,
    should_execute,
)
from app.agent.tools import get_agent_tools


def _extract_missing_fields(state: AgentState) -> AgentState:
    intent = state.get("intent", "")
    task_data = state.get("task_data", {})

    if intent == "create":
        required = {"title"}
    elif intent == "update":
        required = {"task_id"}
    elif intent == "delete":
        required = {"task_id"}
    else:
        required = set()

    task_data = {k: v for k, v in task_data.items() if v is not None and v != ""}
    missing = [f for f in required if f not in task_data]

    return {"missing_fields": missing, "task_data": task_data}


async def _extract_with_llm(state: AgentState, llm) -> AgentState:
    messages = state.get("messages", [])
    last_human = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content
            break

    if not last_human:
        return {"task_data": {}, "missing_fields": []}

    intent = state.get("intent", "")
    if intent == "create":
        prompt = f"""사용자 메시지에서 태스크 정보를 추출하여 JSON으로 반환하세요.
반드시 다음 필드명을 사용하세요:
- title: 태스크 제목 (필수)
- priority: low, medium, high, urgent 중 하나 (기본값: medium)
- description: 상세 내용 (선택)
- due_date: YYYY-MM-DD 형식 (선택)

[예시]
입력: "내일 아침 9시에 주간 보고서 작성해줘. 아주 급해."
출력: {{"title": "주간 보고서 작성", "priority": "urgent", "due_date": "2024-05-02", "description": ""}}

사용자 메시지: {last_human}
출력:"""
    elif intent == "update":
        prompt = f"""사용자 메시지에서 태스크 수정 정보를 JSON 형식으로 추출하세요.
필드: task_id, title, description, priority, status
추출할 수 없는 필드는 null로 설정하세요. 반드시 JSON만 응답하세요.

사용자 메시지: {last_human}"""
    elif intent == "delete":
        prompt = f"""사용자 메시지에서 삭제할 태스크 ID를 추출하세요.
필드: task_id
반드시 JSON만 응답하세요.

사용자 메시지: {last_human}"""
    else:
        return {"task_data": {}, "missing_fields": []}

    response = await llm.ainvoke([
        SystemMessage(content="반드시 유효한 JSON만 반환하세요. 다른 텍스트 없이."),
        HumanMessage(content=prompt),
    ])

    import json
    import re
    content = response.content.strip()
    
    # JSON 블록 추출 ({ ... })
    json_match = re.search(r"({.*})", content, re.DOTALL)
    if json_match:
        content = json_match.group(1)
    
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {}

    return _extract_missing_fields({**state, "task_data": data})


async def _do_list(state: AgentState, tools_by_name: dict) -> AgentState:
    tool = tools_by_name.get("list_tasks")
    if not tool:
        return {"messages": [AIMessage(content="목록 도구를 찾을 수 없습니다.")], "needs_confirmation": False}

    result = await tool.ainvoke({})
    return {"messages": [AIMessage(content=f"태스크 목록:\n{result}")], "confirmed": True}


def build_graph(checkpointer=None) -> "CompiledGraph":
    from langchain_openai import ChatOpenAI
    from app.config import settings

    llm = ChatOpenAI(
        base_url=settings.llm_api_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )
    tools = get_agent_tools()
    tools_by_name = {t.name: t for t in tools}

    workflow = StateGraph(AgentState)

    async def _classify_node(s):
        return await classify_intent(s, llm)

    async def _extract_node(s):
        return await _extract_with_llm(s, llm)

    workflow.add_node("classify_intent", _classify_node)
    workflow.add_node("extract_info", _extract_node)
    workflow.add_node("check_missing", lambda s: _extract_missing_fields(s))
    workflow.add_node("ask_missing_info", ask_missing_info)
    workflow.add_node("confirm", confirm)
    workflow.add_node("parse_confirmation", parse_confirmation)
    async def _execute_node(s):
        return await execute_action(s, tools_by_name)

    async def _list_node(s):
        return await _do_list(s, tools_by_name)

    workflow.add_node("execute", _execute_node)
    workflow.add_node("list_action", _list_node)
    workflow.add_node("respond", respond)
    workflow.add_node("cancel", lambda s: {
        "messages": [AIMessage(content="작업이 취소되었습니다.")],
        "needs_confirmation": False,
    })

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges("classify_intent", route_intent, {
        "extract_info": "extract_info",
        "list_action": "list_action",
        "respond": "respond",
    })
    workflow.add_edge("extract_info", "check_missing")
    workflow.add_conditional_edges("check_missing", has_missing_fields, {
        "ask_missing_info": "ask_missing_info",
        "confirm": "confirm",
    })
    workflow.add_edge("ask_missing_info", END)
    workflow.add_edge("confirm", END)
    workflow.add_conditional_edges("parse_confirmation", should_execute, {
        "execute": "execute",
        "cancel": "cancel",
    })
    workflow.add_edge("execute", END)
    workflow.add_edge("cancel", END)
    workflow.add_edge("list_action", END)
    workflow.add_edge("respond", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)