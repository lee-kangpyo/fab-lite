from __future__ import annotations

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
        if isinstance(msg, (type(msg).__class__, )):
            if hasattr(msg, "type") and msg.type == "human":
                last_human = msg.content
                break

    from langchain_core.messages import HumanMessage, SystemMessage

    if not last_human:
        return {"task_data": {}, "missing_fields": []}

    intent = state.get("intent", "")
    if intent == "create":
        prompt = f"""사용자 메시지에서 태스크 생성 정보를 추출하세요.
        필드: title(필수), description(선택), priority(선택: low/medium/high/urgent), due_date(선택: ISO형식)
        JSON만 반환하세요. 추출할 수 없는 필드는 빈 문자열로 두세요.

        사용자 메시지: {last_human}"""
    elif intent == "update":
        prompt = f"""사용자 메시지에서 태스크 수정 정보를 추출하세요.
        필드: task_id(필수), title(선택), description(선택), priority(선택), status(선택: todo/in_progress/done/cancelled)
        JSON만 반환하세요.

        사용자 메시지: {last_human}"""
    elif intent == "delete":
        prompt = f"""사용자 메시지에서 태스크 ID를 추출하세요.
        필드: task_id(필수)
        JSON만 반환하세요.

        사용자 메시지: {last_human}"""
    else:
        return {"task_data": {}, "missing_fields": []}

    response = await llm.ainvoke([
        SystemMessage(content="반드시 유효한 JSON만 반환하세요. 다른 텍스트 없이."),
        HumanMessage(content=prompt),
    ])

    import json
    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        content = content.rsplit("```", 1)[0] if "```" in content else content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {}

    return _extract_missing_fields({**state, "task_data": data})


async def _do_list(state: AgentState, tools_by_name: dict) -> AgentState:
    tool = tools_by_name.get("list_tasks")
    if not tool:
        return {"messages": [type("AIMessage", (), {"content": "목록 도구를 찾을 수 없습니다."})()]}

    result = await tool.ainvoke({})
    from langchain_core.messages import AIMessage
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

    workflow.add_node("classify_intent", lambda s: classify_intent(s, llm))
    workflow.add_node("extract_info", lambda s: _extract_with_llm(s, llm))
    workflow.add_node("check_missing", lambda s: _extract_missing_fields(s))
    workflow.add_node("ask_missing_info", ask_missing_info)
    workflow.add_node("confirm", confirm)
    workflow.add_node("parse_confirmation", parse_confirmation)
    workflow.add_node("execute", lambda s: execute_action(s, tools_by_name))
    workflow.add_node("list_action", lambda s: _do_list(s, tools_by_name))
    workflow.add_node("respond", respond)
    workflow.add_node("cancel", lambda s: {
        "messages": [type("AIMessage", (), {"content": "작업이 취소되었습니다."})()],
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