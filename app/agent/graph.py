from __future__ import annotations

import re
from typing import Literal
from functools import partial

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from app.agent.state import AgentState
from app.agent.tools import get_agent_tools
from app.config import settings


async def classify_intent(state: AgentState, llm) -> AgentState:
    """사용자 메시지의 의도를 분류합니다."""
    messages = state.get("messages", [])

    last_human = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_human = msg.content
            break

    if not last_human:
        return {
            "classification": {
                "intent": "unknown",
                "confidence": 0.0,
                "description": "No user message found",
            }
        }

    system_prompt = """사용자 메시지의 의도를 다음 중 하나로 분류하세요:
- todo: 태스크 생성, 수정, 삭제, 조회 등 작업 요청
- unknown: 의도를 파악할 수 없는 경우
- other: 그 외의 일반 대화

응답은 반드시 유효한 JSON이어야 합니다:
{"intent": "todo"|"unknown"|"other", "confidence": 0.0~1.0, "description": "..."}"""

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=last_human)
    ])

    content = response.content.strip()
    content = re.sub(r'<think.*?>.*?</think\s*>', '', content, flags=re.DOTALL).strip()

    json_match = re.search(r'(\{.*\})', content, re.DOTALL)
    if json_match:
        content = json_match.group(1)

    try:
        import json
        parsed = json.loads(content)
        intent = parsed.get("intent", "other")
        confidence = float(parsed.get("confidence", 0.5))
        description = parsed.get("description", "")
    except (json.JSONDecodeError, ValueError, AttributeError):
        intent = "other"
        confidence = 0.5
        description = "Classification failed"

    return {
        "classification": {
            "intent": intent,
            "confidence": confidence,
            "description": description,
        }
    }


async def agent(state: AgentState, llm):
    """ReAct 루프의 핵심 노드. LLM이 도구를 선택하거나 직접 응답합니다."""
    classification = state.get("classification")
    if not classification or classification.get("intent") != "todo":
        return {
            "messages": [AIMessage(content="죄송합니다, 이해하지 못했습니다. 태스크 관련 명령을 입력해주세요.")]
        }

    tools = get_agent_tools()
    bound_llm = llm.bind_tools(tools)

    response = await bound_llm.ainvoke(state["messages"])
    message_with_tools = response

    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_call = response.tool_calls[0]

        if tool_call["name"] == "delete_task":
            interrupt_payload = {
                "message": "정말 삭제하시겠습니까?",
                "target": tool_call["args"]
            }
            user_input = interrupt(interrupt_payload)
            if user_input != "응":
                return {
                    "messages": [AIMessage(content="삭제가 취소되었습니다.")]
                }

    return {
        "messages": [response]
    }


def should_continue(state: AgentState) -> Literal["tool_node", END]:
    """tool_call 존재 여부에 따라 다음 노드를 결정합니다."""
    messages = state.get("messages", [])
    if not messages:
        return END

    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool_node"
    return END


def build_graph(checkpointer=None):
    llm = ChatOpenAI(
        base_url=settings.llm_api_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )

    tools = get_agent_tools()
    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)

    workflow.add_node("classify_intent", partial(classify_intent, llm=llm))
    workflow.add_node("agent", partial(agent, llm=llm))
    workflow.add_node("tool_node", tool_node)

    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tool_node": "tool_node", END: END}
    )

    workflow.add_edge("tool_node", "agent")

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)
