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

    system_prompt = """[SYSTEM] 당신은 의도 분류 전용 시스템입니다. 절대로 대화하지 마세요.

사용자 메시지의 의도를 다음 중 하나로 분류하세요:
- todo: 태스크 생성, 수정, 삭제, 조회 등 작업 요청
- unknown: 의도를 파악할 수 없는 경우 (전혀 관련 없는 내용, 외국어 등)
- other: 일반 대화, 이전 대화와 연결된 말("하나더 있어", "응", "아니 그게 아니라" 등), 불완전한 입력

중요: 사용자가 이전 대화 맥락과 연결된 말을 하거나, 불완전한 입력을 준 경우에는 todo가 아닌 other로 분류하세요. 예:
- "하나더 있어" → other
- "아니 진행중에 테스트2가 하나더 있다고" → todo (태스크 정보 제공)
- "응" → other
- "테스트2를 진행중으로 이동해줘" → todo

[CRITICAL] 응답은 반드시 아래 JSON 형식만 사용하세요. 어떤 경우에도 JSON 외의 텍스트(설명, 질문, 생각, markdown 등)를 출력하지 마세요:
{"intent": "todo"|"unknown"|"other", "confidence": 0.0~1.0, "description": "..."}"""

    # Human 메시지만 추출하여 맥락 제공 (AI 메시지는 제외하여 classifier가 대화 모드로 빠지지 않도록 함)
    human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]
    
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        *human_messages[-3:],  # 최근 3개 Human 메시지만 전달
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
    intent = classification.get("intent") if classification else "unknown"

    if intent == "unknown":
        return {
            "messages": [AIMessage(content="죄송합니다, 이해하지 못했습니다. 태스크 관련 명령을 입력해주세요.")]
        }

    # 시스템 메시지 구성: 도구 상시 사용 가능하도록 유도
    system_msg = "당신은 친절한 태스크 관리 어시스턴트입니다."
    if intent == "other":
        system_msg += " 이전 대화 맥락을 바탕으로 사용자의 말을 이해하고 자연스럽게 응답하세요."

    tools = get_agent_tools()
    bound_llm = llm.bind_tools(tools)

    # 대화 맥락에 시스템 메시지 추가
    messages = [SystemMessage(content=system_msg)] + state["messages"]
    response = await bound_llm.ainvoke(messages)

    return {
        "messages": [response]
    }


async def human_approval(state: AgentState) -> AgentState:
    """도구 실행 전 사용자 승인을 기다리는 전용 노드입니다."""
    messages = state.get("messages", [])
    if not messages:
        return state

    last_msg = messages[-1]
    if not (isinstance(last_msg, AIMessage) and last_msg.tool_calls):
        return state

    tool_call = last_msg.tool_calls[0]
    # delete_task 실행 전에만 멈춰서 승인을 받음
    if tool_call["name"] == "delete_task":
        interrupt_payload = {
            "message": "정말 삭제하시겠습니까?",
            "target": tool_call["args"]
        }
        # 여기서 실행을 멈추고 사용자 대답을 기다림
        user_input = interrupt(interrupt_payload)
        
        # 긍정적인 답변이 아니면 루프를 종료시키기 위한 플래그성 메시지 반환
        if user_input not in ["응", "예", "그래", "확인", "yes"]:
            return {
                "messages": [AIMessage(content="삭제 요청이 취소되었습니다.")]
            }
            
    return state


def should_continue(state: AgentState) -> Literal["human_approval", END]:
    """도구 호출 여부에 따라 승인 노드로 갈지 결정합니다."""
    messages = state.get("messages", [])
    if not messages:
        return END

    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "human_approval"
    return END


def should_execute(state: AgentState) -> Literal["tool_node", END]:
    """승인 결과를 확인하여 실제 도구를 실행할지 결정합니다."""
    messages = state.get("messages", [])
    if not messages:
        return END

    last_msg = messages[-1]
    # 만약 "취소되었습니다" 메시지가 있다면 실행하지 않고 종료
    if isinstance(last_msg, AIMessage) and "취소" in last_msg.content:
        return END
        
    return "tool_node"


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
    workflow.add_node("human_approval", human_approval)
    workflow.add_node("tool_node", tool_node)

    workflow.set_entry_point("classify_intent")
    workflow.add_edge("classify_intent", "agent")

    # 에이전트 답변에 도구가 있으면 승인 단계로 이동
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"human_approval": "human_approval", END: END}
    )

    # 승인 노드에서 승인이 완료되면 실제 도구 실행
    workflow.add_conditional_edges(
        "human_approval",
        should_execute,
        {"tool_node": "tool_node", END: END}
    )

    workflow.add_edge("tool_node", "agent")

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)
