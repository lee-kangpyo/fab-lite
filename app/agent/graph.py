from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
import json as json
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
        prompt = f"""사용자 메시지에서 수정할 태스크 정보를 추출하세요.

[중요 규칙]
1. task_id가 직접 제공되면 그것을 사용합니다.
2. task_id가 없고 제목(title)만 있다면, 이 제목을 task_title 필드에 담습니다.
3. 상태 변경 요청("진행중으로", "완료로")이 있으면 status 필드를 추출합니다.

[필드]
- task_id: 태스크의 고유 ID (UUID 형식, 직접 제공된 경우)
- task_title: 태스크 제목 (ID가 없을 때 검색용으로 사용)
- title: 변경할 새 제목
- description: 변경할 설명
- priority: low, medium, high, urgent
- status: todo, in_progress, done

[예시]
입력: "테스트2를 진행중으로 옮겨줘"
출력: {{"task_title": "테스트2", "status": "in_progress"}}

입력: "de7f28cf... 상태를 완료로 변경해"
출력: {{"task_id": "de7f28cf-b248-49de-8745-9d27e6293ee8", "status": "done"}}

사용자 메시지: {last_human}"""
    elif intent == "delete":
        prompt = f"""사용자 메시지에서 삭제할 태스크 ID를 추출하세요.
태스크 ID는 UUID 형식(예: 20c66abc-4089-...)일 수 있습니다.

[필드]
- task_id: 삭제할 태스크의 고유 ID (필수)

반드시 유효한 JSON만 응답하세요.
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
    
    content = re.sub(r'<think.*?>.*?</think\s*>', '', content, flags=re.DOTALL).strip()
    
    print(f"[DEBUG extract] after think removal: {content[:200]}")
    
    json_match = re.search(r"({.*})", content, re.DOTALL)
    if json_match:
        content = json_match.group(1)
    
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {}
    
    print(f"[DEBUG extract] parsed task_data: {data}")

    return _extract_missing_fields({**state, "task_data": data})


async def _resolve_task_reference(state: AgentState, tools_by_name: dict) -> AgentState:
    """제목이나 설명으로 태스크를 검색하여 ID를 해결합니다."""
    task_data = state.get("task_data", {})
    
    if "task_id" not in task_data and "task_title" in task_data:
        list_tool = tools_by_name.get("list_tasks")
        if list_tool:
            result = await list_tool.ainvoke({})
            data = json.loads(result) if isinstance(result, str) else result
            tasks = data.get("tasks", []) if isinstance(data, dict) else data
            title = task_data["task_title"]
            
            print(f"[DEBUG resolve] task_title: {title}")
            print(f"[DEBUG resolve] tasks count: {len(tasks)}")
            for t in tasks:
                print(f"[DEBUG resolve]   - {t.get('title', '?')}")
            
            resolved = _find_task_by_title(tasks, title)
            print(f"[DEBUG resolve] matched: {len(resolved)}")
            
            if len(resolved) == 1:
                task_data["task_id"] = resolved[0]["id"]
                del task_data["task_title"]
            elif len(resolved) > 1:
                task_list = "\n".join(
                    f"{i+1}. {t['title']} (ID: {t['id'][:8]}..., 상태: {t.get('status', '')})" 
                    for i, t in enumerate(resolved)
                )
                action = {k: v for k, v in task_data.items() if k != "task_title"}
                return {
                    "messages": [AIMessage(content=f"'{title}'과(와) 일치하는 태스크가 {len(resolved)}개 있습니다:\n{task_list}\n\n번호, ID 또는 '모두'로 선택해주세요.")],
                    "task_data": task_data,
                    "needs_confirmation": True,
                    "pending_candidates": resolved,
                    "pending_action": action,
                }
            else:
                return {
                    "messages": [AIMessage(content=f"'{title}'과(와) 일치하는 태스크를 찾을 수 없습니다. 목록을 먼저 확인해주세요.")],
                    "task_data": {},
                    "missing_fields": [],
                    "needs_confirmation": False,
                }
    
    return {"task_data": task_data}


def _find_task_by_title(tasks: list, title: str) -> list:
    """태스크 목록에서 제목으로 검색하여 후보를 반환합니다."""
    exact_matches = []
    partial_matches = []
    for task in tasks:
        if isinstance(task, dict):
            task_title = task.get("title", "").lower()
            if task_title == title.lower():
                exact_matches.append(task)
            elif title.lower() in task_title:
                partial_matches.append(task)
    return exact_matches if exact_matches else partial_matches


async def _do_list(state: AgentState, tools_by_name: dict) -> AgentState:
    tool = tools_by_name.get("list_tasks")
    if not tool:
        return {"messages": [AIMessage(content="목록 도구를 찾을 수 없습니다.")], "needs_confirmation": False}

    result = await tool.ainvoke({})
    return {"messages": [AIMessage(content=f"태스크 목록:\n{result}")], "confirmed": True}


async def _execute_all(state: AgentState, tools_by_name: dict) -> AgentState:
    """pending_candidates에 있는 모든 태스크에 pending_action을 일괄 적용합니다."""
    candidates = state.get("pending_candidates", [])
    action = state.get("pending_action", {})
    intent = state.get("intent", "")

    tool_name_map = {
        "create": "create_task",
        "update": "update_task",
        "delete": "delete_task",
    }
    tool_name = tool_name_map.get(intent)
    if not tool_name or tool_name not in tools_by_name:
        return {"messages": [AIMessage(content="실행할 수 없는 작업입니다.")]}

    tool = tools_by_name[tool_name]
    results = []
    success_count = 0
    fail_count = 0

    for candidate in candidates:
        task_data = {"task_id": candidate["id"], **action}
        try:
            result = await tool.ainvoke(task_data)
            result_data = json.loads(result) if isinstance(result, str) else result
            if isinstance(result_data, dict) and result_data.get("status") == "error":
                fail_count += 1
                results.append(f"  ✗ {candidate['title']}: {result_data.get('message', '실패')}")
            else:
                success_count += 1
                results.append(f"  ✓ {candidate['title']}")
        except Exception as e:
            fail_count += 1
            results.append(f"  ✗ {candidate['title']}: {str(e)}")

    summary = "\n".join(results)
    message = f"일괄 처리 완료 (성공 {success_count}건, 실패 {fail_count}건):\n{summary}"

    return {
        "messages": [AIMessage(content=message)],
        "needs_confirmation": False,
        "confirmed": True,
        "error": None,
        "pending_candidates": [],
        "pending_action": {},
    }


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

    async def _resolve_node(s):
        return await _resolve_task_reference(s, tools_by_name)

    workflow.add_node("classify_intent", _classify_node)
    workflow.add_node("extract_info", _extract_node)
    workflow.add_node("resolve_reference", _resolve_node)
    workflow.add_node("check_missing", lambda s: _extract_missing_fields(s))
    workflow.add_node("ask_missing_info", ask_missing_info)
    workflow.add_node("confirm", confirm)

    async def _parse_node(s):
        return await parse_confirmation(s, llm)
    workflow.add_node("parse_confirmation", _parse_node)
    async def _execute_node(s):
        return await execute_action(s, tools_by_name)

    async def _list_node(s):
        return await _do_list(s, tools_by_name)

    workflow.add_node("execute", _execute_node)

    async def _execute_all_node(s):
        return await _execute_all(s, tools_by_name)

    workflow.add_node("execute_all", _execute_all_node)
    workflow.add_node("list_action", _list_node)
    workflow.add_node("respond", respond)
    workflow.add_node("cancel", lambda s: {
        "messages": [AIMessage(content="작업이 취소되었습니다.")],
        "needs_confirmation": False,
    })

    # [수정] 조건부 시작점: 컨펌 대기 중이면 바로 확인 노드로, 아니면 의도 분류로
    def route_start(state: AgentState) -> str:
        if state.get("needs_confirmation"):
            return "parse_confirmation"
        return "classify_intent"

    workflow.set_conditional_entry_point(route_start, {
        "parse_confirmation": "parse_confirmation",
        "classify_intent": "classify_intent"
    })

    # 의도 분류 후 라우팅
    workflow.add_conditional_edges("classify_intent", route_intent, {
        "extract_info": "extract_info",
        "list_action": "list_action",
        "respond": "respond",
    })

    #workflow.add_edge("extract_info", "check_missing")
    # 분리
    def needs_resolve(state: AgentState) -> str:
        task_data = state.get("task_data", {})
        if "task_id" not in task_data and "task_title" in task_data:
            return "resolve_reference"
        return "check_missing"

    workflow.add_conditional_edges("extract_info", needs_resolve, {
        "resolve_reference": "resolve_reference",
        "check_missing": "check_missing",
    })

    def after_resolve(state: AgentState) -> str:
        if state.get("needs_confirmation"):
            return END
        task_data = state.get("task_data", {})
        if "task_id" not in task_data and not state.get("messages"):
            return END
        return "check_missing"

    workflow.add_conditional_edges("resolve_reference", after_resolve, {
        END: END,
        "check_missing": "check_missing",
    })


    workflow.add_conditional_edges("check_missing", has_missing_fields, {
        "ask_missing_info": "ask_missing_info",
        "confirm": "confirm",
    })

    workflow.add_edge("ask_missing_info", END)
    workflow.add_edge("confirm", END)

    # [수정] 확인 결과에 따른 라우팅 (새로운 의도면 다시 분류로!)
    workflow.add_conditional_edges("parse_confirmation", should_execute, {
        "execute": "execute",
        "execute_all": "execute_all",
        "cancel": "cancel",
        "classify_intent": "classify_intent"
    })

    workflow.add_edge("execute", END)
    workflow.add_edge("execute_all", END)
    workflow.add_edge("cancel", END)
    workflow.add_edge("list_action", END)
    workflow.add_edge("respond", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)
