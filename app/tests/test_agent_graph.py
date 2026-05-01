import pytest

from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.state import AgentState
from app.agent.graph import build_graph
from app.schemas.chat import SessionCreateResponse


@pytest.mark.asyncio
async def test_build_graph_creates_compiled_graph():
    graph = build_graph(checkpointer=MemorySaver())
    assert graph is not None
    assert hasattr(graph, "ainvoke")


@pytest.mark.asyncio
async def test_agent_state_creation():
    state: AgentState = {
        "messages": [],
        "intent": None,
        "task_data": {},
        "missing_fields": [],
        "needs_confirmation": False,
        "confirmed": False,
        "error": None,
    }
    assert state["intent"] is None
    assert state["task_data"] == {}
    assert state["missing_fields"] == []
    assert state["needs_confirmation"] is False
    assert state["confirmed"] is False


@pytest.mark.asyncio
async def test_create_session(client):
    resp = await client.post("/api/chat/sessions")
    assert resp.status_code == 201
    data = resp.json()
    assert "session_id" in data


@pytest.mark.asyncio
async def test_get_history_not_found(client):
    resp = await client.get("/api/chat/sessions/nonexistent/history")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_send_message_not_found(client):
    resp = await client.post("/api/chat/sessions/nonexistent/message", json={"message": "hello"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_flow_with_missing_info():
    with patch("langchain_openai.ChatOpenAI") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[
            MagicMock(content="create"),
            MagicMock(content='{"title": "", "priority": "high"}'),
        ])
        mock_llm_class.return_value = mock_llm

        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-missing"}}

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="태스크 만들어줘")]},
            config,
        )
        assert result.get("intent") == "create"
        assert "title" in result.get("missing_fields", [])


@pytest.mark.asyncio
async def test_create_flow_complete():
    with patch("langchain_openai.ChatOpenAI") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=[
            MagicMock(content="create"),
            MagicMock(content='{"title": "배포 준비", "priority": "urgent", "description": "금요일 배포"}'),
        ])
        mock_llm_class.return_value = mock_llm

        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-complete"}}

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="급한 태스크 배포 준비 만들어줘")]},
            config,
        )
        assert result.get("intent") == "create"
        assert result.get("needs_confirmation") is True
        assert result.get("task_data", {}).get("title") == "배포 준비"


@pytest.mark.asyncio
async def test_list_flow():
    with patch("langchain_openai.ChatOpenAI") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="list"))
        mock_llm_class.return_value = mock_llm

        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-list"}}

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="태스크 목록 보여줘")]},
            config,
        )
        assert result.get("intent") == "list"
        assert result.get("confirmed") is True


@pytest.mark.asyncio
async def test_unknown_intent():
    with patch("langchain_openai.ChatOpenAI") as mock_llm_class:
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=MagicMock(content="unknown"))
        mock_llm_class.return_value = mock_llm

        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "test-unknown"}}

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="오늘 날씨 어때?")]},
            config,
        )
        assert result.get("intent") == "unknown"