import pytest

from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.state import AgentState
from app.agent.graph import build_graph


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