import pytest

from app.agent.state import AgentState


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