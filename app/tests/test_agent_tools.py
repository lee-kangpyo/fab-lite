import pytest
from langchain_core.tools import StructuredTool

from app.agent.tools import get_agent_tools


@pytest.mark.asyncio
async def test_create_task_tool(client):
    resp = await client.post("/api/tasks", json={"title": "도구 테스트", "priority": "high"})
    assert resp.status_code == 201
    task_id = resp.json()["id"]

    tools = get_agent_tools()
    create_tool = tools[0]
    assert create_tool.name == "create_task"

    result = await create_tool.ainvoke({"title": "새 태스크", "priority": "urgent"})
    assert "created" in result.lower() or "status" in result.lower()


@pytest.mark.asyncio
async def test_list_tasks_tool(client):
    await client.post("/api/tasks", json={"title": "목록용1", "priority": "high"})
    await client.post("/api/tasks", json={"title": "목록용2", "priority": "low"})

    tools = get_agent_tools()
    list_tool = tools[2]
    assert list_tool.name == "list_tasks"

    result = await list_tool.ainvoke({})
    assert isinstance(result, str)
    assert "목록용" in result


@pytest.mark.asyncio
async def test_get_task_tool(client):
    create_resp = await client.post("/api/tasks", json={"title": "조회도구"})
    task_id = create_resp.json()["id"]

    tools = get_agent_tools()
    get_tool = tools[3]
    assert get_tool.name == "get_task"

    result = await get_tool.ainvoke({"task_id": task_id})
    assert "조회도구" in result


@pytest.mark.asyncio
async def test_delete_task_tool(client):
    create_resp = await client.post("/api/tasks", json={"title": "삭제도구"})
    task_id = create_resp.json()["id"]

    tools = get_agent_tools()
    delete_tool = tools[4]
    assert delete_tool.name == "delete_task"

    result = await delete_tool.ainvoke({"task_id": task_id})
    assert "삭제" in result.lower() or "deleted" in result.lower()


@pytest.mark.asyncio
async def test_update_task_tool(client):
    create_resp = await client.post("/api/tasks", json={"title": "수정대상", "priority": "low"})
    task_id = create_resp.json()["id"]

    tools = get_agent_tools()
    update_tool = tools[1]
    assert update_tool.name == "update_task"

    result = await update_tool.ainvoke({"task_id": task_id, "title": "수정됨", "priority": "high"})
    assert "updated" in result.lower() or "status" in result.lower()
    assert "수정됨" in result


@pytest.mark.asyncio
async def test_tools_return_list(client):
    tools = get_agent_tools()
    assert len(tools) == 5
    names = [t.name for t in tools]
    assert "create_task" in names
    assert "update_task" in names
    assert "list_tasks" in names
    assert "get_task" in names
    assert "delete_task" in names
