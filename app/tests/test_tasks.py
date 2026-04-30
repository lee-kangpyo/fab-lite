import uuid

import pytest


@pytest.mark.asyncio
async def test_create_task(client):
    resp = await client.post("/api/tasks", json={"title": "테스트", "priority": "high"})
    assert resp.status_code == 201
    data = resp.json()
    assert data["title"] == "테스트"
    assert data["priority"] == "high"
    assert data["status"] == "todo"
    assert "id" in data


@pytest.mark.asyncio
async def test_list_tasks(client):
    await client.post("/api/tasks", json={"title": "태스크1"})
    await client.post("/api/tasks", json={"title": "태스크2"})
    resp = await client.get("/api/tasks")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_list_tasks_filter(client):
    await client.post("/api/tasks", json={"title": "긴급", "priority": "urgent"})
    await client.post("/api/tasks", json={"title": "보통", "priority": "medium"})
    resp = await client.get("/api/tasks", params={"priority": "urgent"})
    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["priority"] == "urgent"


@pytest.mark.asyncio
async def test_get_task(client):
    create_resp = await client.post("/api/tasks", json={"title": "조회용"})
    task_id = create_resp.json()["id"]
    resp = await client.get(f"/api/tasks/{task_id}")
    assert resp.status_code == 200
    assert resp.json()["title"] == "조회용"


@pytest.mark.asyncio
async def test_get_task_not_found(client):
    resp = await client.get(f"/api/tasks/{uuid.uuid4()}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_update_task(client):
    create_resp = await client.post("/api/tasks", json={"title": "원본"})
    task_id = create_resp.json()["id"]
    resp = await client.put(f"/api/tasks/{task_id}", json={"title": "수정됨"})
    assert resp.status_code == 200
    assert resp.json()["title"] == "수정됨"


@pytest.mark.asyncio
async def test_change_status(client):
    create_resp = await client.post("/api/tasks", json={"title": "상태변경"})
    task_id = create_resp.json()["id"]
    resp = await client.patch(f"/api/tasks/{task_id}/status", json={"status": "in_progress"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "in_progress"


@pytest.mark.asyncio
async def test_delete_task(client):
    create_resp = await client.post("/api/tasks", json={"title": "삭제용"})
    task_id = create_resp.json()["id"]
    resp = await client.delete(f"/api/tasks/{task_id}")
    assert resp.status_code == 204
    resp = await client.get(f"/api/tasks/{task_id}")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_invalid_priority(client):
    resp = await client.post("/api/tasks", json={"title": "test", "priority": "invalid"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_invalid_status(client):
    create_resp = await client.post("/api/tasks", json={"title": "상태테스트"})
    task_id = create_resp.json()["id"]
    resp = await client.patch(f"/api/tasks/{task_id}/status", json={"status": "invalid"})
    assert resp.status_code == 422
