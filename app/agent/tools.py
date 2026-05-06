from __future__ import annotations

import json
import uuid

from langchain_core.tools import tool
from sqlalchemy import select

from app.database import async_session
from app.models.task import Task


@tool
async def create_task(title: str, description: str = "", priority: str = "medium", due_date: str | None = None) -> str:
    """태스크를 생성합니다. title은 필수, priority는 low/medium/high/urgent, due_date는 ISO 형식(선택)."""
    async with async_session() as session:
        task = Task(title=title, description=description or None, priority=priority)
        if due_date:
            from datetime import datetime
            task.due_date = datetime.fromisoformat(due_date)
        session.add(task)
        await session.commit()
        await session.refresh(task)
        return json.dumps({
            "status": "created",
            "task_id": str(task.id),
            "title": task.title,
            "priority": task.priority,
        }, ensure_ascii=False)


@tool
async def update_task(task_id: str, title: str | None = None, description: str | None = None, priority: str | None = None, status: str | None = None) -> str:
    """
    기존 태스크의 정보를 수정합니다. task_id는 필수입니다.
    - 상태를 변경(이동)할 때는 반드시 status 파라미터('todo', 'in_progress', 'done')를 명시해야 합니다.
    - 제목(title), 설명(description), 우선순위(priority) 등 변경하고 싶은 필드만 선택적으로 입력할 수 있습니다.
    """
    try:
        task_uuid = uuid.UUID(task_id)
    except (ValueError, AttributeError):
        return json.dumps({"status": "error", "message": "Invalid task ID format"}, ensure_ascii=False)
    async with async_session() as session:
        task = await session.get(Task, task_uuid)
        if not task:
            return json.dumps({"status": "error", "message": "Task not found"}, ensure_ascii=False)
        if all(v is None for v in [title, description, priority, status]):
            return json.dumps({
                "status": "error", 
                "message": "수정할 값이 입력되지 않았습니다. title, description, priority, status 중 하나 이상을 입력해주세요."
            }, ensure_ascii=False)
        if title is not None:
            task.title = title
        if description is not None:
            task.description = description
        if priority is not None:
            task.priority = priority
        if status is not None:
            task.status = status
        await session.commit()
        await session.refresh(task)
        return json.dumps({
            "status": "updated",
            "task_id": str(task.id),
            "title": task.title,
        }, ensure_ascii=False)


@tool
async def list_tasks(status: str | None = None, priority: str | None = None) -> str:
    """태스크 목록을 조회합니다. status/priority로 필터링 가능."""
    async with async_session() as session:
        stmt = select(Task)
        if status:
            stmt = stmt.where(Task.status == status)
        if priority:
            stmt = stmt.where(Task.priority == priority)
        result = await session.execute(stmt.order_by(Task.created_at.desc()))
        tasks = result.scalars().all()
        return json.dumps({
            "status": "listed",
            "count": len(tasks),
            "tasks": [{"id": str(t.id), "title": t.title, "priority": t.priority, "status": t.status} for t in tasks],
        }, ensure_ascii=False)


@tool
async def get_task(task_id: str) -> str:
    """단일 태스크를 조회합니다. task_id는 필수."""
    try:
        task_uuid = uuid.UUID(task_id)
    except (ValueError, AttributeError):
        return json.dumps({"status": "error", "message": "Invalid task ID format"}, ensure_ascii=False)
    async with async_session() as session:
        task = await session.get(Task, task_uuid)
        if not task:
            return json.dumps({"status": "error", "message": "Task not found"}, ensure_ascii=False)
        return json.dumps({
            "status": "found",
            "task_id": str(task.id),
            "title": task.title,
            "description": task.description,
            "priority": task.priority,
            "status": task.status,
            "due_date": str(task.due_date) if task.due_date else None,
        }, ensure_ascii=False)


@tool
async def delete_task(task_id: str) -> str:
    """태스크를 삭제합니다. task_id는 필수."""
    try:
        task_uuid = uuid.UUID(task_id)
    except (ValueError, AttributeError):
        return json.dumps({"status": "error", "message": "Invalid task ID format"}, ensure_ascii=False)
    async with async_session() as session:
        task = await session.get(Task, task_uuid)
        if not task:
            return json.dumps({"status": "error", "message": "Task not found"}, ensure_ascii=False)
        await session.delete(task)
        await session.commit()
        return json.dumps({"status": "deleted", "task_id": task_id}, ensure_ascii=False)


def get_agent_tools() -> list:
    return [create_task, update_task, list_tasks, get_task, delete_task]
