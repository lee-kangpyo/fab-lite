import asyncio

import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch
from apscheduler.schedulers.asyncio import AsyncIOScheduler

import fakeredis.aioredis

from app.scheduler.runner import SchedulerRunner


@pytest.mark.asyncio
async def test_scheduler_runner_init():
    runner = SchedulerRunner("redis://localhost:6379/0")
    assert runner._redis_url == "redis://localhost:6379/0"
    assert runner._scheduler is None


@pytest.mark.asyncio
async def test_scheduler_runner_start_stop():
    runner = SchedulerRunner("redis://localhost:6379/0")
    mock_async_redis = AsyncMock()
    mock_async_redis.close = AsyncMock()
    mock_sync_redis = MagicMock()

    with patch("app.scheduler.runner.async_from_url", return_value=mock_async_redis), \
         patch("app.scheduler.runner.SyncRedis", return_value=mock_sync_redis), \
         patch("app.scheduler.runner.RedisJobStore"):
        runner._redis = mock_async_redis
        scheduler = AsyncIOScheduler()
        runner._scheduler = scheduler
        scheduler.start()
        assert runner._scheduler is not None
        assert runner._scheduler.running

        await runner.stop()
        await asyncio.sleep(0.1)
        assert not runner._scheduler.running


@pytest.mark.asyncio
async def test_summarize_urgent_tasks(client):
    await client.post("/api/tasks", json={"title": "긴급1", "priority": "urgent"})
    await client.post("/api/tasks", json={"title": "긴급2", "priority": "urgent"})
    await client.post("/api/tasks", json={"title": "보통", "priority": "medium"})

    runner = SchedulerRunner("redis://localhost:6379/0")
    mock_redis = fakeredis.aioredis.FakeRedis()
    runner._redis = mock_redis

    with patch("app.scheduler.runner.PGAdvisoryLock") as mock_pg_lock:
        @asynccontextmanager
        async def mock_lock(*args, **kwargs):
            yield

        mock_pg_lock.side_effect = mock_lock
        await runner.summarize_urgent_tasks()

        summary_raw = await mock_redis.get("scheduler:urgent_summary:last")
        assert summary_raw is not None
        summary = summary_raw.decode()
        assert "긴급1" in summary
        assert "긴급2" in summary
        assert "보통" not in summary

    await mock_redis.close()


@pytest.mark.asyncio
async def test_heartbeat_updates_redis():
    mock_redis = fakeredis.aioredis.FakeRedis()
    runner = SchedulerRunner("redis://localhost:6379/0")
    runner._redis = mock_redis

    task = asyncio.create_task(runner._heartbeat("test:heartbeat", interval=0.5))
    await asyncio.sleep(0.25)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    val = await mock_redis.get("test:heartbeat")
    assert val == b"alive"
    await mock_redis.close()


@pytest.mark.asyncio
async def test_get_last_summary():
    mock_redis = fakeredis.aioredis.FakeRedis()
    await mock_redis.set("scheduler:urgent_summary:last", "테스트 요약")

    runner = SchedulerRunner("redis://localhost:6379/0")
    runner._redis = mock_redis

    summary = await runner.get_last_summary()
    assert summary == "테스트 요약"
    await mock_redis.close()


@pytest.mark.asyncio
async def test_get_last_summary_none():
    mock_redis = fakeredis.aioredis.FakeRedis()
    runner = SchedulerRunner("redis://localhost:6379/0")
    runner._redis = mock_redis

    summary = await runner.get_last_summary()
    assert summary is None
    await mock_redis.close()