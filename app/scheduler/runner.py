from __future__ import annotations

import asyncio
from urllib.parse import urlparse

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.redis import RedisJobStore
from apscheduler.triggers.cron import CronTrigger
from redis.asyncio import from_url as async_from_url
from sqlalchemy import select

from app.core.lock import (
    DistributedLock,
    advisory_lock_key,
    atomic_state_transition,
    PGAdvisoryLock,
    with_retry_backoff,
)
from app.database import async_session
from app.models.task import Task


_scheduler_runner: "SchedulerRunner | None" = None


async def _scheduled_summarize_urgent_tasks(redis_url: str):
    """스케줄러에서 호출되는 모듈 레벨 요약 함수."""
    redis = await async_from_url(redis_url)
    heartbeat_task = None

    lock = DistributedLock([redis], "scheduler:urgent_summary")

    async def try_lock():
        return await lock.acquire()

    if not await with_retry_backoff(try_lock, max_wait=5.0):
        return

    try:
        transitioned = await atomic_state_transition(
            redis, "urgent_summary", "READY", "RUNNING", ttl=300
        )
        if not transitioned:
            return

        heartbeat_task = asyncio.create_task(
            _heartbeat(redis, "scheduler:urgent_summary:heartbeat")
        )

        async with async_session() as session:
            lock_key = advisory_lock_key(0x02, "schedule", 1)
            async with PGAdvisoryLock(session, lock_key):
                stmt = select(Task).where(
                    Task.priority == "urgent",
                    Task.status.in_(["todo", "in_progress"]),
                )
                result = await session.execute(stmt)
                tasks = result.scalars().all()

                lines = [f"긴급 태스크 요약 ({len(tasks)}건)"]
                for t in tasks:
                    lines.append(f"  - [{t.status}] {t.title}")

                summary = "\n".join(lines)
                await redis.set(
                    "scheduler:urgent_summary:last", summary, ex=86400
                )
    finally:
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        await redis.delete("scheduler:urgent_summary:state")
        await lock.release()
    await redis.close()


async def _heartbeat(redis_client, key: str, interval: float = 5.0):
    """Redis에 interval초 간격으로 생존 신호 갱신."""
    while True:
        await redis_client.set(key, "alive", ex=int(interval * 3))
        await asyncio.sleep(interval)


class SchedulerRunner:
    """APScheduler + Redis 잡 스토어 기반 스케줄러.

    매일 오전 9시 긴급 태스크 요약 생성.
    3중 락 (L1 Redlock + L2 PG Advisory + L3 상태 전이).
    Heartbeat 5초 간격 Redis 생존 신호.
    """

    def __init__(self, redis_url: str):
        self._redis_url = redis_url
        self._redis = None
        self._scheduler = None
        self._heartbeat_task: asyncio.Task | None = None

    async def start(self):
        if self._scheduler is not None and self._scheduler.running:
            return
        self._redis = async_from_url(self._redis_url)

        parsed = urlparse(self._redis_url)
        job_store = RedisJobStore(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            db=int((parsed.path or "/0").lstrip("/")) or 0,
            password=parsed.password,
        )
        self._scheduler = AsyncIOScheduler(
            jobstores={"default": job_store},
        )
        self._scheduler.add_job(
            _scheduled_summarize_urgent_tasks,
            "cron",
            hour=9,
            minute=0,
            id="summarize_urgent",
            replace_existing=True,
            args=[self._redis_url],
        )
        self._scheduler.start()

    async def stop(self):
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        if self._scheduler:
            self._scheduler.shutdown(wait=False)
        if self._redis:
            await self._redis.close()

    async def summarize_urgent_tasks(self):
        lock = DistributedLock([self._redis], "scheduler:urgent_summary")

        async def try_lock():
            return await lock.acquire()

        if not await with_retry_backoff(try_lock, max_wait=5.0):
            return

        try:
            transitioned = await atomic_state_transition(
                self._redis, "urgent_summary", "READY", "RUNNING", ttl=300
            )
            if not transitioned:
                return

            self._heartbeat_task = asyncio.create_task(
                self._heartbeat("scheduler:urgent_summary:heartbeat")
            )

            async with async_session() as session:
                lock_key = advisory_lock_key(0x02, "schedule", 1)
                async with PGAdvisoryLock(session, lock_key):
                        stmt = select(Task).where(
                            Task.priority == "urgent",
                            Task.status.in_(["todo", "in_progress"]),
                        )
                        result = await session.execute(stmt)
                        tasks = result.scalars().all()

                        lines = [f"긴급 태스크 요약 ({len(tasks)}건)"]
                        for t in tasks:
                            lines.append(f"  - [{t.status}] {t.title}")

                        summary = "\n".join(lines)
                        await self._redis.set(
                            "scheduler:urgent_summary:last", summary, ex=86400
                        )
        finally:
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
                self._heartbeat_task = None
            await self._redis.delete("scheduler:urgent_summary:state")
            await lock.release()

    async def _heartbeat(self, key: str, interval: float = 5.0):
        """Redis에 interval초 간격으로 생존 신호 갱신."""
        while True:
            await self._redis.set(key, "alive", ex=int(interval * 3))
            await asyncio.sleep(interval)

    async def get_last_summary(self) -> str | None:
        """마지막 요약 결과 조회."""
        if self._redis is None:
            return None
        data = await self._redis.get("scheduler:urgent_summary:last")
        return data.decode() if data else None