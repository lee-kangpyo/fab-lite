from __future__ import annotations

import asyncio
import time
import uuid
import zlib
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


def advisory_lock_key(namespace: int, entity_type: str, entity_id: int) -> int:
    key_str = f"{entity_type}:{entity_id}"
    entity_hash = zlib.adler32(key_str.encode()) & 0xFFFFFFFF
    return (namespace << 32) | entity_hash


class DistributedLock:
    """Redlock 알고리즘 기반 분산락 (L1).

    다중 Redis 노드에서 과반수 락 획득 시 성공.
    TTL 만료로 교착 상태 방지.
    """

    def __init__(self, redis_clients: list, lock_name: str, ttl: int = 10_000):
        self.redis_clients = redis_clients
        self.lock_name = f"lock:{lock_name}"
        self.ttl = ttl
        self.value = str(uuid.uuid4())
        self.quorum = len(redis_clients) // 2 + 1
        self._acquired_clients: list = []

    async def acquire(self) -> bool:
        start = time.monotonic()
        acquired = 0
        self._acquired_clients = []
        for client in self.redis_clients:
            try:
                result = await client.set(
                    self.lock_name, self.value, nx=True, px=self.ttl
                )
                if result:
                    acquired += 1
                    self._acquired_clients.append(client)
            except Exception:
                continue
        elapsed_ms = (time.monotonic() - start) * 1000
        if acquired >= self.quorum and elapsed_ms < self.ttl:
            return True
        await self.release()
        return False

    async def release(self):
        lua_script = """
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("DEL", KEYS[1])
        else
            return 0
        end
        """
        for client in self._acquired_clients:
            try:
                await client.eval(lua_script, 1, self.lock_name, self.value)
            except Exception:
                try:
                    current = await client.get(self.lock_name)
                    if current is not None:
                        val = current.decode() if isinstance(current, bytes) else current
                        if val == self.value:
                            await client.delete(self.lock_name)
                except Exception:
                    continue
        self._acquired_clients = []


@asynccontextmanager
async def PGAdvisoryLock(session: AsyncSession, key: int):
    result = await session.execute(
        text("SELECT pg_try_advisory_lock(:key)"), {"key": key}
    )
    acquired = result.scalar()
    if not acquired:
        raise RuntimeError(f"Failed to acquire advisory lock for key {key}")
    try:
        yield
    finally:
        await session.execute(
            text("SELECT pg_advisory_unlock(:key)"), {"key": key}
        )


async def atomic_state_transition(redis_client, job_name: str, from_state: str, to_state: str, ttl: int = 300) -> bool:
    """Redis SET NX 기반 원자적 상태 전이 (L3).

    키가 존재하지 않을 때만 to_state로 설정.
    성공 시 True, 이미 실행 중이면 False.
    """
    key = f"job:{job_name}:state"
    current = await redis_client.get(key)
    if current is not None:
        val = current.decode() if isinstance(current, bytes) else current
        if val != from_state:
            return False
    result = await redis_client.set(key, to_state, nx=True, ex=ttl)
    return result is not None


async def with_retry_backoff(coro_func, max_wait: float = 5.0) -> bool:
    delay = 0.1
    elapsed = 0.0
    while elapsed < max_wait:
        result = await coro_func()
        if result:
            return True
        await asyncio.sleep(delay)
        elapsed += delay
        delay = min(delay * 2, 1.0)
    return False