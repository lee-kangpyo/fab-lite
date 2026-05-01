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