from __future__ import annotations

import asyncio

from redis.asyncio import Redis


async def record_token_usage(redis: Redis, session_id: str, tokens: int) -> None:
    await redis.incrby(f"token_usage:{session_id}", tokens)


async def get_session_token_usage(redis: Redis, session_id: str) -> int:
    val = await redis.get(f"token_usage:{session_id}")
    if val is None:
        return 0
    return int(val)


async def check_token_budget(redis: Redis, session_id: str, limit: int) -> bool:
    usage = await get_session_token_usage(redis, session_id)
    return usage <= limit


class TokenCountingCallback:
    """LangChain 콜백으로 LLM 응답에서 토큰 수를 추출해 Redis에 적재."""

    def __init__(self, redis: Redis, session_id: str):
        self._redis = redis
        self._session_id = session_id
        self._pending_tasks: list[asyncio.Task[None]] = []

    def on_llm_end(self, response: object) -> None:
        token_usage = getattr(response, "llm_output", None)
        if not token_usage:
            return
        usage = token_usage.get("token_usage", {})
        total = usage.get("total_tokens", 0)
        if total > 0:
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(
                    record_token_usage(self._redis, self._session_id, total)
                )
                self._pending_tasks.append(task)
            except RuntimeError:
                asyncio.run(
                    record_token_usage(self._redis, self._session_id, total)
                )

    async def await_pending(self) -> None:
        for task in self._pending_tasks:
            await task
        self._pending_tasks.clear()
