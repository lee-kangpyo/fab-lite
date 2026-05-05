from __future__ import annotations

import asyncio

from redis.asyncio import Redis


async def record_token_usage(redis: Redis, session_id: str, tokens: int, reset_period: int = 86400) -> None:
    key = f"token_usage:{session_id}"
    await redis.incrby(key, tokens)
    # 처음 생성된 키일 경우에만 TTL 설정 (nx=True)
    await redis.expire(key, reset_period, nx=True)


async def get_session_token_usage(redis: Redis, session_id: str) -> int:
    val = await redis.get(f"token_usage:{session_id}")
    if val is None:
        return 0
    return int(val)


async def check_token_budget(redis: Redis, session_id: str, limit: int) -> bool:
    usage = await get_session_token_usage(redis, session_id)
    return usage <= limit


from langchain_core.callbacks import BaseCallbackHandler

class TokenCountingCallback(BaseCallbackHandler):
    """LangChain 콜백으로 LLM 응답에서 토큰 수를 추출해 Redis에 적재."""
    
    # 최신 langchain-core 호환성을 위해 추가
    run_inline: bool = True

    def __init__(self, redis: Redis, session_id: str, reset_period: int = 86400) -> None:
        super().__init__()
        self._redis = redis
        self._session_id = session_id
        self._reset_period = reset_period
        self._pending_tasks: list[asyncio.Task[None]] = []

    def on_llm_end(self, response: object, **kwargs: object) -> None:
        token_usage = getattr(response, "llm_output", None)
        if not token_usage:
            return
        usage = token_usage.get("token_usage", {})
        total = usage.get("total_tokens", 0)
        if total > 0:
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(
                    record_token_usage(self._redis, self._session_id, total, self._reset_period)
                )
                self._pending_tasks.append(task)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    record_token_usage(self._redis, self._session_id, total, self._reset_period)
                )
                loop.close()

    async def await_pending(self) -> None:
        for task in self._pending_tasks:
            await task
        self._pending_tasks.clear()
