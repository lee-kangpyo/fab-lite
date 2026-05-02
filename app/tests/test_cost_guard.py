import pytest
import fakeredis.aioredis

from app.core.cost_guard import record_token_usage, check_token_budget


@pytest.mark.asyncio
async def test_토큰_사용량_기록():
    redis = fakeredis.aioredis.FakeRedis()
    await record_token_usage(redis, "session-1", 100)
    val = await redis.get("token_usage:session-1")
    assert val == b"100"
    await redis.aclose()


@pytest.mark.asyncio
async def test_토큰_사용량_누적():
    redis = fakeredis.aioredis.FakeRedis()
    await record_token_usage(redis, "session-1", 100)
    await record_token_usage(redis, "session-1", 50)
    val = await redis.get("token_usage:session-1")
    assert val == b"150"
    await redis.aclose()


@pytest.mark.asyncio
async def test_비용_가드_한도_이내():
    redis = fakeredis.aioredis.FakeRedis()
    await record_token_usage(redis, "session-1", 500)
    result = await check_token_budget(redis, "session-1", limit=1000)
    assert result is True
    await redis.aclose()


@pytest.mark.asyncio
async def test_비용_가드_한도_초과():
    redis = fakeredis.aioredis.FakeRedis()
    await record_token_usage(redis, "session-1", 1500)
    result = await check_token_budget(redis, "session-1", limit=1000)
    assert result is False
    await redis.aclose()


@pytest.mark.asyncio
async def test_비용_가드_사용량_없음():
    redis = fakeredis.aioredis.FakeRedis()
    result = await check_token_budget(redis, "new-session", limit=1000)
    assert result is True
    await redis.aclose()


@pytest.mark.asyncio
async def test_토큰_카운팅_콜백_추출():
    redis = fakeredis.aioredis.FakeRedis()
    from app.core.cost_guard import TokenCountingCallback

    cb = TokenCountingCallback(redis, "session-1")

    mock_response = type("LLMResult", (), {
        "llm_output": {"token_usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
    })()

    cb.on_llm_end(mock_response)
    await cb.await_pending()

    val = await redis.get("token_usage:session-1")
    assert val == b"30"
    await redis.aclose()


@pytest.mark.asyncio
async def test_세션_토큰_사용량_조회():
    redis = fakeredis.aioredis.FakeRedis()
    await record_token_usage(redis, "session-1", 100)
    await record_token_usage(redis, "session-1", 200)
    from app.core.cost_guard import get_session_token_usage
    usage = await get_session_token_usage(redis, "session-1")
    assert usage == 300
    await redis.aclose()
