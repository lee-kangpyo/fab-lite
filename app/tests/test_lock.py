import time as time_mod
from unittest.mock import AsyncMock, MagicMock
import pytest
import fakeredis.aioredis

from app.core.lock import advisory_lock_key, DistributedLock, PGAdvisoryLock, atomic_state_transition, with_retry_backoff


@pytest.mark.asyncio
async def test_어드바이저리_락_키_결정론적_생성():
    key1 = advisory_lock_key(0x01, "task", 42)
    key2 = advisory_lock_key(0x01, "task", 42)
    assert key1 == key2


@pytest.mark.asyncio
async def test_어드바이저리_락_키_네임스페이스_차이():
    key1 = advisory_lock_key(0x01, "task", 42)
    key2 = advisory_lock_key(0x02, "task", 42)
    assert key1 != key2


@pytest.mark.asyncio
async def test_어드바이저리_락_키_엔티티_차이():
    key1 = advisory_lock_key(0x01, "task", 42)
    key2 = advisory_lock_key(0x01, "task", 99)
    assert key1 != key2


@pytest.mark.asyncio
async def test_어드바이저리_락_키_타입_차이():
    key1 = advisory_lock_key(0x01, "task", 42)
    key2 = advisory_lock_key(0x01, "schedule", 42)
    assert key1 != key2


@pytest.mark.asyncio
async def test_어드바이저리_락_키_양수_BigInt_여부():
    key = advisory_lock_key(0x01, "task", 42)
    assert key > 0
    assert key < 2**63


@pytest.mark.asyncio
async def test_분산락_단일_노드_획득():
    redis = fakeredis.aioredis.FakeRedis()
    lock = DistributedLock([redis], "test-lock", ttl=5000)
    acquired = await lock.acquire()
    assert acquired is True
    await lock.release()
    await redis.aclose()


@pytest.mark.asyncio
async def test_분산락_쿼럼_획득():
    r1 = fakeredis.aioredis.FakeRedis()
    r2 = fakeredis.aioredis.FakeRedis()
    r3 = fakeredis.aioredis.FakeRedis()
    lock = DistributedLock([r1, r2, r3], "test-lock", ttl=5000)
    acquired = await lock.acquire()
    assert acquired is True
    await lock.release()
    await r1.aclose()
    await r2.aclose()
    await r3.aclose()


@pytest.mark.asyncio
async def test_분산락_중복_획득_방지():
    redis = fakeredis.aioredis.FakeRedis()
    lock1 = DistributedLock([redis], "test-lock", ttl=5000)
    lock2 = DistributedLock([redis], "test-lock", ttl=5000)
    assert await lock1.acquire() is True
    assert await lock2.acquire() is False
    await lock1.release()
    await redis.aclose()


@pytest.mark.asyncio
async def test_분산락_해제_후_재획득():
    redis = fakeredis.aioredis.FakeRedis()
    lock1 = DistributedLock([redis], "test-lock", ttl=5000)
    lock2 = DistributedLock([redis], "test-lock", ttl=5000)
    await lock1.acquire()
    await lock1.release()
    assert await lock2.acquire() is True
    await lock2.release()
    await redis.aclose()


@pytest.mark.asyncio
async def test_PG_어드바이저리_락_획득_및_해제():
    mock_session = AsyncMock()
    scalar_result = MagicMock()
    scalar_result.scalar.return_value = True
    mock_session.execute = AsyncMock(return_value=scalar_result)

    async with PGAdvisoryLock(mock_session, 12345):
        assert mock_session.execute.call_count == 1

    assert mock_session.execute.call_count == 2


@pytest.mark.asyncio
async def test_PG_어드바이저리_락_획득_실패():
    mock_session = AsyncMock()
    scalar_result = MagicMock()
    scalar_result.scalar.return_value = False
    mock_session.execute = AsyncMock(return_value=scalar_result)

    with pytest.raises(RuntimeError, match="Failed to acquire"):
        async with PGAdvisoryLock(mock_session, 12345):
            pass


@pytest.mark.asyncio
async def test_PG_어드바이저리_락_예외_발생_시_자동_해제():
    mock_session = AsyncMock()
    scalar_result = MagicMock()
    scalar_result.scalar.return_value = True
    mock_session.execute = AsyncMock(return_value=scalar_result)

    with pytest.raises(ValueError):
        async with PGAdvisoryLock(mock_session, 12345):
            raise ValueError("test error")

    assert mock_session.execute.call_count == 2


@pytest.mark.asyncio
async def test_원자적_상태_전이_성공():
    redis = fakeredis.aioredis.FakeRedis()
    result = await atomic_state_transition(redis, "summarize", "READY", "RUNNING", ttl=300)
    assert result is True
    state = await redis.get("job:summarize:state")
    assert state == b"RUNNING"
    await redis.aclose()


@pytest.mark.asyncio
async def test_원자적_상태_전이_이미_실행_중():
    redis = fakeredis.aioredis.FakeRedis()
    await redis.set("job:summarize:state", "RUNNING")
    result = await atomic_state_transition(redis, "summarize", "READY", "RUNNING", ttl=300)
    assert result is False
    await redis.aclose()


@pytest.mark.asyncio
async def test_재시도_백오프_첫_시도_성공():
    call_count = 0

    async def succeeds():
        nonlocal call_count
        call_count += 1
        return True

    result = await with_retry_backoff(succeeds, max_wait=1.0)
    assert result is True
    assert call_count == 1


@pytest.mark.asyncio
async def test_재시도_백오프_여러_번_시도_후_성공():
    call_count = 0

    async def succeeds_on_third():
        nonlocal call_count
        call_count += 1
        return call_count >= 3

    result = await with_retry_backoff(succeeds_on_third, max_wait=2.0)
    assert result is True
    assert call_count >= 3


@pytest.mark.asyncio
async def test_재시도_백오프_타임아웃_실패():
    async def never_succeeds():
        return False

    start = time_mod.monotonic()
    result = await with_retry_backoff(never_succeeds, max_wait=0.5)
    elapsed = time_mod.monotonic() - start
    assert result is False
    assert elapsed >= 0.4