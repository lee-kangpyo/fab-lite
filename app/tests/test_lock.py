import time as time_mod
from unittest.mock import AsyncMock, MagicMock
import pytest
import fakeredis.aioredis

from app.core.lock import advisory_lock_key, DistributedLock, PGAdvisoryLock, atomic_state_transition, with_retry_backoff


@pytest.mark.asyncio
async def test_advisory_lock_key_deterministic():
    key1 = advisory_lock_key(0x01, "task", 42)
    key2 = advisory_lock_key(0x01, "task", 42)
    assert key1 == key2


@pytest.mark.asyncio
async def test_advisory_lock_key_different_namespace():
    key1 = advisory_lock_key(0x01, "task", 42)
    key2 = advisory_lock_key(0x02, "task", 42)
    assert key1 != key2


@pytest.mark.asyncio
async def test_advisory_lock_key_different_entity():
    key1 = advisory_lock_key(0x01, "task", 42)
    key2 = advisory_lock_key(0x01, "task", 99)
    assert key1 != key2


@pytest.mark.asyncio
async def test_advisory_lock_key_different_type():
    key1 = advisory_lock_key(0x01, "task", 42)
    key2 = advisory_lock_key(0x01, "schedule", 42)
    assert key1 != key2


@pytest.mark.asyncio
async def test_advisory_lock_key_is_positive_bigint():
    key = advisory_lock_key(0x01, "task", 42)
    assert key > 0
    assert key < 2**63


@pytest.mark.asyncio
async def test_distributed_lock_acquire_single_node():
    redis = fakeredis.aioredis.FakeRedis()
    lock = DistributedLock([redis], "test-lock", ttl=5000)
    acquired = await lock.acquire()
    assert acquired is True
    await lock.release()
    await redis.close()


@pytest.mark.asyncio
async def test_distributed_lock_acquire_quorum():
    r1 = fakeredis.aioredis.FakeRedis()
    r2 = fakeredis.aioredis.FakeRedis()
    r3 = fakeredis.aioredis.FakeRedis()
    lock = DistributedLock([r1, r2, r3], "test-lock", ttl=5000)
    acquired = await lock.acquire()
    assert acquired is True
    await lock.release()
    await r1.close()
    await r2.close()
    await r3.close()


@pytest.mark.asyncio
async def test_distributed_lock_prevent_double_acquire():
    redis = fakeredis.aioredis.FakeRedis()
    lock1 = DistributedLock([redis], "test-lock", ttl=5000)
    lock2 = DistributedLock([redis], "test-lock", ttl=5000)
    assert await lock1.acquire() is True
    assert await lock2.acquire() is False
    await lock1.release()
    await redis.close()


@pytest.mark.asyncio
async def test_distributed_lock_release_and_reacquire():
    redis = fakeredis.aioredis.FakeRedis()
    lock1 = DistributedLock([redis], "test-lock", ttl=5000)
    lock2 = DistributedLock([redis], "test-lock", ttl=5000)
    await lock1.acquire()
    await lock1.release()
    assert await lock2.acquire() is True
    await lock2.release()
    await redis.close()


@pytest.mark.asyncio
async def test_pg_advisory_lock_acquire_and_release():
    mock_session = AsyncMock()
    scalar_result = MagicMock()
    scalar_result.scalar.return_value = True
    mock_session.execute = AsyncMock(return_value=scalar_result)

    async with PGAdvisoryLock(mock_session, 12345):
        assert mock_session.execute.call_count == 1

    assert mock_session.execute.call_count == 2


@pytest.mark.asyncio
async def test_pg_advisory_lock_acquire_fails():
    mock_session = AsyncMock()
    scalar_result = MagicMock()
    scalar_result.scalar.return_value = False
    mock_session.execute = AsyncMock(return_value=scalar_result)

    with pytest.raises(RuntimeError, match="Failed to acquire"):
        async with PGAdvisoryLock(mock_session, 12345):
            pass


@pytest.mark.asyncio
async def test_pg_advisory_lock_releases_on_exception():
    mock_session = AsyncMock()
    scalar_result = MagicMock()
    scalar_result.scalar.return_value = True
    mock_session.execute = AsyncMock(return_value=scalar_result)

    with pytest.raises(ValueError):
        async with PGAdvisoryLock(mock_session, 12345):
            raise ValueError("test error")

    assert mock_session.execute.call_count == 2


@pytest.mark.asyncio
async def test_atomic_state_transition_success():
    redis = fakeredis.aioredis.FakeRedis()
    result = await atomic_state_transition(redis, "summarize", "READY", "RUNNING", ttl=300)
    assert result is True
    state = await redis.get("job:summarize:state")
    assert state == b"RUNNING"
    await redis.close()


@pytest.mark.asyncio
async def test_atomic_state_transition_already_running():
    redis = fakeredis.aioredis.FakeRedis()
    await redis.set("job:summarize:state", "RUNNING")
    result = await atomic_state_transition(redis, "summarize", "READY", "RUNNING", ttl=300)
    assert result is False
    await redis.close()


@pytest.mark.asyncio
async def test_with_retry_backoff_success_first_try():
    call_count = 0

    async def succeeds():
        nonlocal call_count
        call_count += 1
        return True

    result = await with_retry_backoff(succeeds, max_wait=1.0)
    assert result is True
    assert call_count == 1


@pytest.mark.asyncio
async def test_with_retry_backoff_success_after_retries():
    call_count = 0

    async def succeeds_on_third():
        nonlocal call_count
        call_count += 1
        return call_count >= 3

    result = await with_retry_backoff(succeeds_on_third, max_wait=2.0)
    assert result is True
    assert call_count >= 3


@pytest.mark.asyncio
async def test_with_retry_backoff_timeout():
    async def never_succeeds():
        return False

    start = time_mod.monotonic()
    result = await with_retry_backoff(never_succeeds, max_wait=0.5)
    elapsed = time_mod.monotonic() - start
    assert result is False
    assert elapsed >= 0.4