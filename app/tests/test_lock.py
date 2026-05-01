import pytest
import fakeredis.aioredis

from app.core.lock import advisory_lock_key, DistributedLock


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