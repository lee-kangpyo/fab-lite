import pytest

from app.core.lock import advisory_lock_key


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