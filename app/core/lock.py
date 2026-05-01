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