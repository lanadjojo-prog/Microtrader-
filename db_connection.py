from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator, Any

import psycopg


# The Supabase shared session pool for this project is capped at 15 backend
# sessions. MicroTrader has several workers plus dashboard reads that can burst
# at the same time. Keep enough headroom for Supabase/platform services and
# other clients instead of letting every coroutine open its own session.
_DB_CONNECTION_LIMIT = 6
_DB_SEMAPHORE = asyncio.Semaphore(_DB_CONNECTION_LIMIT)


@asynccontextmanager
async def connect_db(
    database_url: str,
    *,
    row_factory: Any = None,
    connect_timeout: int = 10,
) -> AsyncIterator[psycopg.AsyncConnection]:
    if not database_url:
        raise RuntimeError("database_url is required")

    kwargs = {"connect_timeout": int(connect_timeout)}
    if row_factory is not None:
        kwargs["row_factory"] = row_factory

    async with _DB_SEMAPHORE:
        async with await psycopg.AsyncConnection.connect(
            database_url,
            **kwargs,
        ) as conn:
            yield conn
