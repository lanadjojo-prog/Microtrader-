from __future__ import annotations

from typing import Dict
import psycopg
from db_connection import connect_db
from psycopg.rows import dict_row


class CTraderTokenStore:
    """Persist cTrader OAuth tokens so Render restarts do not lose authorization."""

    def __init__(self, database_url: str):
        self.database_url = (database_url or "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.database_url)

    async def init(self) -> None:
        if not self.enabled:
            return
        async with connect_db(self.database_url) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS microtrader_ctrader_tokens (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    access_token TEXT NOT NULL DEFAULT '',
                    refresh_token TEXT NOT NULL DEFAULT '',
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            await conn.execute("""
                INSERT INTO microtrader_ctrader_tokens (id)
                VALUES (1)
                ON CONFLICT (id) DO NOTHING
            """)
            await conn.commit()

    async def save(self, access_token: str, refresh_token: str) -> None:
        if not self.enabled:
            return
        async with connect_db(self.database_url) as conn:
            await conn.execute("""
                UPDATE microtrader_ctrader_tokens
                SET access_token=%s, refresh_token=%s, updated_at=NOW()
                WHERE id=1
            """, (str(access_token or ""), str(refresh_token or "")))
            await conn.commit()

    async def load(self) -> Dict[str, str]:
        if not self.enabled:
            return {"access_token": "", "refresh_token": ""}
        async with connect_db(
            self.database_url, row_factory=dict_row
        ) as conn:
            cur = await conn.execute("""
                SELECT access_token, refresh_token
                FROM microtrader_ctrader_tokens WHERE id=1
            """)
            row = await cur.fetchone()
            return dict(row) if row else {"access_token": "", "refresh_token": ""}
