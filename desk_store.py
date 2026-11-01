"""Durable demo ledger with expiring CAS lease; no transaction held during LLM calls."""
import json
from uuid import uuid4
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from db_connection import connect_db
from desk_core import initial_state


class DeskStore:
    def __init__(self,url):
        self.url=url

    async def init(self):
        async with connect_db(self.url) as c:
            await c.execute('''CREATE TABLE IF NOT EXISTS microtrader_agent_desk (
              id INTEGER PRIMARY KEY CHECK(id=1), state JSONB NOT NULL,
              lease TEXT, lease_until TIMESTAMPTZ, updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())''')
            await c.execute('''CREATE TABLE IF NOT EXISTS microtrader_desk_events (
              id BIGSERIAL PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              kind TEXT NOT NULL, payload JSONB NOT NULL)''')
            await c.execute('CREATE INDEX IF NOT EXISTS mt_desk_events_created ON microtrader_desk_events(created_at DESC)')
            # API access only; do not expose the journal through a hosted PostgREST API.
            await c.execute('ALTER TABLE microtrader_agent_desk ENABLE ROW LEVEL SECURITY')
            await c.execute('ALTER TABLE microtrader_desk_events ENABLE ROW LEVEL SECURITY')
            await c.execute('INSERT INTO microtrader_agent_desk(id,state) VALUES(1,%s) ON CONFLICT DO NOTHING',
                            (Jsonb(initial_state()),))

    async def read(self):
        async with connect_db(self.url,row_factory=dict_row) as c:
            row=await (await c.execute('SELECT state FROM microtrader_agent_desk WHERE id=1')).fetchone()
            return row['state']

    async def acquire(self):
        owner=str(uuid4())
        async with connect_db(self.url,row_factory=dict_row) as c:
            row=await (await c.execute('''UPDATE microtrader_agent_desk
              SET lease=%s,lease_until=NOW()+INTERVAL '180 seconds'
              WHERE id=1 AND (lease_until IS NULL OR lease_until<NOW()) RETURNING state''',(owner,))).fetchone()
            return (owner,row['state']) if row else (None,None)

    async def save(self,owner,state,events):
        async with connect_db(self.url) as c:
            cur=await c.execute('''UPDATE microtrader_agent_desk SET state=%s,lease=NULL,
              lease_until=NULL,updated_at=NOW() WHERE id=1 AND lease=%s AND lease_until>NOW()''',
                                (Jsonb(state),owner))
            if cur.rowcount!=1:
                raise RuntimeError('Desk lease expired: transaction discarded')
            for kind,payload in events:
                await c.execute('INSERT INTO microtrader_desk_events(kind,payload) VALUES(%s,%s)',
                                (kind,Jsonb(payload)))

    async def release(self,owner):
        async with connect_db(self.url) as c:
            await c.execute('UPDATE microtrader_agent_desk SET lease=NULL,lease_until=NULL WHERE id=1 AND lease=%s',(owner,))

    async def enable(self,enabled):
        # Do not race an in-flight decision. A pause invalidates its lease atomically.
        async with connect_db(self.url) as c:
            await c.execute('''UPDATE microtrader_agent_desk SET
              state=jsonb_set(state,'{enabled}',%s), lease=NULL,lease_until=NULL,updated_at=NOW() WHERE id=1''',
                            (Jsonb(bool(enabled)),))

    async def export(self):
        async with connect_db(self.url,row_factory=dict_row) as c:
            return await (await c.execute('SELECT id,created_at,kind,payload FROM microtrader_desk_events ORDER BY id')).fetchall()
