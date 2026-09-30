from __future__ import annotations

import json
from typing import Any, Dict, List

import psycopg
from psycopg.rows import dict_row


class PaperTradingStore:
    def __init__(self, database_url: str):
        self.database_url = (database_url or "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.database_url)

    async def init(self) -> None:
        if not self.enabled:
            return
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS microtrader_paper_strategies (
                    paper_id TEXT PRIMARY KEY,
                    promoted_run_id TEXT,
                    strategy TEXT NOT NULL,
                    params JSONB NOT NULL,
                    pairs JSONB NOT NULL,
                    timeframe_min INTEGER NOT NULL,
                    start_balance DOUBLE PRECISION NOT NULL,
                    balance DOUBLE PRECISION NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_bar_times JSONB NOT NULL DEFAULT '{}'::jsonb,
                    last_cycle_at TIMESTAMPTZ,
                    last_error TEXT
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS microtrader_paper_positions (
                    paper_id TEXT NOT NULL REFERENCES microtrader_paper_strategies(paper_id)
                        ON DELETE CASCADE,
                    pair TEXT NOT NULL,
                    direction INTEGER NOT NULL,
                    entry_time TIMESTAMPTZ NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    risk_distance DOUBLE PRECISION NOT NULL,
                    stop_price DOUBLE PRECISION NOT NULL,
                    target_price DOUBLE PRECISION NOT NULL,
                    bars_held INTEGER NOT NULL DEFAULT 0,
                    risk_eur DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY (paper_id, pair)
                )
            """)
            await conn.execute("""
                ALTER TABLE microtrader_paper_positions
                ADD COLUMN IF NOT EXISTS current_price DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS unrealized_r DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS unrealized_pnl DOUBLE PRECISION,
                ADD COLUMN IF NOT EXISTS last_mark_at TIMESTAMPTZ
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS microtrader_paper_trades (
                    id BIGSERIAL PRIMARY KEY,
                    paper_id TEXT NOT NULL REFERENCES microtrader_paper_strategies(paper_id)
                        ON DELETE CASCADE,
                    pair TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_time TIMESTAMPTZ NOT NULL,
                    exit_time TIMESTAMPTZ NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    exit_price DOUBLE PRECISION NOT NULL,
                    exit_reason TEXT NOT NULL,
                    risk_eur DOUBLE PRECISION NOT NULL,
                    r_multiple DOUBLE PRECISION NOT NULL,
                    pnl DOUBLE PRECISION NOT NULL,
                    balance_before DOUBLE PRECISION NOT NULL,
                    balance_after DOUBLE PRECISION NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_mt_paper_trades_strategy_time
                ON microtrader_paper_trades(paper_id, exit_time DESC)
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS microtrader_paper_daily (
                    paper_id TEXT NOT NULL REFERENCES microtrader_paper_strategies(paper_id)
                        ON DELETE CASCADE,
                    trade_date DATE NOT NULL,
                    start_balance DOUBLE PRECISION NOT NULL,
                    realized_pnl DOUBLE PRECISION NOT NULL DEFAULT 0,
                    end_balance DOUBLE PRECISION NOT NULL,
                    trade_count INTEGER NOT NULL DEFAULT 0,
                    wins INTEGER NOT NULL DEFAULT 0,
                    losses INTEGER NOT NULL DEFAULT 0,
                    open_positions INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (paper_id, trade_date)
                )
            """)
            await conn.commit()

    async def ensure_strategy(
        self,
        *,
        paper_id: str,
        promoted_run_id: str,
        strategy: str,
        params: dict,
        pairs: list[str],
        timeframe_min: int,
        start_balance: float,
    ) -> None:
        if not self.enabled:
            return
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            await conn.execute(
                """
                INSERT INTO microtrader_paper_strategies (
                    paper_id, promoted_run_id, strategy, params, pairs,
                    timeframe_min, start_balance, balance, status
                ) VALUES (%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s,%s,'active')
                ON CONFLICT (paper_id) DO UPDATE SET
                    promoted_run_id=EXCLUDED.promoted_run_id
                """,
                (
                    paper_id, promoted_run_id, strategy,
                    json.dumps(params), json.dumps(pairs),
                    int(timeframe_min), float(start_balance), float(start_balance),
                ),
            )
            await conn.commit()

    async def list_strategies(self) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        async with await psycopg.AsyncConnection.connect(
            self.database_url, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                """
                SELECT paper_id, promoted_run_id, strategy, params, pairs,
                       timeframe_min, start_balance, balance, status,
                       started_at, last_bar_times, last_cycle_at, last_error
                FROM microtrader_paper_strategies
                ORDER BY started_at ASC
                """
            )
            return [dict(row) for row in await cur.fetchall()]

    async def set_cursor(self, paper_id: str, pair: str, bar_time: str) -> None:
        if not self.enabled:
            return
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            cur = await conn.execute(
                "SELECT last_bar_times FROM microtrader_paper_strategies WHERE paper_id=%s",
                (paper_id,),
            )
            row = await cur.fetchone()
            cursors = dict((row[0] if row else {}) or {})
            cursors[pair] = bar_time
            await conn.execute(
                """
                UPDATE microtrader_paper_strategies
                SET last_bar_times=%s::jsonb, last_cycle_at=NOW(), last_error=NULL
                WHERE paper_id=%s
                """,
                (json.dumps(cursors), paper_id),
            )
            await conn.commit()

    async def set_status(self, paper_id: str, status: str) -> None:
        if not self.enabled:
            return
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            await conn.execute(
                """
                UPDATE microtrader_paper_strategies
                SET status=%s, last_cycle_at=NOW()
                WHERE paper_id=%s
                """,
                (str(status), paper_id),
            )
            if str(status) == "policy_rejected":
                await conn.execute(
                    "DELETE FROM microtrader_paper_positions WHERE paper_id=%s",
                    (paper_id,),
                )
            await conn.commit()

    async def set_error(self, paper_id: str, error: str) -> None:
        if not self.enabled:
            return
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            await conn.execute(
                """
                UPDATE microtrader_paper_strategies
                SET last_error=%s, last_cycle_at=NOW()
                WHERE paper_id=%s
                """,
                (str(error), paper_id),
            )
            await conn.commit()

    async def list_positions(self, paper_id: str) -> List[Dict[str, Any]]:
        if not self.enabled:
            return []
        async with await psycopg.AsyncConnection.connect(
            self.database_url, row_factory=dict_row
        ) as conn:
            cur = await conn.execute(
                """
                SELECT paper_id, pair, direction, entry_time, entry_price,
                       risk_distance, stop_price, target_price, bars_held, risk_eur,
                       current_price, unrealized_r, unrealized_pnl, last_mark_at
                FROM microtrader_paper_positions
                WHERE paper_id=%s
                ORDER BY pair
                """,
                (paper_id,),
            )
            return [dict(row) for row in await cur.fetchall()]

    async def upsert_position(self, paper_id: str, pair: str, position: dict) -> None:
        if not self.enabled:
            return
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            await conn.execute(
                """
                INSERT INTO microtrader_paper_positions (
                    paper_id, pair, direction, entry_time, entry_price,
                    risk_distance, stop_price, target_price, bars_held, risk_eur,
                    current_price, unrealized_r, unrealized_pnl, last_mark_at
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (paper_id, pair) DO UPDATE SET
                    direction=EXCLUDED.direction,
                    entry_time=EXCLUDED.entry_time,
                    entry_price=EXCLUDED.entry_price,
                    risk_distance=EXCLUDED.risk_distance,
                    stop_price=EXCLUDED.stop_price,
                    target_price=EXCLUDED.target_price,
                    bars_held=EXCLUDED.bars_held,
                    risk_eur=EXCLUDED.risk_eur,
                    current_price=EXCLUDED.current_price,
                    unrealized_r=EXCLUDED.unrealized_r,
                    unrealized_pnl=EXCLUDED.unrealized_pnl,
                    last_mark_at=EXCLUDED.last_mark_at
                """,
                (
                    paper_id, pair, int(position["direction"]),
                    position["entry_time"], float(position["entry_price"]),
                    float(position["risk_distance"]), float(position["stop_price"]),
                    float(position["target_price"]), int(position.get("bars_held", 0)),
                    float(position["risk_eur"]),
                    float(position.get("current_price") or position["entry_price"]),
                    float(position.get("unrealized_r") or 0.0),
                    float(position.get("unrealized_pnl") or 0.0),
                    position.get("last_mark_at") or position["entry_time"],
                ),
            )
            await conn.commit()

    async def mark_position(
        self,
        paper_id: str,
        pair: str,
        *,
        current_price: float,
        unrealized_r: float,
        unrealized_pnl: float,
        mark_time: str,
    ) -> None:
        if not self.enabled:
            return
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            await conn.execute(
                """
                UPDATE microtrader_paper_positions
                SET current_price=%s,
                    unrealized_r=%s,
                    unrealized_pnl=%s,
                    last_mark_at=%s
                WHERE paper_id=%s AND pair=%s
                """,
                (
                    float(current_price), float(unrealized_r),
                    float(unrealized_pnl), mark_time, paper_id, pair,
                ),
            )
            await conn.commit()

    async def delete_position(self, paper_id: str, pair: str) -> None:
        if not self.enabled:
            return
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            await conn.execute(
                "DELETE FROM microtrader_paper_positions WHERE paper_id=%s AND pair=%s",
                (paper_id, pair),
            )
            await conn.commit()

    async def record_trade(
        self,
        *,
        paper_id: str,
        pair: str,
        side: str,
        entry_time: str,
        exit_time: str,
        entry_price: float,
        exit_price: float,
        exit_reason: str,
        risk_eur: float,
        r_multiple: float,
        pnl: float,
    ) -> float:
        if not self.enabled:
            return 0.0
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            cur = await conn.execute(
                "SELECT balance FROM microtrader_paper_strategies WHERE paper_id=%s FOR UPDATE",
                (paper_id,),
            )
            row = await cur.fetchone()
            if not row:
                return 0.0
            before = float(row[0])
            after = round(before + float(pnl), 6)
            day = str(exit_time)[:10]
            await conn.execute(
                """
                INSERT INTO microtrader_paper_daily (
                    paper_id, trade_date, start_balance, end_balance
                ) VALUES (%s,%s,%s,%s)
                ON CONFLICT (paper_id, trade_date) DO NOTHING
                """,
                (paper_id, day, before, before),
            )
            await conn.execute(
                """
                INSERT INTO microtrader_paper_trades (
                    paper_id, pair, side, entry_time, exit_time,
                    entry_price, exit_price, exit_reason, risk_eur,
                    r_multiple, pnl, balance_before, balance_after
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    paper_id, pair, side, entry_time, exit_time,
                    float(entry_price), float(exit_price), exit_reason,
                    float(risk_eur), float(r_multiple), float(pnl), before, after,
                ),
            )
            await conn.execute(
                """
                UPDATE microtrader_paper_daily
                SET realized_pnl=realized_pnl+%s,
                    end_balance=%s,
                    trade_count=trade_count+1,
                    wins=wins+%s,
                    losses=losses+%s,
                    updated_at=NOW()
                WHERE paper_id=%s AND trade_date=%s
                """,
                (
                    float(pnl), after, 1 if pnl > 0 else 0, 1 if pnl < 0 else 0,
                    paper_id, day,
                ),
            )
            await conn.execute(
                """
                UPDATE microtrader_paper_strategies
                SET balance=%s,
                    status=CASE WHEN %s <= 0 THEN 'ruined' ELSE status END,
                    last_cycle_at=NOW()
                WHERE paper_id=%s
                """,
                (after, after, paper_id),
            )
            await conn.commit()
            return after

    async def touch_daily(self, paper_id: str, open_positions: int) -> None:
        if not self.enabled:
            return
        async with await psycopg.AsyncConnection.connect(self.database_url) as conn:
            cur = await conn.execute(
                "SELECT balance FROM microtrader_paper_strategies WHERE paper_id=%s",
                (paper_id,),
            )
            row = await cur.fetchone()
            if not row:
                return
            balance = float(row[0])
            await conn.execute(
                """
                INSERT INTO microtrader_paper_daily (
                    paper_id, trade_date, start_balance, end_balance, open_positions
                ) VALUES (%s, CURRENT_DATE, %s, %s, %s)
                ON CONFLICT (paper_id, trade_date) DO UPDATE SET
                    end_balance=EXCLUDED.end_balance,
                    open_positions=EXCLUDED.open_positions,
                    updated_at=NOW()
                """,
                (paper_id, balance, balance, int(open_positions)),
            )
            await conn.commit()

    async def dashboard(self, daily_limit: int = 60, trade_limit: int = 150) -> Dict[str, Any]:
        if not self.enabled:
            return {"strategies": [], "daily": [], "trades": [], "positions": []}
        async with await psycopg.AsyncConnection.connect(
            self.database_url, row_factory=dict_row
        ) as conn:
            strategies = [
                dict(row) for row in await (
                    await conn.execute(
                        """
                        SELECT s.*,
                               COALESCE(t.trades,0) AS trades,
                               COALESCE(t.wins,0) AS wins,
                               COALESCE(t.losses,0) AS losses,
                               COALESCE(t.net_pnl,0) AS net_pnl,
                               COALESCE(p.open_positions,0) AS open_positions
                        FROM microtrader_paper_strategies s
                        LEFT JOIN (
                            SELECT paper_id, COUNT(*) AS trades,
                                   SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) AS wins,
                                   SUM(CASE WHEN pnl<0 THEN 1 ELSE 0 END) AS losses,
                                   SUM(pnl) AS net_pnl
                            FROM microtrader_paper_trades GROUP BY paper_id
                        ) t ON t.paper_id=s.paper_id
                        LEFT JOIN (
                            SELECT paper_id, COUNT(*) AS open_positions
                            FROM microtrader_paper_positions GROUP BY paper_id
                        ) p ON p.paper_id=s.paper_id
                        WHERE s.status <> 'policy_rejected'
                        ORDER BY s.started_at ASC
                        """
                    )
                ).fetchall()
            ]
            daily = [
                dict(row) for row in await (
                    await conn.execute(
                        """
                        SELECT d.paper_id, d.trade_date, d.start_balance, d.realized_pnl,
                               d.end_balance, d.trade_count, d.wins, d.losses, d.open_positions,
                               d.updated_at, s.strategy, s.params
                        FROM microtrader_paper_daily d
                        JOIN microtrader_paper_strategies s ON s.paper_id=d.paper_id
                        WHERE s.status <> 'policy_rejected'
                        ORDER BY d.trade_date DESC, d.paper_id
                        LIMIT %s
                        """,
                        (daily_limit,),
                    )
                ).fetchall()
            ]
            trades = [
                dict(row) for row in await (
                    await conn.execute(
                        """
                        SELECT t.id, t.paper_id, t.pair, t.side, t.entry_time, t.exit_time,
                               t.entry_price, t.exit_price, t.exit_reason, t.risk_eur,
                               t.r_multiple, t.pnl, t.balance_before, t.balance_after,
                               s.strategy, s.params
                        FROM microtrader_paper_trades t
                        JOIN microtrader_paper_strategies s ON s.paper_id=t.paper_id
                        WHERE s.status <> 'policy_rejected'
                        ORDER BY t.exit_time DESC
                        LIMIT %s
                        """,
                        (trade_limit,),
                    )
                ).fetchall()
            ]
            positions = [
                dict(row) for row in await (
                    await conn.execute(
                        """
                        SELECT p.paper_id, p.pair, p.direction, p.entry_time, p.entry_price,
                               p.stop_price, p.target_price, p.bars_held, p.risk_eur,
                               p.current_price, p.unrealized_r, p.unrealized_pnl,
                               p.last_mark_at, s.strategy, s.params
                        FROM microtrader_paper_positions p
                        JOIN microtrader_paper_strategies s ON s.paper_id=p.paper_id
                        WHERE s.status <> 'policy_rejected'
                        ORDER BY p.paper_id, p.pair
                        """
                    )
                ).fetchall()
            ]
            return {
                "strategies": strategies,
                "daily": daily,
                "trades": trades,
                "positions": positions,
            }
