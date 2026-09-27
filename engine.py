from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, Optional, List

from alpaca_client import AlpacaClient, AlpacaError
from config import Settings
from strategy import moving_average_momentum

log = logging.getLogger("microtrader")


@dataclass
class EngineState:
    running: bool = False
    killed: bool = False
    kill_reason: Optional[str] = None
    last_cycle_at: Optional[str] = None
    last_error: Optional[str] = None
    submitted_orders: int = 0
    cycles: int = 0


class TradingEngine:
    def __init__(self, settings: Settings, client: AlpacaClient):
        self.settings = settings
        self.client = client
        self.state = EngineState()
        self._task: Optional[asyncio.Task] = None
        self._last_trade_ts: Dict[str, float] = {}
        self._last_bar_action: Dict[str, str] = {}
        self.execution_log: List[dict] = []

    def public_state(self) -> dict:
        return asdict(self.state)

    def executions(self) -> List[dict]:
        return list(reversed(self.execution_log[-500:]))

    async def start(self):
        if self.state.running:
            return
        self.state.running = True
        self.state.killed = False
        self.state.kill_reason = None
        self.state.last_error = None
        self._task = asyncio.create_task(self._loop(), name="microtrader-engine")

    async def stop(self):
        self.state.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def kill(self, reason: str):
        self.state.killed = True
        self.state.kill_reason = reason
        self.state.running = False
        log.error("KILL SWITCH: %s", reason)
        if self.settings.kill_close_positions and self.settings.can_trade:
            try:
                await self.client.cancel_all_orders()
                await self.client.close_all_positions()
            except Exception as exc:
                log.exception("Failed to flatten account during kill switch: %s", exc)
        elif self.settings.kill_close_positions:
            log.warning("KILL SWITCH flatten blocked by live safety lock")

    async def _loop(self):
        while self.state.running:
            try:
                await self.cycle()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state.last_error = str(exc)
                log.exception("cycle failed")
            await asyncio.sleep(self.settings.poll_seconds)

    async def cycle(self):
        self.state.cycles += 1
        self.state.last_cycle_at = datetime.now(timezone.utc).isoformat()

        if not self.settings.api_key or not self.settings.api_secret:
            self.state.last_error = "Missing ALPACA_API_KEY / ALPACA_API_SECRET"
            return

        await self._reconcile_executions()

        clock = await self.client.clock()
        if not clock.get("is_open"):
            return

        account = await self.client.account()
        equity = float(account.get("equity", 0) or 0)
        last_equity = float(account.get("last_equity", equity) or equity)
        day_pnl = equity - last_equity
        if day_pnl <= -abs(self.settings.max_daily_loss):
            await self.kill(f"daily P&L {day_pnl:.2f} <= -{self.settings.max_daily_loss:.2f}")
            return

        orders = await self.client.orders_today()
        trade_count = len([o for o in orders if o.get("side") in {"buy", "sell"}])
        if trade_count >= self.settings.max_trades_per_day:
            self.state.last_error = f"Daily trade cap reached: {trade_count}"
            return

        positions = {p["symbol"]: p for p in await self.client.positions()}
        bars_map = await self.client.latest_bars(self.settings.symbols, self.settings.bars_lookback)

        # Exits first.
        for symbol, position in list(positions.items()):
            if symbol not in self.settings.symbols:
                continue
            bars = bars_map.get(symbol, [])
            if not bars:
                continue
            signal = moving_average_momentum(
                bars,
                self.settings.fast_window,
                self.settings.slow_window,
                self.settings.entry_edge_bps,
                self.settings.exit_edge_bps,
                True,
            )
            avg_entry = float(position.get("avg_entry_price", 0) or 0)
            current = float(position.get("current_price", signal.last_price or 0) or 0)
            pnl_pct = ((current / avg_entry) - 1.0) if avg_entry and current else 0.0
            should_exit = (
                signal.action == "SELL"
                or pnl_pct >= self.settings.take_profit_pct
                or pnl_pct <= -self.settings.stop_loss_pct
            )
            if should_exit and self._allowed_action(symbol, signal.bar_time):
                if self.settings.can_trade:
                    order = await self.client.close_position(symbol)
                    self._mark_trade(symbol, signal.bar_time)
                    self.state.submitted_orders += 1
                    trade_count += 1
                    self._record_execution(order, symbol, "sell", signal.last_price or current, signal.reason)
                    log.info("SELL %s pnl=%.4f reason=%s", symbol, pnl_pct, signal.reason)
                else:
                    log.warning("LIVE SAFETY BLOCK: would SELL %s", symbol)

        # Refresh positions after exits before looking for entries.
        positions = {p["symbol"]: p for p in await self.client.positions()}
        slots = max(0, self.settings.max_open_positions - len(positions))
        if slots <= 0:
            return

        available_cash = float(account.get("cash", 0) or 0)

        for symbol in self.settings.symbols:
            if slots <= 0:
                break
            if symbol in positions:
                continue
            if trade_count >= self.settings.max_trades_per_day:
                break
            bars = bars_map.get(symbol, [])
            signal = moving_average_momentum(
                bars,
                self.settings.fast_window,
                self.settings.slow_window,
                self.settings.entry_edge_bps,
                self.settings.exit_edge_bps,
                False,
            )
            if signal.action != "BUY" or not self._allowed_action(symbol, signal.bar_time):
                continue

            notional = min(self.settings.trade_notional, available_cash)
            if notional < 1.0:
                self.state.last_error = "Cash below $1 minimum micro-trade size"
                return

            if self.settings.can_trade:
                order = await self.client.submit_market_buy(symbol, notional)
                self._mark_trade(symbol, signal.bar_time)
                self.state.submitted_orders += 1
                trade_count += 1
                available_cash -= notional
                slots -= 1
                self._record_execution(order, symbol, "buy", signal.last_price or 0.0, signal.reason)
                log.info("BUY %s $%.2f edge=%.2fbps", symbol, notional, signal.edge_bps or 0)
            else:
                log.warning("LIVE SAFETY BLOCK: would BUY %s $%.2f", symbol, notional)


    def _record_execution(self, order: Optional[dict], symbol: str, side: str, signal_price: float, reason: str):
        if not order or not isinstance(order, dict):
            return
        self.execution_log.append({
            "order_id": order.get("id"),
            "symbol": symbol,
            "side": side,
            "submitted_at": order.get("submitted_at") or datetime.now(timezone.utc).isoformat(),
            "signal_price": float(signal_price or 0),
            "filled_avg_price": order.get("filled_avg_price"),
            "status": order.get("status"),
            "adverse_slippage_bps": None,
            "reason": reason,
        })
        if len(self.execution_log) > 1000:
            del self.execution_log[:-1000]

    async def _reconcile_executions(self):
        pending = [e for e in self.execution_log if e.get("order_id") and e.get("status") not in {"filled", "canceled", "expired", "rejected"}]
        for event in pending[-50:]:
            try:
                order = await self.client.get_order(event["order_id"])
            except Exception:
                continue
            event["status"] = order.get("status")
            event["filled_avg_price"] = order.get("filled_avg_price")
            fill_raw = order.get("filled_avg_price")
            signal_price = float(event.get("signal_price") or 0)
            if fill_raw and signal_price > 0:
                fill = float(fill_raw)
                if event.get("side") == "buy":
                    slip = ((fill / signal_price) - 1.0) * 10_000
                else:
                    slip = ((signal_price / fill) - 1.0) * 10_000
                event["adverse_slippage_bps"] = round(slip, 3)

    def _allowed_action(self, symbol: str, bar_time: Optional[str]) -> bool:
        now = datetime.now(timezone.utc).timestamp()
        if now - self._last_trade_ts.get(symbol, 0.0) < self.settings.cooldown_seconds:
            return False
        if bar_time and self._last_bar_action.get(symbol) == bar_time:
            return False
        return True

    def _mark_trade(self, symbol: str, bar_time: Optional[str]):
        self._last_trade_ts[symbol] = datetime.now(timezone.utc).timestamp()
        if bar_time:
            self._last_bar_action[symbol] = bar_time
