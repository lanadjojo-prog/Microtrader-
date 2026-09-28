from __future__ import annotations

import asyncio
import logging
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Deque, Dict, Optional

import httpx

from config import Settings

log = logging.getLogger("microtrader.crypto_lab")


@dataclass
class CryptoLabState:
    running: bool = False
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    last_error: Optional[str] = None
    stage: str = "idle"
    message: str = "Ready"
    observations: int = 0
    cycles: int = 0
    last_update: Optional[str] = None


class CryptoMicrostructureLab:
    """Simulation-only crypto microstructure research using public Bitvavo market data."""

    BASE_URL = "https://api.bitvavo.com/v2"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.state = CryptoLabState()
        self._task: Optional[asyncio.Task] = None
        self._client = httpx.AsyncClient(timeout=10.0)
        self._mid: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=max(30, settings.crypto_lab_window * 4))
            for symbol in settings.crypto_lab_symbols
        }
        self._latest: Dict[str, dict] = {}
        self._stats = defaultdict(lambda: {
            "signals": 0,
            "mean_reversion_signals": 0,
            "maker_positive_spread": 0,
            "imbalance_signals": 0,
            "hybrid_signals": 0,
        })
        self._sim = {}
        self._sim_history = defaultdict(list)

    def _sim_key(self, strategy: str, symbol: str) -> str:
        return f"{strategy}:{symbol}"

    def _ensure_sim(self, strategy: str, symbol: str) -> dict:
        key = self._sim_key(strategy, symbol)
        if key not in self._sim:
            self._sim[key] = {
                "strategy": strategy,
                "market": symbol,
                "state": "flat",
                "pending_side": None,
                "pending_price": None,
                "pending_age": 0,
                "entry_price": None,
                "entry_cycle": None,
                "exit_quote": None,
                "exit_age": 0,
                "trades": 0,
                "wins": 0,
                "gross_pnl_eur": 0.0,
                "fees_eur": 0.0,
                "net_pnl_eur": 0.0,
                "equity_eur": 0.0,
                "peak_equity_eur": 0.0,
                "max_drawdown_eur": 0.0,
                "sum_return_bps": 0.0,
            }
        return self._sim[key]

    def _entry_condition(self, strategy: str, zscore: float, imbalance: float, maker_signal: bool) -> bool:
        if strategy == "mean_reversion":
            return zscore <= -self.settings.crypto_lab_z_entry
        if strategy == "market_maker":
            return maker_signal
        if strategy == "imbalance":
            return imbalance >= self.settings.crypto_lab_imbalance_threshold
        if strategy == "hybrid":
            return (
                zscore <= -self.settings.crypto_lab_z_entry
                and imbalance >= self.settings.crypto_lab_imbalance_threshold
                and maker_signal
            )
        return False

    def _exit_condition(self, strategy: str, zscore: float, imbalance: float, hold_cycles: int) -> bool:
        if hold_cycles >= self.settings.crypto_lab_max_hold_cycles:
            return True
        if strategy == "mean_reversion":
            return zscore >= 0
        if strategy == "market_maker":
            return True
        if strategy == "imbalance":
            return imbalance <= 0
        if strategy == "hybrid":
            return zscore >= 0 or imbalance <= 0
        return False

    def _advance_simulator(self, symbol: str, bid: float, ask: float, zscore: float, imbalance: float, maker_signal: bool, now: str):
        notional = self.settings.crypto_lab_notional_eur
        fee_rate = self.settings.crypto_lab_maker_fee_bps / 10000.0
        strategies = ("mean_reversion", "market_maker", "imbalance", "hybrid")

        for strategy in strategies:
            s = self._ensure_sim(strategy, symbol)

            if s["state"] == "flat" and self._entry_condition(strategy, zscore, imbalance, maker_signal):
                s["state"] = "entry_pending"
                s["pending_side"] = "buy"
                s["pending_price"] = bid
                s["pending_age"] = 0

            elif s["state"] == "entry_pending":
                s["pending_age"] += 1
                # Conservative maker fill model: buy quote only fills after the ask trades down to our bid.
                if ask <= float(s["pending_price"]):
                    s["state"] = "open"
                    s["entry_price"] = float(s["pending_price"])
                    s["entry_cycle"] = self.state.cycles
                    s["pending_price"] = None
                    s["pending_age"] = 0
                elif s["pending_age"] >= self.settings.crypto_lab_pending_cycles:
                    s["state"] = "flat"
                    s["pending_price"] = None
                    s["pending_age"] = 0

            elif s["state"] == "open":
                hold = max(0, self.state.cycles - int(s["entry_cycle"] or self.state.cycles))
                if self._exit_condition(strategy, zscore, imbalance, hold):
                    target = ask
                    if strategy == "market_maker":
                        target = max(
                            ask,
                            float(s["entry_price"]) * (
                                1 + (
                                    2 * self.settings.crypto_lab_maker_fee_bps
                                    + self.settings.crypto_lab_target_edge_bps
                                ) / 10000.0
                            ),
                        )
                    s["state"] = "exit_pending"
                    s["exit_quote"] = target
                    s["exit_age"] = 0

            elif s["state"] == "exit_pending":
                s["exit_age"] += 1
                # Conservative maker fill model: sell quote only fills after bid reaches our ask.
                if bid >= float(s["exit_quote"]):
                    entry = float(s["entry_price"])
                    exit_price = float(s["exit_quote"])
                    qty = notional / entry if entry > 0 else 0.0
                    gross = qty * (exit_price - entry)
                    fees = notional * fee_rate + (qty * exit_price) * fee_rate
                    net = gross - fees
                    ret_bps = (net / notional) * 10000 if notional > 0 else 0.0

                    s["trades"] += 1
                    s["wins"] += int(net > 0)
                    s["gross_pnl_eur"] += gross
                    s["fees_eur"] += fees
                    s["net_pnl_eur"] += net
                    s["equity_eur"] += net
                    s["peak_equity_eur"] = max(s["peak_equity_eur"], s["equity_eur"])
                    dd = s["peak_equity_eur"] - s["equity_eur"]
                    s["max_drawdown_eur"] = max(s["max_drawdown_eur"], dd)
                    s["sum_return_bps"] += ret_bps

                    self._sim_history[strategy].append({
                        "market": symbol,
                        "entry": round(entry, 8),
                        "exit": round(exit_price, 8),
                        "gross_pnl_eur": round(gross, 6),
                        "fees_eur": round(fees, 6),
                        "net_pnl_eur": round(net, 6),
                        "return_bps": round(ret_bps, 4),
                        "closed_at": now,
                    })
                    if len(self._sim_history[strategy]) > 200:
                        self._sim_history[strategy] = self._sim_history[strategy][-200:]

                    s["state"] = "flat"
                    s["entry_price"] = None
                    s["entry_cycle"] = None
                    s["exit_quote"] = None
                    s["exit_age"] = 0
                elif s["exit_age"] >= self.settings.crypto_lab_pending_cycles:
                    # Cancel stale exit and re-evaluate next cycle.
                    s["state"] = "open"
                    s["exit_quote"] = None
                    s["exit_age"] = 0

    def _sim_summary(self) -> dict:
        out = {}
        for strategy in ("mean_reversion", "market_maker", "imbalance", "hybrid"):
            rows = [s for s in self._sim.values() if s["strategy"] == strategy]
            trades = sum(int(s["trades"]) for s in rows)
            wins = sum(int(s["wins"]) for s in rows)
            gross = sum(float(s["gross_pnl_eur"]) for s in rows)
            fees = sum(float(s["fees_eur"]) for s in rows)
            net = sum(float(s["net_pnl_eur"]) for s in rows)
            sum_bps = sum(float(s["sum_return_bps"]) for s in rows)
            max_dd = max([float(s["max_drawdown_eur"]) for s in rows] or [0.0])
            open_positions = sum(1 for s in rows if s["state"] in {"open", "exit_pending"})
            pending = sum(1 for s in rows if s["state"] == "entry_pending")
            out[strategy] = {
                "trades": trades,
                "wins": wins,
                "win_rate_pct": round((wins / trades * 100) if trades else 0.0, 2),
                "gross_pnl_eur": round(gross, 4),
                "fees_eur": round(fees, 4),
                "net_pnl_eur": round(net, 4),
                "expectancy_bps": round((sum_bps / trades) if trades else 0.0, 4),
                "max_drawdown_eur": round(max_dd, 4),
                "open_positions": open_positions,
                "pending_entries": pending,
                "recent_trades": list(self._sim_history[strategy][-8:]),
            }
        return out

    async def start(self):
        if self.state.running:
            return
        self.state = CryptoLabState(
            running=True,
            started_at=datetime.now(timezone.utc).isoformat(),
            stage="collecting",
            message="Collecting public Bitvavo order-book snapshots",
        )
        self._task = asyncio.create_task(self._run(), name="microtrader-crypto-lab")

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.state.running = False
        self.state.stage = "stopped"
        self.state.message = "Crypto Lab stopped"
        self.state.stopped_at = datetime.now(timezone.utc).isoformat()
        self._task = None

    async def close(self):
        await self.stop()
        await self._client.aclose()

    async def _run(self):
        try:
            while self.state.running:
                await self._collect_once()
                await asyncio.sleep(max(1, self.settings.crypto_lab_poll_seconds))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.stage = "error"
            self.state.message = str(exc)
            log.exception("Crypto Lab failed")
        finally:
            self.state.running = False

    async def _collect_once(self):
        response = await self._client.get(f"{self.BASE_URL}/ticker/book")
        response.raise_for_status()
        rows = response.json()
        if not isinstance(rows, list):
            rows = [rows]
        wanted = set(self.settings.crypto_lab_symbols)
        now = datetime.now(timezone.utc).isoformat()

        for row in rows:
            symbol = str(row.get("market", "")).upper()
            if symbol not in wanted:
                continue
            try:
                bid = float(row["bid"])
                ask = float(row["ask"])
                bid_size = float(row.get("bidSize") or 0)
                ask_size = float(row.get("askSize") or 0)
            except (KeyError, TypeError, ValueError):
                continue
            if bid <= 0 or ask <= 0 or ask < bid:
                continue

            mid = (bid + ask) / 2
            spread_bps = ((ask - bid) / mid) * 10000 if mid else 0
            sizes = bid_size + ask_size
            imbalance = ((bid_size - ask_size) / sizes) if sizes > 0 else 0.0

            history = self._mid.setdefault(
                symbol, deque(maxlen=max(30, self.settings.crypto_lab_window * 4))
            )
            history.append(mid)
            window = list(history)[-self.settings.crypto_lab_window:]
            zscore = 0.0
            if len(window) >= max(10, self.settings.crypto_lab_window // 2):
                avg = mean(window)
                sd = pstdev(window)
                if sd > 0:
                    zscore = (mid - avg) / sd

            maker_roundtrip_bps = 2 * self.settings.crypto_lab_maker_fee_bps
            maker_net_spread_bps = spread_bps - maker_roundtrip_bps
            mean_reversion_signal = abs(zscore) >= self.settings.crypto_lab_z_entry
            imbalance_signal = abs(imbalance) >= self.settings.crypto_lab_imbalance_threshold
            maker_signal = maker_net_spread_bps > 0
            hybrid_signal = mean_reversion_signal and imbalance_signal and maker_signal

            stats = self._stats[symbol]
            stats["signals"] += 1
            stats["mean_reversion_signals"] += int(mean_reversion_signal)
            stats["maker_positive_spread"] += int(maker_signal)
            stats["imbalance_signals"] += int(imbalance_signal)
            stats["hybrid_signals"] += int(hybrid_signal)

            self._advance_simulator(symbol, bid, ask, zscore, imbalance, maker_signal, now)

            self._latest[symbol] = {
                "market": symbol,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_bps": round(spread_bps, 4),
                "bid_size": bid_size,
                "ask_size": ask_size,
                "imbalance": round(imbalance, 4),
                "zscore": round(zscore, 4),
                "maker_fee_bps_one_way": self.settings.crypto_lab_maker_fee_bps,
                "maker_net_spread_bps": round(maker_net_spread_bps, 4),
                "mean_reversion_signal": mean_reversion_signal,
                "market_maker_signal": maker_signal,
                "imbalance_signal": imbalance_signal,
                "hybrid_signal": hybrid_signal,
                "samples": len(history),
                "updated_at": now,
            }
            self.state.observations += 1

        self.state.cycles += 1
        self.state.last_update = now
        self.state.stage = "collecting"
        self.state.message = (
            f"Watching {len(self._latest)}/{len(wanted)} configured markets; "
            f"{self.state.observations} observations collected"
        )

    def public_state(self) -> dict:
        payload = asdict(self.state)
        payload.update({
            "mode": "SIMULATION_ONLY",
            "venue": "Bitvavo public market data",
            "symbols": self.settings.crypto_lab_symbols,
            "poll_seconds": self.settings.crypto_lab_poll_seconds,
            "window": self.settings.crypto_lab_window,
            "maker_fee_bps_one_way": self.settings.crypto_lab_maker_fee_bps,
            "z_entry": self.settings.crypto_lab_z_entry,
            "imbalance_threshold": self.settings.crypto_lab_imbalance_threshold,
            "markets": [self._latest[s] for s in self.settings.crypto_lab_symbols if s in self._latest],
            "strategy_stats": {k: dict(v) for k, v in self._stats.items()},
            "paper_simulation": {
                "fill_model": "conservative maker-touch",
                "notional_eur_per_trade": self.settings.crypto_lab_notional_eur,
                "pending_cycles": self.settings.crypto_lab_pending_cycles,
                "max_hold_cycles": self.settings.crypto_lab_max_hold_cycles,
                "target_edge_bps": self.settings.crypto_lab_target_edge_bps,
                "strategies": self._sim_summary(),
            },
        })
        return payload
