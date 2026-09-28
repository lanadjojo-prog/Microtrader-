from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Deque, Dict, Optional

import httpx
import websockets

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
    book_events: int = 0
    trade_events: int = 0
    reconnects: int = 0
    last_update: Optional[str] = None


class CryptoMicrostructureLab:
    """Simulation-only realtime crypto microstructure research on public Bitvavo data."""

    BASE_URL = "https://api.bitvavo.com/v2"
    WS_URL = "wss://ws.bitvavo.com/v2/"
    STRATEGIES = (
        "mean_reversion",
        "market_maker",
        "top_imbalance",
        "full_book_imbalance",
        "order_flow",
        "liquidity_vacuum",
        "flow_reversal",
        "momentum",
        "hybrid",
        "adaptive",
    )

    def __init__(self, settings: Settings):
        self.settings = settings
        self.state = CryptoLabState()
        self._task: Optional[asyncio.Task] = None
        self._client = httpx.AsyncClient(timeout=10.0)
        self._mid: Dict[str, Deque[float]] = {
            symbol: deque(maxlen=max(60, settings.crypto_lab_window * 6))
            for symbol in settings.crypto_lab_symbols
        }
        self._latest: Dict[str, dict] = {}
        self._books: Dict[str, dict] = {}
        self._flows: Dict[str, Deque[tuple[float, float]]] = {
            symbol: deque(maxlen=5000) for symbol in settings.crypto_lab_symbols
        }
        self._sim: Dict[str, dict] = {}
        self._sim_history = defaultdict(list)

    def _maker_fee_bps(self, symbol: str) -> float:
        quote = symbol.rsplit("-", 1)[-1].upper()
        if quote == "USDC":
            return 5.0
        return self.settings.crypto_lab_maker_fee_bps

    def _sim_key(self, strategy: str, symbol: str) -> str:
        return f"{strategy}:{symbol}"

    def _ensure_sim(self, strategy: str, symbol: str) -> dict:
        key = self._sim_key(strategy, symbol)
        if key not in self._sim:
            self._sim[key] = {
                "strategy": strategy,
                "market": symbol,
                "quote": symbol.rsplit("-", 1)[-1],
                "state": "flat",
                "pending_price": None,
                "pending_age": 0,
                "entry_price": None,
                "entry_cycle": None,
                "exit_quote": None,
                "exit_age": 0,
                "trades": 0,
                "wins": 0,
                "gross_pnl_quote": 0.0,
                "fees_quote": 0.0,
                "net_pnl_quote": 0.0,
                "equity_quote": 0.0,
                "peak_equity_quote": 0.0,
                "max_drawdown_quote": 0.0,
                "sum_return_bps": 0.0,
            }
        return self._sim[key]

    async def start(self):
        if self.state.running:
            return
        self.state = CryptoLabState(
            running=True,
            started_at=datetime.now(timezone.utc).isoformat(),
            stage="seeding",
            message="Seeding public Bitvavo order books",
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

    async def _seed_book(self, symbol: str):
        depth = max(5, min(1000, self.settings.crypto_lab_book_depth))
        r = await self._client.get(f"{self.BASE_URL}/{symbol}/book", params={"depth": depth})
        r.raise_for_status()
        data = r.json()
        self._books[symbol] = {
            "nonce": int(data.get("nonce") or 0),
            "bids": {float(p): float(q) for p, q in data.get("bids", []) if float(q) > 0},
            "asks": {float(p): float(q) for p, q in data.get("asks", []) if float(q) > 0},
        }

    async def _seed_all(self):
        for symbol in self.settings.crypto_lab_symbols:
            try:
                await self._seed_book(symbol)
                self._process_symbol(symbol)
            except Exception as exc:
                log.warning("Could not seed %s: %s", symbol, exc)

    def _apply_book_event(self, data: dict):
        symbol = str(data.get("market", "")).upper()
        if symbol not in self._books:
            return False
        book = self._books[symbol]
        nonce = int(data.get("nonce") or 0)
        if book["nonce"] and nonce and nonce <= book["nonce"]:
            return False
        if book["nonce"] and nonce and nonce > book["nonce"] + 1:
            return None
        for p, q in data.get("bids", []):
            price, size = float(p), float(q)
            if size <= 0:
                book["bids"].pop(price, None)
            else:
                book["bids"][price] = size
        for p, q in data.get("asks", []):
            price, size = float(p), float(q)
            if size <= 0:
                book["asks"].pop(price, None)
            else:
                book["asks"][price] = size
        if nonce:
            book["nonce"] = nonce
        return True

    def _record_trade(self, data: dict):
        symbol = str(data.get("market", "")).upper()
        if symbol not in self._flows:
            return
        try:
            amount = float(data.get("amount") or 0)
            price = float(data.get("price") or 0)
        except (TypeError, ValueError):
            return
        if amount <= 0 or price <= 0:
            return
        side = str(data.get("side", "")).lower()
        signed_quote = amount * price * (1.0 if side == "buy" else -1.0)
        now = time.time()
        flow = self._flows[symbol]
        flow.append((now, signed_quote))
        cutoff = now - self.settings.crypto_lab_flow_window_seconds
        while flow and flow[0][0] < cutoff:
            flow.popleft()

    def _book_features(self, symbol: str) -> Optional[dict]:
        book = self._books.get(symbol)
        if not book or not book["bids"] or not book["asks"]:
            return None
        depth = max(1, self.settings.crypto_lab_book_depth)
        bids = sorted(book["bids"].items(), reverse=True)[:depth]
        asks = sorted(book["asks"].items())[:depth]
        bid, bid_size = bids[0]
        ask, ask_size = asks[0]
        if bid <= 0 or ask <= 0 or ask < bid:
            return None

        mid = (bid + ask) / 2
        spread_bps = ((ask - bid) / mid) * 10000 if mid else 0.0
        top_total = bid_size + ask_size
        top_imbalance = ((bid_size - ask_size) / top_total) if top_total > 0 else 0.0

        bid_depth = sum(p * q for p, q in bids)
        ask_depth = sum(p * q for p, q in asks)
        depth_total = bid_depth + ask_depth
        full_book_imbalance = ((bid_depth - ask_depth) / depth_total) if depth_total > 0 else 0.0

        flow = self._flows[symbol]
        buy_flow = sum(v for _, v in flow if v > 0)
        sell_flow = -sum(v for _, v in flow if v < 0)
        flow_total = buy_flow + sell_flow
        flow_imbalance = ((buy_flow - sell_flow) / flow_total) if flow_total > 0 else 0.0

        history = self._mid.setdefault(
            symbol, deque(maxlen=max(60, self.settings.crypto_lab_window * 6))
        )
        if not history or abs(history[-1] - mid) > 1e-12:
            history.append(mid)
        window = list(history)[-self.settings.crypto_lab_window:]
        zscore = 0.0
        if len(window) >= max(10, self.settings.crypto_lab_window // 2):
            avg = mean(window)
            sd = pstdev(window)
            if sd > 0:
                zscore = (mid - avg) / sd

        fee_bps = self._maker_fee_bps(symbol)
        maker_edge_bps = spread_bps - (2 * fee_bps)
        book_t = self.settings.crypto_lab_book_threshold
        flow_t = self.settings.crypto_lab_flow_threshold

        signals = {
            "mean_reversion": zscore <= -self.settings.crypto_lab_z_entry,
            "market_maker": maker_edge_bps >= self.settings.crypto_lab_target_edge_bps,
            "top_imbalance": top_imbalance >= book_t,
            "full_book_imbalance": full_book_imbalance >= book_t,
            "order_flow": flow_imbalance >= flow_t,
            "liquidity_vacuum": ask_depth > 0 and bid_depth / ask_depth >= 1.8 and flow_imbalance >= flow_t,
            "flow_reversal": zscore <= -self.settings.crypto_lab_z_entry and flow_imbalance <= -flow_t and full_book_imbalance > 0,
            "momentum": flow_imbalance >= flow_t and full_book_imbalance >= book_t,
            "hybrid": zscore <= -self.settings.crypto_lab_z_entry and full_book_imbalance > 0 and flow_imbalance > 0,
            "adaptive": (
                (abs(zscore) >= self.settings.crypto_lab_z_entry and full_book_imbalance > 0)
                or (flow_imbalance >= flow_t and full_book_imbalance >= book_t)
            ),
        }
        return {
            "market": symbol,
            "quote": symbol.rsplit("-", 1)[-1],
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_bps": spread_bps,
            "top_imbalance": top_imbalance,
            "full_book_imbalance": full_book_imbalance,
            "bid_depth_quote": bid_depth,
            "ask_depth_quote": ask_depth,
            "buy_flow_quote": buy_flow,
            "sell_flow_quote": sell_flow,
            "flow_imbalance": flow_imbalance,
            "zscore": zscore,
            "maker_fee_bps_one_way": fee_bps,
            "maker_net_spread_bps": maker_edge_bps,
            "signals": signals,
        }

    def _exit_condition(self, strategy: str, feat: dict, hold_cycles: int) -> bool:
        if hold_cycles >= self.settings.crypto_lab_max_hold_cycles:
            return True
        if strategy in {"mean_reversion", "flow_reversal", "hybrid"}:
            return feat["zscore"] >= 0
        if strategy == "market_maker":
            return True
        if strategy in {"top_imbalance", "full_book_imbalance", "order_flow", "liquidity_vacuum", "momentum"}:
            return not feat["signals"].get(strategy, False)
        if strategy == "adaptive":
            return not feat["signals"]["adaptive"]
        return True

    def _advance_simulator(self, feat: dict, now: str):
        symbol = feat["market"]
        bid = feat["bid"]
        ask = feat["ask"]
        notional = self.settings.crypto_lab_notional_eur
        fee_rate = feat["maker_fee_bps_one_way"] / 10000.0

        for strategy in self.STRATEGIES:
            s = self._ensure_sim(strategy, symbol)
            if s["state"] == "flat" and feat["signals"].get(strategy, False):
                s["state"] = "entry_pending"
                s["pending_price"] = bid
                s["pending_age"] = 0

            elif s["state"] == "entry_pending":
                s["pending_age"] += 1
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
                if self._exit_condition(strategy, feat, hold):
                    target = ask
                    if strategy == "market_maker":
                        target = max(
                            ask,
                            float(s["entry_price"]) * (
                                1 + (
                                    2 * feat["maker_fee_bps_one_way"]
                                    + self.settings.crypto_lab_target_edge_bps
                                ) / 10000.0
                            ),
                        )
                    s["state"] = "exit_pending"
                    s["exit_quote"] = target
                    s["exit_age"] = 0

            elif s["state"] == "exit_pending":
                s["exit_age"] += 1
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
                    s["gross_pnl_quote"] += gross
                    s["fees_quote"] += fees
                    s["net_pnl_quote"] += net
                    s["equity_quote"] += net
                    s["peak_equity_quote"] = max(s["peak_equity_quote"], s["equity_quote"])
                    s["max_drawdown_quote"] = max(
                        s["max_drawdown_quote"],
                        s["peak_equity_quote"] - s["equity_quote"],
                    )
                    s["sum_return_bps"] += ret_bps
                    self._sim_history[strategy].append({
                        "market": symbol,
                        "quote": feat["quote"],
                        "entry": round(entry, 8),
                        "exit": round(exit_price, 8),
                        "fees_quote": round(fees, 6),
                        "net_pnl_quote": round(net, 6),
                        "return_bps": round(ret_bps, 4),
                        "closed_at": now,
                    })
                    self._sim_history[strategy] = self._sim_history[strategy][-200:]
                    s["state"] = "flat"
                    s["entry_price"] = None
                    s["entry_cycle"] = None
                    s["exit_quote"] = None
                    s["exit_age"] = 0
                elif s["exit_age"] >= self.settings.crypto_lab_pending_cycles:
                    s["state"] = "open"
                    s["exit_quote"] = None
                    s["exit_age"] = 0

    def _process_symbol(self, symbol: str):
        feat = self._book_features(symbol)
        if not feat:
            return
        now = datetime.now(timezone.utc).isoformat()
        self.state.cycles += 1
        self._advance_simulator(feat, now)
        self._latest[symbol] = {
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in feat.items() if k != "signals"},
            "signals": feat["signals"],
            "samples": len(self._mid[symbol]),
            "updated_at": now,
        }
        self.state.observations += 1
        self.state.last_update = now

    async def _websocket_loop(self):
        markets = self.settings.crypto_lab_symbols
        subscribe = {
            "action": "subscribe",
            "channels": [
                {"name": "book", "markets": markets},
                {"name": "trades", "markets": markets},
            ],
        }
        while self.state.running:
            try:
                self.state.stage = "streaming"
                self.state.message = "Realtime Bitvavo order book + trade flow active"
                async with websockets.connect(self.WS_URL, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(json.dumps(subscribe))
                    async for raw in ws:
                        if not self.state.running:
                            break
                        data = json.loads(raw)
                        event = data.get("event")
                        if event == "book":
                            symbol = str(data.get("market", "")).upper()
                            applied = self._apply_book_event(data)
                            if applied is None:
                                await self._seed_book(symbol)
                            if applied is not False:
                                self.state.book_events += 1
                                self._process_symbol(symbol)
                        elif event == "trade":
                            symbol = str(data.get("market", "")).upper()
                            self._record_trade(data)
                            self.state.trade_events += 1
                            self._process_symbol(symbol)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.state.reconnects += 1
                self.state.last_error = str(exc)
                self.state.stage = "reconnecting"
                self.state.message = f"WebSocket reconnecting: {exc}"
                log.warning("Bitvavo WebSocket reconnect: %s", exc)
                await asyncio.sleep(2)

    async def _run(self):
        try:
            await self._seed_all()
            if not self._books:
                raise RuntimeError("No usable Bitvavo order books returned")
            await self._websocket_loop()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.stage = "error"
            self.state.message = str(exc)
            log.exception("Crypto Lab failed")
        finally:
            self.state.running = False

    def _sim_summary(self) -> dict:
        out = {}
        for strategy in self.STRATEGIES:
            rows = [s for s in self._sim.values() if s["strategy"] == strategy]
            trades = sum(int(s["trades"]) for s in rows)
            wins = sum(int(s["wins"]) for s in rows)
            net = sum(float(s["net_pnl_quote"]) for s in rows)
            fees = sum(float(s["fees_quote"]) for s in rows)
            sum_bps = sum(float(s["sum_return_bps"]) for s in rows)
            max_dd = max([float(s["max_drawdown_quote"]) for s in rows] or [0.0])
            out[strategy] = {
                "trades": trades,
                "wins": wins,
                "win_rate_pct": round((wins / trades * 100) if trades else 0.0, 2),
                "fees_quote": round(fees, 5),
                "net_pnl_quote": round(net, 5),
                "net_pnl_eur": round(net, 5),
                "expectancy_bps": round((sum_bps / trades) if trades else 0.0, 4),
                "max_drawdown_quote": round(max_dd, 5),
                "max_drawdown_eur": round(max_dd, 5),
                "open_positions": sum(1 for s in rows if s["state"] in {"open", "exit_pending"}),
                "pending_entries": sum(1 for s in rows if s["state"] == "entry_pending"),
                "recent_trades": list(self._sim_history[strategy][-8:]),
            }
        return out

    def _quote_summary(self) -> dict:
        out = {}
        for quote in ("EUR", "USDC"):
            rows = [s for s in self._sim.values() if s["quote"] == quote]
            trades = sum(int(s["trades"]) for s in rows)
            sum_bps = sum(float(s["sum_return_bps"]) for s in rows)
            net = sum(float(s["net_pnl_quote"]) for s in rows)
            out[quote] = {
                "trades": trades,
                "net_pnl_quote": round(net, 5),
                "expectancy_bps": round(sum_bps / trades, 4) if trades else 0.0,
            }
        return out

    def public_state(self) -> dict:
        payload = asdict(self.state)
        payload.update({
            "mode": "SIMULATION_ONLY",
            "venue": "Bitvavo public WebSocket + REST seed",
            "symbols": self.settings.crypto_lab_symbols,
            "stream": "book + trades",
            "book_depth": self.settings.crypto_lab_book_depth,
            "flow_window_seconds": self.settings.crypto_lab_flow_window_seconds,
            "window": self.settings.crypto_lab_window,
            "markets": [self._latest[s] for s in self.settings.crypto_lab_symbols if s in self._latest],
            "paper_simulation": {
                "fill_model": "conservative maker-touch",
                "notional_eur_per_trade": self.settings.crypto_lab_notional_eur,
                "pending_cycles": self.settings.crypto_lab_pending_cycles,
                "max_hold_cycles": self.settings.crypto_lab_max_hold_cycles,
                "strategies": self._sim_summary(),
                "quote_summary": self._quote_summary(),
            },
        })
        return payload
