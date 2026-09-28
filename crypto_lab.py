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
        })
        return payload
