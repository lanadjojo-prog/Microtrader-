from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Dict, List, Optional

from alpaca_client import AlpacaClient
from config import Settings
from strategy_store import StrategyStore

log = logging.getLogger("microtrader.strategy_lab")


@dataclass(frozen=True)
class Candidate:
    strategy: str
    params: dict


@dataclass
class LabState:
    running: bool = False
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    last_error: Optional[str] = None
    progress: int = 0
    total: int = 0
    stage: str = "idle"
    message: str = "Not started"
    symbols_loaded: int = 0
    symbols_total: int = 0
    generation: int = 0
    tested_total: int = 0
    promoted_total: int = 0
    target_promoted: int = 0


class StrategyLab:
    def __init__(self, settings: Settings, client: AlpacaClient):
        self.settings = settings
        self.client = client
        self.state = LabState()
        self._task: Optional[asyncio.Task] = None
        self._results: List[dict] = []
        self._summary: dict = {}
        self.store = StrategyStore(settings.database_url)

    def public_state(self) -> dict:
        payload = asdict(self.state)
        payload["summary"] = self._summary
        return payload

    def results(self) -> List[dict]:
        return list(self._results)

    async def start(self):
        if self.state.running:
            return
        self.state = LabState(
            running=True,
            started_at=datetime.now(timezone.utc).isoformat(),
            stage="starting",
            message="Preparing Strategy Lab",
            symbols_total=len(self.settings.lab_symbols),
            target_promoted=self.settings.lab_target_promoted,
        )
        self._results = []
        self._summary = {}
        self._task = asyncio.create_task(self._run(), name="microtrader-strategy-lab")

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.state.running = False
        self._task = None

    async def _run(self):
        try:
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=self.settings.lab_lookback_days)
            bars_by_symbol: Dict[str, List[dict]] = {}
            self.state.stage = "loading_data"
            self.state.message = "Loading historical market data"
            for symbol in self.settings.lab_symbols:
                self.state.message = f"Loading {symbol}"
                bars = await self.client.historical_bars(
                    symbol=symbol,
                    start=start,
                    end=end,
                    timeframe=self.settings.lab_timeframe,
                    max_bars=self.settings.lab_max_bars_per_symbol,
                )
                if len(bars) >= 100:
                    bars_by_symbol[symbol] = bars
                self.state.symbols_loaded += 1
                await asyncio.sleep(0)

            if not bars_by_symbol:
                raise RuntimeError("No usable historical bars returned for Strategy Lab")

            await self.store.init()
            persisted = await self.store.load_results(limit=250)
            persisted_signatures = await self.store.load_signatures()
            persisted_state = await self.store.load_state()

            stream = candidate_stream()
            seen = set(persisted_signatures)
            results: List[dict] = list(persisted)
            promoted: List[dict] = [r for r in results if r.get("promoted")]
            batch_size = max(1, self.settings.lab_batch_size)

            self.state.generation = int(persisted_state.get("generation", 0))
            self.state.tested_total = int(persisted_state.get("tested_total", len(seen)))
            self.state.promoted_total = int(persisted_state.get("promoted_total", len(promoted)))
            self._results = list(results)
            log.info(
                "Strategy Lab resume: loaded_results=%s loaded_signatures=%s generation=%s tested_total=%s promoted_total=%s",
                len(results), len(seen), self.state.generation, self.state.tested_total, self.state.promoted_total
            )

            self.state.stage = "testing"
            self.state.total = batch_size

            while self.state.running:
                self.state.generation += 1
                self.state.progress = 0
                self.state.total = batch_size
                self.state.message = (
                    f"Generation {self.state.generation}: testing next {batch_size} candidates"
                )

                batch: List[Candidate] = []
                while len(batch) < batch_size:
                    candidate = next(stream)
                    signature = candidate_signature(candidate)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    batch.append(candidate)

                for idx, candidate in enumerate(batch, start=1):
                    result = evaluate_candidate(
                        candidate,
                        bars_by_symbol,
                        cost_bps=self.settings.lab_cost_bps,
                        stress_cost_multiplier=self.settings.lab_stress_cost_multiplier,
                        min_oos_trades=self.settings.lab_min_oos_trades,
                        min_profit_factor=self.settings.lab_min_profit_factor,
                        max_drawdown_pct=self.settings.lab_max_drawdown_pct,
                        min_positive_symbol_ratio=self.settings.lab_min_positive_symbol_ratio,
                    )
                    results.append(result)
                    signature = candidate_signature(candidate)
                    await self.store.save_result(signature, result)
                    if result["promoted"]:
                        promoted.append(result)

                    self.state.progress = idx
                    self.state.tested_total += 1
                    self.state.promoted_total = len(promoted)
                    self.state.message = (
                        f"Generation {self.state.generation}: {idx}/{batch_size} tested · "
                        f"{len(promoted)}/{self.settings.lab_target_promoted} promoted"
                    )

                    results.sort(
                        key=lambda row: (
                            bool(row["promoted"]),
                            row["oos"]["expectancy_bps"],
                            row["oos"]["profit_factor"],
                        ),
                        reverse=True,
                    )
                    self._results = results[:250]
                    await self.store.save_state(
                        self.state.generation,
                        self.state.tested_total,
                        self.state.promoted_total,
                    )
                    log.info(
                        "Strategy Lab persisted: tested_total=%s promoted_total=%s generation=%s",
                        self.state.tested_total, self.state.promoted_total, self.state.generation
                    )
                    self._summary = {
                        "symbols": list(bars_by_symbol.keys()),
                        "bars": {s: len(v) for s, v in bars_by_symbol.items()},
                        "candidates_tested": self.state.tested_total,
                        "promoted_count": len(promoted),
                        "target_promoted": self.settings.lab_target_promoted,
                        "generation": self.state.generation,
                        "best_candidate": promoted[0] if promoted else (results[0] if results else None),
                        "cost_bps_per_side": self.settings.lab_cost_bps,
                        "stress_cost_multiplier": self.settings.lab_stress_cost_multiplier,
                        "method": (
                            "continuous deterministic candidate search; chronological 70/30 holdout; "
                            "next-bar-open fills; long-only; per-symbol robustness filter"
                        ),
                    }
                    await asyncio.sleep(0)

                    if (
                        self.settings.lab_target_promoted > 0
                        and len(promoted) >= self.settings.lab_target_promoted
                    ):
                        self.state.stage = "target_reached"
                        self.state.message = (
                            f"Target reached: {len(promoted)} robust candidates found after "
                            f"{self.state.tested_total} tests"
                        )
                        return

                if not self.settings.lab_continuous:
                    self.state.stage = "completed"
                    self.state.message = (
                        f"Batch completed: {self.state.tested_total} candidates tested"
                    )
                    return

        except asyncio.CancelledError:
            self.state.stage = "stopped"
            self.state.message = "Strategy Lab stopped"
            raise
        except Exception as exc:
            self.state.last_error = str(exc)
            self.state.stage = "error"
            self.state.message = str(exc)
        finally:
            self.state.running = False
            self.state.completed_at = datetime.now(timezone.utc).isoformat()


def candidate_signature(candidate: Candidate) -> str:
    import json
    return candidate.strategy + ":" + json.dumps(candidate.params, sort_keys=True, separators=(",", ":"))


def candidate_stream():
    """Yield an open-ended, deterministic stream of unique parameter combinations."""
    generation = 0
    while True:
        generation += 1

        # Momentum: gradually widen the search instead of random data-mining.
        slow_values = [10 + generation, 14 + generation * 2, 20 + generation * 3, 30 + generation * 4]
        for slow in slow_values:
            fast_values = sorted({2, 3, 4, 5, max(2, slow // 4), max(2, slow // 3)})
            for fast in fast_values:
                if fast >= slow:
                    continue
                for entry_bps in (3.0, 5.0, 8.0, 12.0, 18.0, 25.0):
                    for hold in (6, 12, 18, 24, 36, 48):
                        yield Candidate("momentum", {
                            "fast": fast,
                            "slow": slow,
                            "entry_bps": entry_bps,
                            "max_hold": hold,
                        })

        # Mean reversion: widen lookback and entry strictness gradually.
        windows = [8 + generation, 12 + generation * 2, 20 + generation * 3, 30 + generation * 4]
        for window in windows:
            for z_entry in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25):
                for z_exit in (0.0, 0.15, 0.30, 0.50):
                    if z_exit >= z_entry:
                        continue
                    for hold in (6, 12, 18, 24, 36, 48):
                        yield Candidate("mean_reversion", {
                            "window": window,
                            "z_entry": z_entry,
                            "z_exit": z_exit,
                            "max_hold": hold,
                        })


def candidate_grid() -> List[Candidate]:
    candidates: List[Candidate] = []
    for fast, slow, entry, hold in [
        (3, 10, 5.0, 12),
        (4, 12, 8.0, 15),
        (5, 20, 8.0, 20),
        (8, 24, 10.0, 24),
        (10, 30, 12.0, 30),
    ]:
        candidates.append(Candidate("momentum", {
            "fast": fast,
            "slow": slow,
            "entry_bps": entry,
            "max_hold": hold,
        }))

    for window, z_entry, z_exit, hold in [
        (10, 1.0, 0.15, 12),
        (15, 1.25, 0.20, 18),
        (20, 1.5, 0.25, 24),
        (30, 1.5, 0.25, 30),
        (40, 1.75, 0.30, 36),
    ]:
        candidates.append(Candidate("mean_reversion", {
            "window": window,
            "z_entry": z_entry,
            "z_exit": z_exit,
            "max_hold": hold,
        }))
    return candidates


def evaluate_candidate(
    candidate: Candidate,
    bars_by_symbol: Dict[str, List[dict]],
    cost_bps: float,
    stress_cost_multiplier: float,
    min_oos_trades: int,
    min_profit_factor: float = 1.15,
    max_drawdown_pct: float = 6.0,
    min_positive_symbol_ratio: float = 0.60,
) -> dict:
    train_trades: List[dict] = []
    oos_trades: List[dict] = []
    stress_oos_trades: List[dict] = []
    per_symbol: Dict[str, dict] = {}

    for symbol, bars in bars_by_symbol.items():
        split = max(2, int(len(bars) * 0.70))
        train = bars[:split]
        test = bars[split:]

        symbol_train = simulate(candidate, symbol, train, cost_bps)
        symbol_oos = simulate(candidate, symbol, test, cost_bps)
        symbol_stress = simulate(candidate, symbol, test, cost_bps * stress_cost_multiplier)
        train_trades.extend(symbol_train)
        oos_trades.extend(symbol_oos)
        stress_oos_trades.extend(symbol_stress)
        per_symbol[symbol] = {
            "oos": metrics(symbol_oos),
            "stress_oos": metrics(symbol_stress),
        }

    train_metrics = metrics(train_trades)
    oos_metrics = metrics(oos_trades)
    stress_metrics = metrics(stress_oos_trades)
    positive_symbols = sum(
        1 for row in per_symbol.values()
        if row["oos"]["trades"] > 0 and row["oos"]["expectancy_bps"] > 0
    )
    symbol_count = max(1, len(per_symbol))
    positive_symbol_ratio = positive_symbols / symbol_count

    reasons: List[str] = []
    if train_metrics["expectancy_bps"] <= 0:
        reasons.append("negative in-sample expectancy")
    if oos_metrics["trades"] < min_oos_trades:
        reasons.append(f"fewer than {min_oos_trades} out-of-sample trades")
    if oos_metrics["expectancy_bps"] <= 0:
        reasons.append("negative out-of-sample expectancy")
    if oos_metrics["profit_factor"] < min_profit_factor:
        reasons.append(f"out-of-sample profit factor below {min_profit_factor:.2f}")
    if oos_metrics["max_drawdown_pct"] > max_drawdown_pct:
        reasons.append(f"out-of-sample drawdown above {max_drawdown_pct:.1f}%")
    if positive_symbol_ratio < min_positive_symbol_ratio:
        reasons.append(
            f"positive on only {positive_symbols}/{len(per_symbol)} symbols "
            f"(< {min_positive_symbol_ratio:.0%})"
        )
    if stress_metrics["expectancy_bps"] <= 0:
        reasons.append("fails stressed transaction-cost test")

    return {
        "strategy": candidate.strategy,
        "params": candidate.params,
        "promoted": not reasons,
        "rejection_reasons": reasons,
        "train": train_metrics,
        "oos": oos_metrics,
        "stress_oos": stress_metrics,
        "positive_symbol_ratio": round(positive_symbol_ratio, 3),
        "positive_symbols": positive_symbols,
        "symbol_count": len(per_symbol),
        "per_symbol": per_symbol,
    }


def simulate(
    candidate: Candidate,
    symbol: str,
    bars: List[dict],
    cost_bps: float,
) -> List[dict]:
    if candidate.strategy == "momentum":
        return _simulate_momentum(candidate, symbol, bars, cost_bps)
    if candidate.strategy == "mean_reversion":
        return _simulate_mean_reversion(candidate, symbol, bars, cost_bps)
    raise ValueError(f"Unknown strategy: {candidate.strategy}")


def _simulate_momentum(
    candidate: Candidate,
    symbol: str,
    bars: List[dict],
    cost_bps: float,
) -> List[dict]:
    p = candidate.params
    slow = int(p["slow"])
    trades: List[dict] = []
    i = slow

    while i < len(bars) - 1:
        history = [float(x["c"]) for x in bars[i - slow:i]]
        fast_ma = mean(history[-int(p["fast"]):])
        slow_ma = mean(history)
        edge_bps = ((fast_ma / slow_ma) - 1.0) * 10_000 if slow_ma else 0.0
        if edge_bps < float(p["entry_bps"]):
            i += 1
            continue

        entry_idx = i
        entry_price = float(bars[entry_idx]["o"])
        exit_idx = min(entry_idx + int(p["max_hold"]), len(bars) - 1)

        for j in range(entry_idx + 1, exit_idx + 1):
            trailing = [float(x["c"]) for x in bars[max(0, j - slow):j]]
            if len(trailing) < slow:
                continue
            f = mean(trailing[-int(p["fast"]):])
            s = mean(trailing)
            if s and ((f / s) - 1.0) * 10_000 <= 0:
                exit_idx = j
                break

        exit_price = float(bars[exit_idx]["o"])
        trades.append(_trade(symbol, bars[entry_idx], bars[exit_idx], entry_price, exit_price, cost_bps))
        i = exit_idx + 1
    return trades


def _simulate_mean_reversion(
    candidate: Candidate,
    symbol: str,
    bars: List[dict],
    cost_bps: float,
) -> List[dict]:
    p = candidate.params
    window = int(p["window"])
    trades: List[dict] = []
    i = window

    while i < len(bars) - 1:
        history = [float(x["c"]) for x in bars[i - window:i]]
        mu = mean(history)
        sigma = pstdev(history)
        if sigma <= 0:
            i += 1
            continue
        z = (float(bars[i - 1]["c"]) - mu) / sigma
        if z > -float(p["z_entry"]):
            i += 1
            continue

        entry_idx = i
        entry_price = float(bars[entry_idx]["o"])
        exit_idx = min(entry_idx + int(p["max_hold"]), len(bars) - 1)

        for j in range(entry_idx + 1, exit_idx + 1):
            trailing = [float(x["c"]) for x in bars[max(0, j - window):j]]
            if len(trailing) < window:
                continue
            mu_j = mean(trailing)
            sigma_j = pstdev(trailing)
            if sigma_j <= 0:
                continue
            z_j = (float(bars[j - 1]["c"]) - mu_j) / sigma_j
            if z_j >= -float(p["z_exit"]):
                exit_idx = j
                break

        exit_price = float(bars[exit_idx]["o"])
        trades.append(_trade(symbol, bars[entry_idx], bars[exit_idx], entry_price, exit_price, cost_bps))
        i = exit_idx + 1
    return trades


def _trade(
    symbol: str,
    entry_bar: dict,
    exit_bar: dict,
    entry_price: float,
    exit_price: float,
    cost_bps: float,
) -> dict:
    gross = (exit_price / entry_price) - 1.0 if entry_price else 0.0
    net = gross - (2.0 * cost_bps / 10_000.0)
    return {
        "symbol": symbol,
        "entry_time": entry_bar.get("t"),
        "exit_time": exit_bar.get("t"),
        "gross_return": gross,
        "net_return": net,
    }


def metrics(trades: List[dict]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "expectancy_bps": 0.0,
            "profit_factor": 0.0,
            "compounded_trade_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
        }

    ordered = sorted(trades, key=lambda x: str(x.get("exit_time") or ""))
    returns = [float(t["net_return"]) for t in ordered]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= max(0.000001, 1.0 + r)
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)

    return {
        "trades": len(returns),
        "win_rate_pct": round(100.0 * len(wins) / len(returns), 2),
        "expectancy_bps": round(mean(returns) * 10_000.0, 3),
        "profit_factor": round(min(pf, 999.0), 3),
        "compounded_trade_return_pct": round((equity - 1.0) * 100.0, 3),
        "max_drawdown_pct": round(max_dd * 100.0, 3),
    }
