from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Dict, List, Optional

from alpaca_client import AlpacaClient
from config import Settings


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


class StrategyLab:
    def __init__(self, settings: Settings, client: AlpacaClient):
        self.settings = settings
        self.client = client
        self.state = LabState()
        self._task: Optional[asyncio.Task] = None
        self._results: List[dict] = []
        self._summary: dict = {}

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

            candidates = candidate_grid()
            self.state.total = len(candidates)
            self.state.stage = "testing"
            self.state.message = f"Testing {len(candidates)} strategy candidates"
            if not bars_by_symbol:
                raise RuntimeError("No usable historical bars returned for Strategy Lab")

            results: List[dict] = []
            for idx, candidate in enumerate(candidates, start=1):
                result = evaluate_candidate(
                    candidate,
                    bars_by_symbol,
                    cost_bps=self.settings.lab_cost_bps,
                    stress_cost_multiplier=self.settings.lab_stress_cost_multiplier,
                    min_oos_trades=self.settings.lab_min_oos_trades,
                )
                results.append(result)
                self.state.progress = idx
                self.state.message = f"Tested {idx}/{len(candidates)} candidates"
                await asyncio.sleep(0)

            results.sort(
                key=lambda row: (
                    bool(row["promoted"]),
                    row["oos"]["expectancy_bps"],
                    row["oos"]["profit_factor"],
                ),
                reverse=True,
            )
            self._results = results
            promoted = [r for r in results if r["promoted"]]
            self.state.stage = "completed"
            self.state.message = "Strategy Lab completed"
            self._summary = {
                "symbols": list(bars_by_symbol.keys()),
                "bars": {s: len(v) for s, v in bars_by_symbol.items()},
                "candidates_tested": len(results),
                "promoted_count": len(promoted),
                "best_candidate": promoted[0] if promoted else None,
                "cost_bps_per_side": self.settings.lab_cost_bps,
                "stress_cost_multiplier": self.settings.lab_stress_cost_multiplier,
                "method": "chronological 70/30 holdout; next-bar-open fills; long-only; no overlapping position per symbol",
            }
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
) -> dict:
    train_trades: List[dict] = []
    oos_trades: List[dict] = []
    stress_oos_trades: List[dict] = []

    for symbol, bars in bars_by_symbol.items():
        split = max(2, int(len(bars) * 0.70))
        train = bars[:split]
        test = bars[split:]

        train_trades.extend(simulate(candidate, symbol, train, cost_bps))
        oos_trades.extend(simulate(candidate, symbol, test, cost_bps))
        stress_oos_trades.extend(
            simulate(candidate, symbol, test, cost_bps * stress_cost_multiplier)
        )

    train_metrics = metrics(train_trades)
    oos_metrics = metrics(oos_trades)
    stress_metrics = metrics(stress_oos_trades)

    reasons: List[str] = []
    if train_metrics["expectancy_bps"] <= 0:
        reasons.append("negative in-sample expectancy")
    if oos_metrics["trades"] < min_oos_trades:
        reasons.append(f"fewer than {min_oos_trades} out-of-sample trades")
    if oos_metrics["expectancy_bps"] <= 0:
        reasons.append("negative out-of-sample expectancy")
    if oos_metrics["profit_factor"] < 1.10:
        reasons.append("out-of-sample profit factor below 1.10")
    if oos_metrics["max_drawdown_pct"] > 8.0:
        reasons.append("out-of-sample drawdown above 8%")
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
