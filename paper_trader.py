from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Dict, List, Optional

from config import Settings
from ctrader_client import CTraderClient
from forex_store import ForexStrategyStore
from forex_backtest import FOREX_EVALUATION_POLICY_VERSION
from paper_store import PaperTradingStore
from research_store import ResearchStore
from strategy_lab import RESEARCH_POLICY_VERSION, _regime_ensemble_signal
from market_filters import entry_allowed

log = logging.getLogger("microtrader.paper")


@dataclass
class PaperTradingState:
    running: bool = False
    stage: str = "idle"
    message: str = "Waiting for promoted strategies"
    strategies: int = 0
    retiring_strategies: int = 0
    paused_strategies: int = 0
    expected_portfolio_trades_per_day: float = 0.0
    portfolio_target_trades_per_day: float = 10.0
    portfolio_frequency_ready: bool = False
    last_cycle_at: Optional[str] = None
    last_error: Optional[str] = None


def _loss_streak_limit(strategy: str, normal_limit: int, asymmetric_limit: int) -> int:
    return int(asymmetric_limit) if str(strategy) == "asymmetric_breakout" else int(normal_limit)


def _paper_id(row: dict) -> str:
    params = {
        k: v for k, v in dict(row.get("params") or {}).items()
        if k not in {"_phase", "risk_eur"}
        and not k.startswith("_research_")
    }
    raw = json.dumps(
        {
            "strategy": row.get("strategy"),
            "params": params,
            "pairs": sorted(row.get("pairs") or []),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "paper-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _dt(value) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _atr(bars: List[dict], i: int, window: int = 14) -> float:
    vals: List[float] = []
    for j in range(max(1, i - window), i):
        high = float(bars[j]["h"])
        low = float(bars[j]["l"])
        prev = float(bars[j - 1]["c"])
        vals.append(max(high - low, abs(high - prev), abs(low - prev)))
    return mean(vals) if vals else 0.0


def _exit_management(
    params: dict,
    *,
    entry: float,
    risk_distance: float,
    cost_bps: float,
) -> tuple[float | None, float | None]:
    """Trade-management levels expressed as NET R after modeled costs."""
    mode = str(params.get("exit_mode") or "baseline")
    trigger_r = params.get("management_trigger_r")
    desired_net_r = params.get("management_lock_net_r")

    if trigger_r is not None and desired_net_r is not None:
        trigger_r = float(trigger_r)
        desired_net_r = max(0.0, float(desired_net_r))
    elif mode == "breakeven_2r":
        trigger_r = 2.0
        desired_net_r = max(0.0, float(params.get("breakeven_buffer_r", 0.05)))
    elif mode == "protect_2r_025r":
        trigger_r, desired_net_r = 2.0, 0.25
    elif mode == "lock_2r_05r":
        trigger_r, desired_net_r = 2.0, 0.50
    else:
        return None, None

    risk_pct = risk_distance / entry if entry > 0 else 0.0
    roundtrip_cost_pct = 2.0 * float(cost_bps) / 10_000.0
    cost_r = roundtrip_cost_pct / risk_pct if risk_pct > 0 else 0.0
    return float(trigger_r), cost_r + float(desired_net_r)


def _max_loss_stop_price(
    *,
    entry: float,
    risk_distance: float,
    direction: int,
    cost_bps: float,
    max_loss_r: float = 1.0,
) -> float:
    """Price that limits NET modeled loss to max_loss_r after round-trip costs."""
    if entry <= 0 or risk_distance <= 0 or direction not in (-1, 1):
        return entry
    risk_pct = risk_distance / entry
    roundtrip_cost_pct = 2.0 * float(cost_bps) / 10_000.0
    desired_net = -abs(float(max_loss_r)) * risk_pct
    signed_gross = desired_net + roundtrip_cost_pct
    return entry * (1.0 + signed_gross / float(direction))


def _rolling_vwap(bars: List[dict], a: int, b: int) -> float:
    sample = bars[a:b]
    if not sample:
        return 0.0
    pv = 0.0
    vol = 0.0
    for x in sample:
        v = float(x.get("v") or 0.0)
        typ = (float(x["h"]) + float(x["l"]) + float(x["c"])) / 3.0
        pv += typ * v
        vol += v
    return pv / vol if vol > 0 else mean(float(x["c"]) for x in sample)


def _forex_signal(strategy: str, params: dict, bars: List[dict], i: int) -> int:
    if i < 2:
        return 0
    if strategy == "range_reversal":
        window = int(params.get("window", 30))
        if i < window:
            return 0
        closes = [float(x["c"]) for x in bars[i-window:i]]
        mu = mean(closes)
        sigma = pstdev(closes)
        if sigma <= 0:
            return 0
        z = (closes[-1] - mu) / sigma
        z_entry = float(params.get("z_entry", 1.8))
        return 1 if z <= -z_entry else (-1 if z >= z_entry else 0)

    if strategy == "trend_pullback":
        fast = int(params.get("fast", 10))
        slow = int(params.get("slow", 30))
        if i < slow:
            return 0
        closes = [float(x["c"]) for x in bars[i-slow:i]]
        fast_ma = mean(closes[-fast:])
        slow_ma = mean(closes)
        prev_close = closes[-1]
        if fast_ma > slow_ma and slow_ma < prev_close <= fast_ma:
            return 1
        if fast_ma < slow_ma and fast_ma <= prev_close < slow_ma:
            return -1
        return 0

    if strategy == "asymmetric_breakout":
        window = int(params.get("window", 20))
        if i < window + 2:
            return 0
        prior = bars[i-window-1:i-1]
        prev_close = float(bars[i-1]["c"])
        high_break = max(float(x["h"]) for x in prior)
        low_break = min(float(x["l"]) for x in prior)
        return 1 if prev_close > high_break else (-1 if prev_close < low_break else 0)

    return 0


def _research_signal(strategy: str, params: dict, bars: List[dict], i: int) -> int:
    """Forward signal equivalent of the Research Lab candidate families."""
    if i < 2:
        return 0

    if strategy == "momentum":
        slow = int(params["slow"])
        fast = int(params["fast"])
        if i < slow:
            return 0
        history = [float(x["c"]) for x in bars[i-slow:i]]
        fast_ma = mean(history[-fast:])
        slow_ma = mean(history)
        edge_bps = ((fast_ma / slow_ma) - 1.0) * 10_000 if slow_ma else 0.0
        threshold = float(params["entry_bps"])
        return 1 if edge_bps >= threshold else (-1 if edge_bps <= -threshold else 0)

    if strategy == "mean_reversion":
        window = int(params["window"])
        if i < window:
            return 0
        history = [float(x["c"]) for x in bars[i-window:i]]
        mu = mean(history)
        sigma = pstdev(history)
        if sigma <= 0:
            return 0
        z = (history[-1] - mu) / sigma
        threshold = float(params["z_entry"])
        return 1 if z <= -threshold else (-1 if z >= threshold else 0)

    if strategy == "breakout":
        window = int(params["window"])
        if i < window + 1:
            return 0
        prior = bars[i-window-1:i-1]
        if not prior:
            return 0
        buf = float(params.get("buffer_bps", 0.0)) / 10_000.0
        px = float(bars[i-1]["c"])
        high_level = max(float(x["h"]) for x in prior) * (1.0 + buf)
        low_level = min(float(x["l"]) for x in prior) * (1.0 - buf)
        return 1 if px > high_level else (-1 if px < low_level else 0)

    if strategy == "extreme_reversal":
        window = int(params["window"])
        if i < window + 1:
            return 0
        rets = [
            float(bars[j]["c"]) / float(bars[j-1]["c"]) - 1.0
            for j in range(i-window, i)
        ]
        sigma = pstdev(rets) if len(rets) > 1 else 0.0
        if sigma <= 0:
            return 0
        threshold = float(params["shock_z"]) * sigma
        return 1 if rets[-1] <= -threshold else (-1 if rets[-1] >= threshold else 0)

    if strategy == "volatility_breakout":
        window = int(params["window"])
        if i < window + 1:
            return 0
        ranges = [float(x["h"]) - float(x["l"]) for x in bars[i-window:i]]
        avg_range = mean(ranges) if ranges else 0.0
        cur = float(bars[i-1]["h"]) - float(bars[i-1]["l"])
        if avg_range <= 0 or cur < avg_range * float(params["vol_mult"]):
            return 0
        o = float(bars[i-1]["o"])
        cl = float(bars[i-1]["c"])
        return 1 if cl > o else (-1 if cl < o else 0)

    if strategy == "trend_pullback":
        slow = int(params["slow"])
        fast = int(params["fast"])
        if i < slow:
            return 0
        closes = [float(x["c"]) for x in bars[i-slow:i]]
        f = mean(closes[-fast:])
        s = mean(closes)
        sd = pstdev(closes)
        z = (closes[-1] - f) / sd if sd > 0 else 0.0
        threshold = float(params["pullback_z"])
        return 1 if (f > s and z <= -threshold) else (-1 if (f < s and z >= threshold) else 0)

    if strategy == "vwap_reversion":
        window = int(params["window"])
        if i < window:
            return 0
        vw = _rolling_vwap(bars, i-window, i)
        closes = [float(x["c"]) for x in bars[i-window:i]]
        sd = pstdev(closes)
        z = (closes[-1] - vw) / sd if sd > 0 else 0.0
        threshold = float(params["z_entry"])
        return 1 if z <= -threshold else (-1 if z >= threshold else 0)

    if strategy == "vwap_momentum":
        window = int(params["window"])
        if i < window:
            return 0
        vw = _rolling_vwap(bars, i-window, i)
        px = float(bars[i-1]["c"])
        edge = ((px / vw) - 1.0) * 10_000 if vw > 0 else 0.0
        threshold = float(params["buffer_bps"])
        return 1 if edge >= threshold else (-1 if edge <= -threshold else 0)

    if strategy == "regime_ensemble":
        direction, _, _ = _regime_ensemble_signal(params, bars, i)
        return direction

    if strategy == "asymmetric_breakout":
        window = int(params["window"])
        if i < max(window + 1, 15):
            return 0
        prior = bars[i-window-1:i-1]
        if not prior:
            return 0
        prev = float(bars[i-1]["c"])
        high_break = max(float(x["h"]) for x in prior)
        low_break = min(float(x["l"]) for x in prior)
        return 1 if prev > high_break else (-1 if prev < low_break else 0)

    return 0


def _signal(strategy: str, params: dict, bars: List[dict], i: int) -> int:
    if str(params.get("_paper_source") or "forex") == "research":
        return _research_signal(strategy, params, bars, i)
    return _forex_signal(strategy, params, bars, i)


def _research_exit_reason(
    strategy: str,
    params: dict,
    bars: List[dict],
    i: int,
    direction: int,
) -> str | None:
    """Native Research Lab signal exits evaluated on fully closed bars."""
    j = i + 1
    if strategy == "momentum":
        slow = int(params["slow"])
        fast = int(params["fast"])
        if j < slow:
            return None
        trailing = [float(x["c"]) for x in bars[j-slow:j]]
        f = mean(trailing[-fast:])
        s = mean(trailing)
        edge = ((f / s) - 1.0) * 10_000 if s else 0.0
        if (direction > 0 and edge <= 0) or (direction < 0 and edge >= 0):
            return "research_signal_exit"

    if strategy == "mean_reversion":
        window = int(params["window"])
        if j < window:
            return None
        trailing = [float(x["c"]) for x in bars[j-window:j]]
        mu = mean(trailing)
        sigma = pstdev(trailing)
        if sigma <= 0:
            return None
        z = (trailing[-1] - mu) / sigma
        z_exit = float(params.get("z_exit", 0.0))
        if (direction > 0 and z >= -z_exit) or (direction < 0 and z <= z_exit):
            return "research_mean_exit"

    return None


def _causal_close_entry(
    strategy: str,
    params: dict,
    closed: List[dict],
    i: int,
    tf: int,
    *,
    default_min_volume_ratio: float = 0.70,
    default_volume_window: int = 50,
) -> dict | None:
    """Build a causal paper entry from a bar that has just fully closed.

    The decision uses the just-closed bar plus older bars only. The historical
    open is never used as a fill; the observable close is the simulated fill.
    """
    if i < 0 or i >= len(closed):
        return None
    bar = closed[i]
    entry_time = _dt(bar["t"]) + timedelta(minutes=tf)
    decision_bar = dict(bar)
    decision_bar["t"] = entry_time.isoformat()
    decision_bars = closed[: i + 1] + [decision_bar]
    decision_idx = len(decision_bars) - 1
    allowed, session_name, volume_ratio = entry_allowed(
        decision_bars,
        decision_idx,
        min_volume_ratio=float(
            params.get("min_volume_ratio", default_min_volume_ratio)
        ),
        volume_window=int(params.get("volume_window", default_volume_window)),
    )
    if not allowed:
        return None
    direction = _signal(strategy, params, closed, i + 1)
    if not direction:
        return None
    entry_model = None
    entry_regime = None
    if strategy == "regime_ensemble":
        _, entry_model, entry_regime = _regime_ensemble_signal(
            params, closed, i + 1
        )
    atr = _atr(closed, i + 1, 14)
    risk_distance = atr * float(params.get("stop_atr", 1.0))
    if risk_distance <= 0:
        return None
    return {
        "direction": direction,
        "entry_time": entry_time.isoformat(),
        "entry_price": float(bar["c"]),
        "risk_distance": risk_distance,
        "entry_session": session_name,
        "entry_volume_ratio": volume_ratio,
        "entry_model": entry_model,
        "entry_regime": entry_regime,
    }


class PaperTradingEngine:
    """Forward-only simulated execution for promoted Forex strategies.

    It never sends broker orders. New promoted configurations are frozen and
    tracked from EUR 50 using subsequent cTrader market bars only.
    """

    SUPPORTED = {
        "range_reversal", "trend_pullback", "asymmetric_breakout",
        "momentum", "mean_reversion", "breakout", "extreme_reversal",
        "volatility_breakout", "vwap_reversion", "vwap_momentum",
        "regime_ensemble",
    }

    def __init__(self, settings: Settings, client: CTraderClient):
        self.settings = settings
        self.client = client
        self.research_store = ForexStrategyStore(settings.database_url)
        self.validation_store = ResearchStore(settings.database_url)
        self.store = PaperTradingStore(settings.database_url)
        self.state = PaperTradingState()
        self._task: Optional[asyncio.Task] = None

    def public_state(self) -> dict:
        payload = asdict(self.state)
        payload["start_balance_eur"] = self.settings.paper_start_balance
        payload["poll_seconds"] = self.settings.paper_poll_seconds
        payload["paper_risk_eur"] = self.settings.paper_risk_eur
        payload["max_loss_streak"] = self.settings.strategy_max_loss_streak
        payload["max_loss_streak_asymmetric"] = self.settings.strategy_max_loss_streak_asymmetric
        payload["execution"] = (
            "simulated-only; causal close execution; signals use only completed bars; "
            "no broker orders"
        )
        payload["entry_timing"] = "signal at completed-bar close; fill at that observable close; management starts next bar"
        payload["allowed_timeframes_min"] = [1, 5]
        payload["entry_sessions"] = ["London", "New York"]
        payload["min_volume_ratio"] = self.settings.strategy_min_volume_ratio
        payload["strategy_min_trades_per_day"] = self.settings.strategy_min_trades_per_day
        payload["strategy_preferred_trades_per_day"] = self.settings.strategy_preferred_trades_per_day
        payload["strategy_target_trades_per_day"] = self.settings.strategy_target_trades_per_day
        payload["portfolio_min_trades_per_day"] = self.settings.portfolio_min_trades_per_day
        payload["frequency_policy"] = (
            "hard >=3/day per strategy; preference 5-10/day; "
            "portfolio target >=10/day combined"
        )
        payload["paper_sources"] = ["forex_promoted", "research_promoted_17of17_validated"]
        return payload

    async def start(self) -> None:
        if self.state.running:
            return
        if not self.client.api_ready:
            self.state.stage = "waiting_credentials"
            self.state.message = "Paper trading waits for cTrader demo market-data authorization."
            return
        self.state = PaperTradingState(
            running=True,
            stage="starting",
            message="Starting promoted-strategy paper trading",
        )
        self._task = asyncio.create_task(self._run(), name="microtrader-paper")

    async def stop(self) -> None:
        self.state.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def results(self) -> dict:
        return {
            "state": self.public_state(),
            **(await self.store.dashboard()),
        }

    async def _discover(self) -> List[dict]:
        forex_promoted = await self.research_store.load_promoted(
            limit=100,
            min_trades_per_day=self.settings.strategy_min_trades_per_day,
            evaluation_policy_version=FOREX_EVALUATION_POLICY_VERSION,
        )
        research_validated = await self.validation_store.load_paper_eligible_validations(
            research_policy_version=RESEARCH_POLICY_VERSION,
            min_trades_per_day=self.settings.strategy_min_trades_per_day,
            limit=100,
        )

        eligible_ids: set[str] = set()
        portfolio_sources: set[str] = set()
        expected_portfolio_trades_per_day = 0.0

        # Native Forex Lab promotions.
        for row in forex_promoted:
            strategy = str(row.get("strategy") or "")
            if strategy not in self.SUPPORTED:
                continue
            base_tf = int(
                row.get("timeframe_min")
                or (row.get("params") or {}).get("timeframe_min")
                or 0
            )
            if base_tf not in (1, 5):
                continue
            base_params = {
                k: v for k, v in dict(row.get("params") or {}).items()
                if k != "_phase"
            }
            base_params["_paper_source"] = "forex"
            base_params["risk_eur"] = float(self.settings.paper_risk_eur)
            source_key = _paper_id({**row, "params": base_params})
            if source_key not in portfolio_sources:
                portfolio_sources.add(source_key)
                expected_portfolio_trades_per_day += float(
                    (row.get("oos") or {}).get("avg_trades_per_day") or 0.0
                )

            variants = [base_params]
            for params in variants:
                variant_row = {**row, "params": params}
                paper_id = _paper_id(variant_row)
                eligible_ids.add(paper_id)
                await self.store.ensure_strategy(
                    paper_id=paper_id,
                    promoted_run_id=str(row.get("run_id") or ""),
                    strategy=strategy,
                    params=params,
                    pairs=list(row.get("pairs") or self.settings.forex_pairs),
                    timeframe_min=base_tf,
                    start_balance=float(self.settings.paper_start_balance),
                )

        # Research candidates enter Paper only after BOTH Research promotion
        # and a completed exact 17/17 validation under the current policy.
        for row in research_validated:
            strategy = str(row.get("strategy") or "")
            if strategy not in self.SUPPORTED:
                continue
            params = {
                k: v for k, v in dict(row.get("params") or {}).items()
                if k not in {"_phase", "_policy_version"}
            }
            tf = int(params.get("timeframe_min") or 0)
            if tf not in (1, 5):
                continue
            params["_paper_source"] = "research"
            params["_paper_exit_model"] = "research_native"
            params["risk_eur"] = float(self.settings.paper_risk_eur)
            params.setdefault("stop_atr", 1.0)
            params["_research_max_loss_streak"] = int(
                (row.get("oos") or {}).get("max_loss_streak") or 0
            )
            params["_research_max_win_streak"] = int(
                (row.get("oos") or {}).get("max_win_streak") or 0
            )

            source_row = {
                "strategy": strategy,
                "params": params,
                "pairs": list(self.settings.forex_pairs),
            }
            paper_id = _paper_id(source_row)
            eligible_ids.add(paper_id)
            if paper_id not in portfolio_sources:
                portfolio_sources.add(paper_id)
                expected_portfolio_trades_per_day += float(
                    (row.get("oos") or {}).get("avg_trades_per_day") or 0.0
                )
            await self.store.ensure_strategy(
                paper_id=paper_id,
                promoted_run_id="research-validation:" + str(row.get("candidate_signature") or ""),
                strategy=strategy,
                params=params,
                pairs=list(self.settings.forex_pairs),
                timeframe_min=tf,
                start_balance=float(self.settings.paper_start_balance),
            )

        strategies = await self.store.list_strategies()
        for row in strategies:
            paper_id = str(row.get("paper_id") or "")
            status = str(row.get("status") or "")
            strategy = str(row.get("strategy") or "")
            open_positions = await self.store.list_positions(paper_id)
            loss_streak, max_loss_streak = await self.store.loss_streak_stats(paper_id)
            loss_streak_limit = _loss_streak_limit(
                strategy,
                self.settings.strategy_max_loss_streak,
                self.settings.strategy_max_loss_streak_asymmetric,
            )
            row["paper_current_loss_streak"] = loss_streak
            row["paper_max_loss_streak"] = max_loss_streak
            row["paper_loss_streak_limit"] = loss_streak_limit

            # A breached forward-loss streak is a sticky review state. Existing
            # positions keep being managed, but no new entries are allowed.
            if (
                paper_id in eligible_ids
                and status in {"active", "frequency_rejected", "retiring"}
                and max_loss_streak > loss_streak_limit
            ):
                if status != "review_pause":
                    await self.store.set_status(paper_id, "review_pause")
                row["status"] = "review_pause"
                status = "review_pause"

            if paper_id in eligible_ids and status in {"frequency_rejected", "retiring"}:
                await self.store.set_status(paper_id, "active")
                row["status"] = "active"
            elif paper_id not in eligible_ids:
                if open_positions and status in {"active", "frequency_rejected", "retiring", "review_pause"}:
                    if status != "retiring":
                        await self.store.set_status(paper_id, "retiring")
                    row["status"] = "retiring"
                elif status in {"active", "retiring"}:
                    await self.store.set_status(paper_id, "frequency_rejected")
                    row["status"] = "frequency_rejected"

        active = [row for row in strategies if str(row.get("status") or "") == "active"]
        retiring = [row for row in strategies if str(row.get("status") or "") == "retiring"]
        paused = [row for row in strategies if str(row.get("status") or "") == "review_pause"]
        self.state.strategies = len(active)
        self.state.retiring_strategies = len(retiring)
        self.state.paused_strategies = len(paused)
        self.state.expected_portfolio_trades_per_day = round(
            expected_portfolio_trades_per_day, 3
        )
        self.state.portfolio_target_trades_per_day = float(
            self.settings.portfolio_min_trades_per_day
        )
        self.state.portfolio_frequency_ready = (
            expected_portfolio_trades_per_day
            >= float(self.settings.portfolio_min_trades_per_day)
        )
        log.info(
            "Paper discovery: forex_promoted=%s research_validated=%s active=%s "
            "combined_expected_tpd=%.2f hard_per_strategy=%.1f portfolio_target=%.1f",
            len(forex_promoted),
            len(research_validated),
            len(active),
            expected_portfolio_trades_per_day,
            float(self.settings.strategy_min_trades_per_day),
            float(self.settings.portfolio_min_trades_per_day),
        )
        return strategies

    async def _run(self) -> None:
        try:
            await self.store.init()
            await self.research_store.init()
            await self.validation_store.init()
            while self.state.running:
                try:
                    strategies = await self._discover()
                    if not strategies:
                        self.state.stage = "waiting_promoted"
                        self.state.message = "No promoted strategies yet."
                    else:
                        self.state.stage = (
                            "paper_trading"
                            if self.state.portfolio_frequency_ready
                            else "building_portfolio"
                        )
                        self.state.message = (
                            f"Paper trading {self.state.strategies} active promoted/validated "
                            f"strateg{'y' if self.state.strategies==1 else 'ies'} from €{self.settings.paper_start_balance:.0f}; "
                            f"strategy hard minimum {self.settings.strategy_min_trades_per_day:g}/day; "
                            f"combined portfolio target {self.state.expected_portfolio_trades_per_day:.1f}/"
                            f"{self.settings.portfolio_min_trades_per_day:g} trades/day "
                            f"(not a hard per-strategy requirement)."
                        )
                        for row in strategies:
                            if not self.state.running:
                                break
                            status = str(row.get("status") or "")
                            if status not in {"active", "retiring", "review_pause"}:
                                continue
                            try:
                                await self._process_strategy(
                                    row,
                                    allow_entries=(status == "active"),
                                )
                            except Exception as exc:
                                await self.store.set_error(str(row["paper_id"]), str(exc))
                                log.exception("Paper strategy failed: %s", row.get("paper_id"))
                    self.state.last_error = None
                    self.state.last_cycle_at = datetime.now(timezone.utc).isoformat()
                except Exception as exc:
                    self.state.stage = "error"
                    self.state.last_error = str(exc)
                    self.state.message = str(exc)
                    log.exception("Paper trading cycle failed")
                await asyncio.sleep(max(30, int(self.settings.paper_poll_seconds)))
        except asyncio.CancelledError:
            raise
        finally:
            self.state.running = False

    async def _process_strategy(self, row: dict, *, allow_entries: bool = True) -> None:
        paper_id = str(row["paper_id"])
        strategy = str(row["strategy"])
        strategy_status = str(row.get("status") or "")
        params = dict(row.get("params") or {})
        pairs = list(row.get("pairs") or self.settings.forex_pairs)
        tf = int(row.get("timeframe_min") or params.get("timeframe_min") or 1)
        cursors = dict(row.get("last_bar_times") or {})
        positions = {p["pair"]: p for p in await self.store.list_positions(paper_id)}

        for pair in pairs:
            bars = await self.client.historical_bars(
                pair,
                timeframe_min=tf,
                max_bars=600,
                lookback_days=7,
            )
            if len(bars) < 50:
                continue

            # Only bars whose interval is fully closed are eligible.
            cutoff = datetime.now(timezone.utc) - timedelta(minutes=tf)
            closed = [b for b in bars if _dt(b["t"]) <= cutoff]
            if len(closed) < 40:
                continue

            cursor = str(cursors.get(pair) or "")
            if not cursor:
                # Strict forward start: do not retroactively paper-trade data
                # from before this strategy was enrolled.
                await self.store.set_cursor(paper_id, pair, str(closed[-1]["t"]))
                cursors[pair] = str(closed[-1]["t"])
                continue

            new_indices = [
                i for i, bar in enumerate(closed)
                if str(bar.get("t") or "") > cursor
            ]
            for i in new_indices:
                bar = closed[i]
                pos = positions.get(pair)
                if pos:
                    exited = await self._manage_existing(
                        paper_id, pair, pos, bar, params,
                        strategy=strategy, bars=closed, index=i,
                    )
                    if exited:
                        positions.pop(pair, None)
                    else:
                        pos = dict(pos)
                        pos["bars_held"] = int(pos.get("bars_held", 0)) + 1
                        await self.store.upsert_position(paper_id, pair, pos)
                        positions[pair] = pos
                elif allow_entries:
                    decision = _causal_close_entry(
                        strategy,
                        params,
                        closed,
                        i,
                        tf,
                        default_min_volume_ratio=self.settings.strategy_min_volume_ratio,
                        default_volume_window=self.settings.strategy_volume_window,
                    )
                    if decision:
                        direction = int(decision["direction"])
                        entry = float(decision["entry_price"])
                        risk_distance = float(decision["risk_distance"])
                        target_r = float(params.get("target_r", 2.0))
                        risk_eur = float(self.settings.paper_risk_eur)
                        stop = _max_loss_stop_price(
                            entry=entry,
                            risk_distance=risk_distance,
                            direction=direction,
                            cost_bps=float(self.settings.forex_cost_bps),
                            max_loss_r=1.0,
                        )
                        target = (
                            entry + risk_distance * target_r
                            if direction > 0
                            else entry - risk_distance * target_r
                        )
                        new_pos = {
                            **decision,
                            "stop_price": stop,
                            "target_price": target,
                            "bars_held": 0,
                            "risk_eur": risk_eur,
                            "execution_timing": "completed_bar_close",
                        }
                        # The position did not exist during this finished bar.
                        # Stop/target management starts with the next bar.
                        await self.store.upsert_position(paper_id, pair, new_pos)
                        positions[pair] = new_pos

                await self.store.set_cursor(paper_id, pair, str(bar["t"]))
                cursors[pair] = str(bar["t"])

        await self.store.touch_daily(paper_id, len(positions))
        # Retiring means "no longer eligible; manage open positions only".
        # Once those positions are gone it may become frequency_rejected.
        # review_pause is intentionally sticky and must never be overwritten here.
        if not allow_entries and not positions and strategy_status == "retiring":
            await self.store.set_status(paper_id, "frequency_rejected")

    def _maybe_protect_stop(self, pos: dict, bar: dict, params: dict) -> bool:
        trigger_r, lock_r = _exit_management(
            params,
            entry=float(pos["entry_price"]),
            risk_distance=float(pos["risk_distance"]),
            cost_bps=float(self.settings.forex_cost_bps),
        )
        if trigger_r is None or lock_r is None:
            return False
        direction = int(pos["direction"])
        entry = float(pos["entry_price"])
        risk_distance = float(pos["risk_distance"])
        current_stop = float(pos["stop_price"])
        protected_stop = (
            entry + risk_distance * lock_r
            if direction > 0
            else entry - risk_distance * lock_r
        )
        # Already at least this well protected.
        if direction > 0 and current_stop >= protected_stop:
            return False
        if direction < 0 and current_stop <= protected_stop:
            return False
        trigger_price = (
            entry + risk_distance * trigger_r
            if direction > 0
            else entry - risk_distance * trigger_r
        )
        high, low = float(bar["h"]), float(bar["l"])
        reached = high >= trigger_price if direction > 0 else low <= trigger_price
        if reached:
            pos["stop_price"] = protected_stop
            return True
        return False

    async def _manage_entry_bar(
        self, paper_id: str, pair: str, pos: dict, bar: dict, params: dict
    ) -> bool:
        direction = int(pos["direction"])
        low, high = float(bar["l"]), float(bar["h"])
        stop, target = float(pos["stop_price"]), float(pos["target_price"])
        if direction > 0 and low <= stop:
            await self._close(paper_id, pair, pos, bar, stop, "stop")
            return True
        if direction < 0 and high >= stop:
            await self._close(paper_id, pair, pos, bar, stop, "stop")
            return True
        if direction > 0 and high >= target:
            await self._close(paper_id, pair, pos, bar, target, "target")
            return True
        if direction < 0 and low <= target:
            await self._close(paper_id, pair, pos, bar, target, "target")
            return True
        # Protection earned on this candle becomes active from the next candle.
        self._maybe_protect_stop(pos, bar, params)
        return False

    async def _manage_existing(
        self,
        paper_id: str,
        pair: str,
        pos: dict,
        bar: dict,
        params: dict,
        *,
        strategy: str,
        bars: List[dict],
        index: int,
    ) -> bool:
        direction = int(pos["direction"])
        source = str(params.get("_paper_source") or "forex")
        low, high = float(bar["l"]), float(bar["h"])

        # Hard risk comes first for every paper strategy. This also repairs
        # positions opened by older builds whose stored stop did not reserve
        # modeled round-trip costs inside the 1R budget.
        hard_stop = _max_loss_stop_price(
            entry=float(pos["entry_price"]),
            risk_distance=float(pos["risk_distance"]),
            direction=direction,
            cost_bps=float(self.settings.forex_cost_bps),
            max_loss_r=1.0,
        )
        stored_stop = float(pos["stop_price"])
        stop = max(stored_stop, hard_stop) if direction > 0 else min(stored_stop, hard_stop)
        if direction > 0 and low <= stop:
            await self._close(paper_id, pair, pos, bar, stop, "stop")
            return True
        if direction < 0 and high >= stop:
            await self._close(paper_id, pair, pos, bar, stop, "stop")
            return True

        # Research-native signal/max-hold exits are secondary to the 1R stop.
        if source == "research" and strategy not in {"asymmetric_breakout", "regime_ensemble"}:
            reason = _research_exit_reason(
                strategy, params, bars, index, direction
            )
            if reason:
                await self._close(
                    paper_id, pair, pos, bar, float(bar["c"]), reason
                )
                return True
            held = int(pos.get("bars_held", 0)) + 1
            if held >= int(params.get("max_hold", 36)):
                await self._close(
                    paper_id, pair, pos, bar, float(bar["c"]), "max_hold"
                )
                return True
            return False

        target = float(pos["target_price"])
        if direction > 0 and high >= target:
            await self._close(paper_id, pair, pos, bar, target, "target")
            return True
        if direction < 0 and low <= target:
            await self._close(paper_id, pair, pos, bar, target, "target")
            return True

        self._maybe_protect_stop(pos, bar, params)

        held = int(pos.get("bars_held", 0)) + 1
        if held >= int(params.get("max_hold", 36)):
            await self._close(
                paper_id, pair, pos, bar, float(bar["c"]), "max_hold"
            )
            return True
        return False

    async def _close(
        self,
        paper_id: str,
        pair: str,
        pos: dict,
        bar: dict,
        exit_price: float,
        reason: str,
    ) -> None:
        direction = int(pos["direction"])
        entry = float(pos["entry_price"])
        risk_distance = float(pos["risk_distance"])
        gross = direction * ((float(exit_price) / entry) - 1.0)
        net = gross - (2.0 * float(self.settings.forex_cost_bps) / 10_000.0)
        risk_pct = risk_distance / entry if entry > 0 else 0.0
        r_multiple = net / risk_pct if risk_pct > 0 else 0.0

        # Data-integrity backstop: paper execution may model a tiny amount of
        # adverse slippage, but never an unbounded multi-R loss. Normal stops
        # are cost-adjusted to -1.00R; this 1.05R cap is only a final guard.
        if r_multiple < -1.05:
            capped_price = _max_loss_stop_price(
                entry=entry,
                risk_distance=risk_distance,
                direction=direction,
                cost_bps=float(self.settings.forex_cost_bps),
                max_loss_r=1.05,
            )
            log.warning(
                "Paper loss guard applied: paper_id=%s pair=%s reason=%s raw_r=%.4f capped_r=-1.05",
                paper_id, pair, reason, r_multiple,
            )
            exit_price = capped_price
            gross = direction * ((float(exit_price) / entry) - 1.0)
            net = gross - (2.0 * float(self.settings.forex_cost_bps) / 10_000.0)
            r_multiple = net / risk_pct if risk_pct > 0 else -1.05

        risk_eur = float(pos["risk_eur"])
        pnl = r_multiple * risk_eur
        await self.store.record_trade(
            paper_id=paper_id,
            pair=pair,
            side="long" if direction > 0 else "short",
            entry_time=str(pos["entry_time"]),
            exit_time=str(bar["t"]),
            entry_price=entry,
            exit_price=float(exit_price),
            exit_reason=reason,
            risk_eur=risk_eur,
            r_multiple=round(r_multiple, 6),
            pnl=round(pnl, 6),
        )
        await self.store.delete_position(paper_id, pair)
