from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, List, Optional

from forex_metrics import strategy_metrics


@dataclass(frozen=True)
class ForexCandidate:
    strategy: str
    params: dict


def candidate_signature(candidate: ForexCandidate) -> str:
    raw = json.dumps(
        {"strategy": candidate.strategy, "params": candidate.params},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def candidate_grid() -> List[ForexCandidate]:
    """Deliberately small forex-first search space for CPU-constrained Render."""
    out: List[ForexCandidate] = []
    for tf in (1, 5, 15):
        for target_r in (2.0, 3.0, 4.0, 5.0, 8.0):
            for stop_atr in (0.5, 0.75, 1.0):
                common = {
                    "timeframe_min": tf,
                    "target_r": target_r,
                    "stop_atr": stop_atr,
                    "risk_eur": 0.75,
                    "_phase": "discovery",
                }
                out.append(ForexCandidate(
                    "asymmetric_breakout",
                    {**common, "window": 20, "max_hold": 48},
                ))
                out.append(ForexCandidate(
                    "trend_pullback",
                    {**common, "fast": 10, "slow": 30, "max_hold": 36},
                ))
                out.append(ForexCandidate(
                    "range_reversal",
                    {**common, "window": 30, "z_entry": 1.8, "max_hold": 36},
                ))
    return out


def _atr(bars: List[dict], i: int, window: int = 14) -> float:
    vals: List[float] = []
    for j in range(max(1, i - window), i):
        high = float(bars[j]["h"])
        low = float(bars[j]["l"])
        prev = float(bars[j - 1]["c"])
        vals.append(max(high - low, abs(high - prev), abs(low - prev)))
    return mean(vals) if vals else 0.0


def _trade(
    pair: str,
    entry_bar: dict,
    exit_bar: dict,
    entry: float,
    exit_price: float,
    direction: int,
    risk_distance: float,
    cost_bps: float,
    risk_eur: float,
) -> dict:
    if entry <= 0 or risk_distance <= 0:
        gross = net = r_multiple = 0.0
    else:
        gross = direction * ((exit_price / entry) - 1.0)
        net = gross - (2.0 * cost_bps / 10_000.0)
        risk_pct = risk_distance / entry
        r_multiple = net / risk_pct if risk_pct > 0 else 0.0
    return {
        "pair": pair,
        "symbol": pair,
        "entry_time": entry_bar.get("t"),
        "exit_time": exit_bar.get("t"),
        "entry_price": entry,
        "exit_price": exit_price,
        "direction": "long" if direction > 0 else "short",
        "gross_return": gross,
        "net_return": net,
        "r_multiple": r_multiple,
        "pnl": r_multiple * risk_eur,
        "risk_eur": risk_eur,
    }


def _run_asymmetric_exit(
    bars: List[dict],
    entry_idx: int,
    *,
    direction: int,
    entry: float,
    risk_distance: float,
    target_r: float,
    max_hold: int,
) -> tuple[int, float]:
    if direction > 0:
        stop = entry - risk_distance
        target = entry + risk_distance * target_r
    else:
        stop = entry + risk_distance
        target = entry - risk_distance * target_r

    last_idx = min(entry_idx + max_hold, len(bars) - 1)
    exit_price = float(bars[last_idx]["c"])

    for j in range(entry_idx, last_idx + 1):
        low = float(bars[j]["l"])
        high = float(bars[j]["h"])

        # Conservative assumption when stop and target are both touched in
        # one candle: the stop is assumed to have happened first.
        if direction > 0:
            if low <= stop:
                return j, stop
            if high >= target:
                return j, target
        else:
            if high >= stop:
                return j, stop
            if low <= target:
                return j, target
    return last_idx, exit_price


def _simulate_asymmetric_breakout(
    candidate: ForexCandidate,
    pair: str,
    bars: List[dict],
    cost_bps: float,
) -> List[dict]:
    p = candidate.params
    window = int(p["window"])
    stop_atr = float(p["stop_atr"])
    target_r = float(p["target_r"])
    risk_eur = float(p.get("risk_eur", 0.75))
    max_hold = int(p["max_hold"])
    trades: List[dict] = []
    i = max(window + 2, 16)

    while i < len(bars) - 2:
        prior = bars[i - window - 1:i - 1]
        prev_close = float(bars[i - 1]["c"])
        high_break = max(float(x["h"]) for x in prior)
        low_break = min(float(x["l"]) for x in prior)
        direction = 1 if prev_close > high_break else (-1 if prev_close < low_break else 0)
        if not direction:
            i += 1
            continue

        entry = float(bars[i]["o"])
        atr = _atr(bars, i, 14)
        risk_distance = atr * stop_atr
        if risk_distance <= 0:
            i += 1
            continue

        exit_idx, exit_price = _run_asymmetric_exit(
            bars, i, direction=direction, entry=entry,
            risk_distance=risk_distance, target_r=target_r, max_hold=max_hold,
        )
        trades.append(_trade(
            pair, bars[i], bars[exit_idx], entry, exit_price,
            direction, risk_distance, cost_bps, risk_eur,
        ))
        i = exit_idx + 1
    return trades


def _simulate_trend_pullback(
    candidate: ForexCandidate,
    pair: str,
    bars: List[dict],
    cost_bps: float,
) -> List[dict]:
    p = candidate.params
    fast = int(p["fast"])
    slow = int(p["slow"])
    stop_atr = float(p["stop_atr"])
    target_r = float(p["target_r"])
    risk_eur = float(p.get("risk_eur", 0.75))
    max_hold = int(p["max_hold"])
    trades: List[dict] = []
    i = max(slow + 1, 16)

    while i < len(bars) - 2:
        closes = [float(x["c"]) for x in bars[i - slow:i]]
        fast_ma = mean(closes[-fast:])
        slow_ma = mean(closes)
        prev_close = closes[-1]

        direction = 0
        if fast_ma > slow_ma and slow_ma < prev_close <= fast_ma:
            direction = 1
        elif fast_ma < slow_ma and fast_ma <= prev_close < slow_ma:
            direction = -1
        if not direction:
            i += 1
            continue

        entry = float(bars[i]["o"])
        atr = _atr(bars, i, 14)
        risk_distance = atr * stop_atr
        if risk_distance <= 0:
            i += 1
            continue

        exit_idx, exit_price = _run_asymmetric_exit(
            bars, i, direction=direction, entry=entry,
            risk_distance=risk_distance, target_r=target_r, max_hold=max_hold,
        )
        trades.append(_trade(
            pair, bars[i], bars[exit_idx], entry, exit_price,
            direction, risk_distance, cost_bps, risk_eur,
        ))
        i = exit_idx + 1
    return trades


def _simulate_range_reversal(
    candidate: ForexCandidate,
    pair: str,
    bars: List[dict],
    cost_bps: float,
) -> List[dict]:
    p = candidate.params
    window = int(p["window"])
    z_entry = float(p["z_entry"])
    stop_atr = float(p["stop_atr"])
    target_r = float(p["target_r"])
    risk_eur = float(p.get("risk_eur", 0.75))
    max_hold = int(p["max_hold"])
    trades: List[dict] = []
    i = max(window + 1, 16)

    while i < len(bars) - 2:
        closes = [float(x["c"]) for x in bars[i - window:i]]
        mu = mean(closes)
        sigma = pstdev(closes)
        if sigma <= 0:
            i += 1
            continue
        z = (closes[-1] - mu) / sigma
        direction = 1 if z <= -z_entry else (-1 if z >= z_entry else 0)
        if not direction:
            i += 1
            continue

        entry = float(bars[i]["o"])
        atr = _atr(bars, i, 14)
        risk_distance = atr * stop_atr
        if risk_distance <= 0:
            i += 1
            continue

        exit_idx, exit_price = _run_asymmetric_exit(
            bars, i, direction=direction, entry=entry,
            risk_distance=risk_distance, target_r=target_r, max_hold=max_hold,
        )
        trades.append(_trade(
            pair, bars[i], bars[exit_idx], entry, exit_price,
            direction, risk_distance, cost_bps, risk_eur,
        ))
        i = exit_idx + 1
    return trades


def simulate(
    candidate: ForexCandidate,
    pair: str,
    bars: List[dict],
    cost_bps: float,
) -> List[dict]:
    if candidate.strategy == "asymmetric_breakout":
        return _simulate_asymmetric_breakout(candidate, pair, bars, cost_bps)
    if candidate.strategy == "trend_pullback":
        return _simulate_trend_pullback(candidate, pair, bars, cost_bps)
    if candidate.strategy == "range_reversal":
        return _simulate_range_reversal(candidate, pair, bars, cost_bps)
    raise ValueError(f"Unknown forex strategy: {candidate.strategy}")


def _days(bars_by_pair: Dict[str, List[dict]]) -> set[str]:
    return {
        str(bar.get("t") or "")[:10]
        for bars in bars_by_pair.values()
        for bar in bars
        if bar.get("t")
    }


def evaluate_candidate(
    candidate: ForexCandidate,
    bars_by_pair: Dict[str, List[dict]],
    *,
    cost_bps: float = 0.6,
    stress_multiplier: float = 2.0,
    min_oos_trades: int = 80,
    min_profit_factor: float = 1.25,
    min_payoff_ratio: float = 1.8,
    start_capital: float = 50.0,
) -> dict:
    train_trades: List[dict] = []
    oos_trades: List[dict] = []
    stress_trades: List[dict] = []
    train_bars: Dict[str, List[dict]] = {}
    test_bars: Dict[str, List[dict]] = {}
    per_pair: Dict[str, dict] = {}

    for pair, bars in bars_by_pair.items():
        split = max(2, int(len(bars) * 0.70))
        tr = bars[:split]
        te = bars[split:]
        train_bars[pair] = tr
        test_bars[pair] = te

        pair_train = simulate(candidate, pair, tr, cost_bps)
        pair_oos = simulate(candidate, pair, te, cost_bps)
        pair_stress = simulate(candidate, pair, te, cost_bps * stress_multiplier)
        train_trades.extend(pair_train)
        oos_trades.extend(pair_oos)
        stress_trades.extend(pair_stress)

        pair_days = {str(x.get("t") or "")[:10] for x in te if x.get("t")}
        per_pair[pair] = {
            "oos": strategy_metrics(
                pair_oos, trading_days=pair_days, start_capital=start_capital
            ),
            "stress_oos": strategy_metrics(
                pair_stress, trading_days=pair_days, start_capital=start_capital
            ),
        }

    train_metrics = strategy_metrics(
        train_trades, trading_days=_days(train_bars), start_capital=start_capital
    )
    oos_metrics = strategy_metrics(
        oos_trades, trading_days=_days(test_bars), start_capital=start_capital
    )
    stress_metrics = strategy_metrics(
        stress_trades, trading_days=_days(test_bars), start_capital=start_capital
    )

    reasons: List[str] = []
    if oos_metrics["trades"] < min_oos_trades:
        reasons.append(f"fewer than {min_oos_trades} OOS trades")
    if oos_metrics["expectancy_r"] <= 0:
        reasons.append("non-positive OOS expectancy in R")
    if oos_metrics["profit_factor"] < min_profit_factor:
        reasons.append(f"OOS profit factor below {min_profit_factor:.2f}")
    if oos_metrics["payoff_ratio"] < min_payoff_ratio:
        reasons.append(f"OOS payoff ratio below {min_payoff_ratio:.2f}x")
    if stress_metrics["expectancy_r"] <= 0:
        reasons.append("negative expectancy under stressed costs")

    positive_pairs = sum(
        1 for row in per_pair.values()
        if row["oos"]["trades"] > 0 and row["oos"]["expectancy_r"] > 0
    )
    pair_count = max(1, len(per_pair))
    positive_pair_ratio = positive_pairs / pair_count

    score = 0.0
    score += min(25.0, max(0.0, oos_metrics["expectancy_r"] * 25.0))
    score += min(20.0, max(0.0, (oos_metrics["profit_factor"] - 1.0) * 20.0))
    score += min(20.0, max(0.0, (oos_metrics["payoff_ratio"] - 1.0) * 10.0))
    score += 15.0 * min(1.0, positive_pair_ratio)
    score += 10.0 * min(1.0, oos_metrics["trades"] / max(1, min_oos_trades))
    score += min(10.0, max(0.0, stress_metrics["expectancy_r"] * 10.0))
    score = round(score, 2)

    raw_pass = not reasons
    phase = str(candidate.params.get("_phase", "discovery"))
    promoted = raw_pass and phase == "deep_search"

    if promoted:
        stage = "promoted"
    elif raw_pass or score >= 70:
        stage = "deep_search"
    elif (
        oos_metrics["trades"] >= 20
        and oos_metrics["expectancy_r"] > 0
        and oos_metrics["profit_factor"] >= 1.05
    ):
        stage = "incubator"
    else:
        stage = "rejected"

    return {
        "strategy": candidate.strategy,
        "family": candidate.strategy,
        "phase": phase,
        "status": stage,
        "funnel_stage": stage,
        "funnel_score": score,
        "params": candidate.params,
        "timeframe_min": int(candidate.params.get("timeframe_min", 1)),
        "pairs": list(bars_by_pair.keys()),
        "dataset": {
            "pairs": list(bars_by_pair.keys()),
            "bars": {pair: len(bars) for pair, bars in bars_by_pair.items()},
            "split": "70/30 chronological",
        },
        "risk_model": {
            "start_capital_eur": start_capital,
            "fixed_risk_eur": float(candidate.params.get("risk_eur", 0.75)),
            "target_r": float(candidate.params.get("target_r", 0.0)),
            "note": "PnL is a fixed-risk R simulation; broker margin/lot constraints are validated separately.",
        },
        "train": train_metrics,
        "oos": oos_metrics,
        "stress_oos": stress_metrics,
        "per_pair": per_pair,
        "positive_pair_ratio": round(positive_pair_ratio, 3),
        "positive_pairs": positive_pairs,
        "pair_count": len(per_pair),
        "promoted": promoted,
        "rejection_reasons": reasons,
    }
