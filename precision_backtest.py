from __future__ import annotations

import bisect
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean
from typing import Dict, List, Optional, Tuple

from forex_metrics import strategy_metrics


QuoteTick = Tuple[int, float, float]  # timestamp_ms, bid, ask


@dataclass(frozen=True)
class PrecisionCandidate:
    strategy: str
    params: dict


def candidate_signature(candidate: PrecisionCandidate) -> str:
    raw = json.dumps(
        {"strategy": candidate.strategy, "params": candidate.params},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def candidate_grid() -> List[PrecisionCandidate]:
    """Tight-stop search space: 3/4 pip stops and 5..10 pip targets.

    These are deliberately separate from the ATR-based normal Forex Lab.
    The families are mechanical, ICT-inspired hypotheses, not assumptions
    that any named trading methodology is profitable.
    """
    out: List[PrecisionCandidate] = []
    for stop_pips in (3.0, 4.0):
        for target_pips in (5.0, 6.0, 7.0, 8.0, 9.0, 10.0):
            common = {
                "timeframe_min": 1,
                "stop_pips": stop_pips,
                "target_pips": target_pips,
                "risk_eur": 0.75,
                "_phase": "precision_discovery",
            }
            out.extend([
                PrecisionCandidate(
                    "liquidity_sweep_fvg",
                    {**common, "lookback": 20, "displacement_atr": 0.8,
                     "pending_ttl_min": 20, "max_hold_min": 60},
                ),
                PrecisionCandidate(
                    "displacement_fvg_retrace",
                    {**common, "atr_window": 14, "displacement_atr": 1.2,
                     "pending_ttl_min": 20, "max_hold_min": 60},
                ),
                PrecisionCandidate(
                    "session_sweep_reversal",
                    {**common, "range_start_utc": 0, "range_end_utc": 6,
                     "trade_start_utc": 7, "trade_end_utc": 16,
                     "pending_ttl_min": 15, "max_hold_min": 45},
                ),
                PrecisionCandidate(
                    "opening_range_retest",
                    {**common, "or_start_utc": 7, "or_minutes": 15,
                     "pending_ttl_min": 20, "max_hold_min": 60},
                ),
            ])
    return out


def pip_size(pair: str) -> float:
    return 0.01 if pair.upper().replace("/", "").endswith("JPY") else 0.0001


def _dt(raw) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None


def _atr(bars: List[dict], i: int, window: int = 14) -> float:
    vals: List[float] = []
    for j in range(max(1, i-window), i):
        h = float(bars[j]["h"])
        l = float(bars[j]["l"])
        prev = float(bars[j-1]["c"])
        vals.append(max(h-l, abs(h-prev), abs(l-prev)))
    return mean(vals) if vals else 0.0


def _signal(
    pair: str,
    side: str,
    order_type: str,
    entry_price: float,
    created_at: datetime,
    p: dict,
    tag: str,
) -> dict:
    return {
        "pair": pair,
        "side": side,
        "order_type": order_type,
        "entry_price": float(entry_price),
        "created_at": created_at.astimezone(timezone.utc),
        "expires_at": created_at.astimezone(timezone.utc) + timedelta(
            minutes=int(p.get("pending_ttl_min", 20))
        ),
        "stop_pips": float(p["stop_pips"]),
        "target_pips": float(p["target_pips"]),
        "risk_eur": float(p.get("risk_eur", 0.75)),
        "max_hold_min": int(p.get("max_hold_min", 60)),
        "tag": tag,
    }


def _liquidity_sweep_fvg_signals(
    candidate: PrecisionCandidate, pair: str, bars: List[dict]
) -> List[dict]:
    p = candidate.params
    lookback = int(p.get("lookback", 20))
    out: List[dict] = []
    i = max(lookback + 3, 18)
    while i < len(bars):
        before = bars[i-lookback-2:i-2]
        sweep = bars[i-2]
        disp = bars[i-1]
        created = _dt(bars[i].get("t"))
        if not before or not created:
            i += 1
            continue

        prior_high = max(float(x["h"]) for x in before)
        prior_low = min(float(x["l"]) for x in before)
        atr = _atr(bars, i-1, 14)
        body = abs(float(disp["c"]) - float(disp["o"]))
        strong = atr > 0 and body >= atr * float(p.get("displacement_atr", 0.8))

        # Long: prior low is swept, price closes back above it, then bullish
        # displacement creates a 3-candle fair-value gap.
        if (
            strong
            and float(sweep["l"]) < prior_low
            and float(sweep["c"]) > prior_low
            and float(disp["c"]) > float(disp["o"])
        ):
            gap_low = float(bars[i-3]["h"])
            gap_high = float(disp["l"])
            if gap_high > gap_low:
                out.append(_signal(
                    pair, "long", "limit", (gap_low + gap_high) / 2.0,
                    created, p, "sweep_low_bullish_fvg",
                ))

        if (
            strong
            and float(sweep["h"]) > prior_high
            and float(sweep["c"]) < prior_high
            and float(disp["c"]) < float(disp["o"])
        ):
            gap_low = float(disp["h"])
            gap_high = float(bars[i-3]["l"])
            if gap_high > gap_low:
                out.append(_signal(
                    pair, "short", "limit", (gap_low + gap_high) / 2.0,
                    created, p, "sweep_high_bearish_fvg",
                ))
        i += 1
    return out


def _displacement_fvg_signals(
    candidate: PrecisionCandidate, pair: str, bars: List[dict]
) -> List[dict]:
    p = candidate.params
    out: List[dict] = []
    i = 18
    while i < len(bars):
        a, _, c = bars[i-3], bars[i-2], bars[i-1]
        created = _dt(bars[i].get("t"))
        if not created:
            i += 1
            continue
        atr = _atr(bars, i-1, int(p.get("atr_window", 14)))
        body = abs(float(c["c"]) - float(c["o"]))
        if atr <= 0 or body < atr * float(p.get("displacement_atr", 1.2)):
            i += 1
            continue

        if float(c["c"]) > float(c["o"]) and float(c["l"]) > float(a["h"]):
            out.append(_signal(
                pair, "long", "limit", (float(a["h"]) + float(c["l"])) / 2.0,
                created, p, "bullish_displacement_fvg_mid",
            ))
        elif float(c["c"]) < float(c["o"]) and float(c["h"]) < float(a["l"]):
            out.append(_signal(
                pair, "short", "limit", (float(c["h"]) + float(a["l"])) / 2.0,
                created, p, "bearish_displacement_fvg_mid",
            ))
        i += 1
    return out


def _session_sweep_signals(
    candidate: PrecisionCandidate, pair: str, bars: List[dict]
) -> List[dict]:
    p = candidate.params
    by_day: Dict[str, List[dict]] = {}
    for bar in bars:
        dt = _dt(bar.get("t"))
        if dt:
            by_day.setdefault(dt.date().isoformat(), []).append(bar)

    out: List[dict] = []
    for _, daybars in sorted(by_day.items()):
        range_bars = []
        trade_bars = []
        for bar in daybars:
            dt = _dt(bar.get("t"))
            if not dt:
                continue
            if int(p["range_start_utc"]) <= dt.hour < int(p["range_end_utc"]):
                range_bars.append(bar)
            if int(p["trade_start_utc"]) <= dt.hour < int(p["trade_end_utc"]):
                trade_bars.append(bar)
        if len(range_bars) < 10:
            continue
        hi = max(float(x["h"]) for x in range_bars)
        lo = min(float(x["l"]) for x in range_bars)
        for idx, bar in enumerate(trade_bars[:-1]):
            next_dt = _dt(trade_bars[idx+1].get("t"))
            if not next_dt:
                continue
            o, h, l, c = (float(bar[k]) for k in ("o","h","l","c"))
            if l < lo and c > lo:
                # Retracement into the lower half of the reversal candle.
                entry = (o + l) / 2.0
                out.append(_signal(
                    pair, "long", "limit", entry, next_dt, p,
                    "session_low_sweep_reversal",
                ))
                break
            if h > hi and c < hi:
                entry = (o + h) / 2.0
                out.append(_signal(
                    pair, "short", "limit", entry, next_dt, p,
                    "session_high_sweep_reversal",
                ))
                break
    return out


def _opening_range_retest_signals(
    candidate: PrecisionCandidate, pair: str, bars: List[dict]
) -> List[dict]:
    p = candidate.params
    by_day: Dict[str, List[dict]] = {}
    for bar in bars:
        dt = _dt(bar.get("t"))
        if dt:
            by_day.setdefault(dt.date().isoformat(), []).append(bar)

    out: List[dict] = []
    for _, daybars in sorted(by_day.items()):
        start = int(p.get("or_start_utc", 7))
        minutes = int(p.get("or_minutes", 15))
        opening = []
        later = []
        for bar in daybars:
            dt = _dt(bar.get("t"))
            if not dt:
                continue
            minute_of_day = dt.hour * 60 + dt.minute
            a = start * 60
            if a <= minute_of_day < a + minutes:
                opening.append(bar)
            elif a + minutes <= minute_of_day < a + minutes + 180:
                later.append(bar)
        if len(opening) < max(5, minutes // 2):
            continue
        hi = max(float(x["h"]) for x in opening)
        lo = min(float(x["l"]) for x in opening)
        for idx, bar in enumerate(later[:-1]):
            next_dt = _dt(later[idx+1].get("t"))
            if not next_dt:
                continue
            if float(bar["c"]) > hi:
                out.append(_signal(
                    pair, "long", "limit", hi, next_dt, p,
                    "opening_range_break_retest_long",
                ))
                break
            if float(bar["c"]) < lo:
                out.append(_signal(
                    pair, "short", "limit", lo, next_dt, p,
                    "opening_range_break_retest_short",
                ))
                break
    return out


def generate_signals(
    candidate: PrecisionCandidate, pair: str, bars: List[dict]
) -> List[dict]:
    if candidate.strategy == "liquidity_sweep_fvg":
        return _liquidity_sweep_fvg_signals(candidate, pair, bars)
    if candidate.strategy == "displacement_fvg_retrace":
        return _displacement_fvg_signals(candidate, pair, bars)
    if candidate.strategy == "session_sweep_reversal":
        return _session_sweep_signals(candidate, pair, bars)
    if candidate.strategy == "opening_range_retest":
        return _opening_range_retest_signals(candidate, pair, bars)
    raise ValueError(f"Unknown precision strategy: {candidate.strategy}")


def execute_signals(
    signals: List[dict],
    ticks: List[QuoteTick],
    pair: str,
    commission_pips_roundtrip: float,
) -> List[dict]:
    """Execute pending entries and exits against bid/ask ticks.

    Long limit fills use ask; long exits use bid. Short limit fills use bid;
    short exits use ask. This makes spread a real part of the simulation.
    """
    if not signals or not ticks:
        return []

    times = [row[0] for row in ticks]
    pip = pip_size(pair)
    trades: List[dict] = []
    unavailable_until = -1

    for signal in sorted(signals, key=lambda x: x["created_at"]):
        created_ms = int(signal["created_at"].timestamp() * 1000)
        expiry_ms = int(signal["expires_at"].timestamp() * 1000)
        if created_ms <= unavailable_until:
            continue

        idx = bisect.bisect_left(times, created_ms)
        fill_idx = None
        fill_px = None
        while idx < len(ticks) and ticks[idx][0] <= expiry_ms:
            _, bid, ask = ticks[idx]
            level = float(signal["entry_price"])
            if signal["side"] == "long":
                if signal["order_type"] == "limit" and ask <= level:
                    fill_idx, fill_px = idx, min(ask, level)
                    break
                if signal["order_type"] == "stop" and ask >= level:
                    fill_idx, fill_px = idx, max(ask, level)
                    break
            else:
                if signal["order_type"] == "limit" and bid >= level:
                    fill_idx, fill_px = idx, max(bid, level)
                    break
                if signal["order_type"] == "stop" and bid <= level:
                    fill_idx, fill_px = idx, min(bid, level)
                    break
            idx += 1

        if fill_idx is None or fill_px is None:
            continue

        side_mult = 1 if signal["side"] == "long" else -1
        stop_dist = float(signal["stop_pips"]) * pip
        target_dist = float(signal["target_pips"]) * pip
        stop_px = fill_px - stop_dist if side_mult > 0 else fill_px + stop_dist
        target_px = fill_px + target_dist if side_mult > 0 else fill_px - target_dist
        max_exit_ms = ticks[fill_idx][0] + int(signal["max_hold_min"]) * 60_000

        exit_idx = None
        exit_px = None
        exit_reason = "timeout"
        j = fill_idx
        while j < len(ticks) and ticks[j][0] <= max_exit_ms:
            _, bid, ask = ticks[j]
            if side_mult > 0:
                if bid <= stop_px:
                    exit_idx, exit_px, exit_reason = j, bid, "stop"
                    break
                if bid >= target_px:
                    exit_idx, exit_px, exit_reason = j, bid, "target"
                    break
            else:
                if ask >= stop_px:
                    exit_idx, exit_px, exit_reason = j, ask, "stop"
                    break
                if ask <= target_px:
                    exit_idx, exit_px, exit_reason = j, ask, "target"
                    break
            j += 1

        if exit_idx is None:
            exit_idx = min(max(fill_idx, j-1), len(ticks)-1)
            _, bid, ask = ticks[exit_idx]
            exit_px = bid if side_mult > 0 else ask

        gross_pips = side_mult * (float(exit_px) - float(fill_px)) / pip
        net_pips = gross_pips - float(commission_pips_roundtrip)
        r_multiple = net_pips / float(signal["stop_pips"])
        risk_eur = float(signal["risk_eur"])
        spread_pips = max(
            0.0, (ticks[fill_idx][2] - ticks[fill_idx][1]) / pip
        )

        trades.append({
            "pair": pair,
            "symbol": pair,
            "strategy_tag": signal["tag"],
            "order_type": signal["order_type"],
            "side": signal["side"],
            "entry_time": datetime.fromtimestamp(
                ticks[fill_idx][0] / 1000, tz=timezone.utc
            ).isoformat(),
            "exit_time": datetime.fromtimestamp(
                ticks[exit_idx][0] / 1000, tz=timezone.utc
            ).isoformat(),
            "entry_price": round(float(fill_px), 6),
            "exit_price": round(float(exit_px), 6),
            "stop_pips": float(signal["stop_pips"]),
            "target_pips": float(signal["target_pips"]),
            "gross_pips": round(gross_pips, 3),
            "net_pips": round(net_pips, 3),
            "entry_spread_pips": round(spread_pips, 3),
            "commission_pips": float(commission_pips_roundtrip),
            "r_multiple": round(r_multiple, 4),
            "pnl": round(r_multiple * risk_eur, 4),
            "risk_eur": risk_eur,
            "net_return": round((r_multiple * risk_eur) / 50.0, 8),
            "exit_reason": exit_reason,
        })
        unavailable_until = ticks[exit_idx][0]

    return trades


def _days(bars_by_pair: Dict[str, List[dict]]) -> set[str]:
    return {
        str(bar.get("t") or "")[:10]
        for bars in bars_by_pair.values()
        for bar in bars
        if bar.get("t")
    }


def evaluate_candidate(
    candidate: PrecisionCandidate,
    bars_by_pair: Dict[str, List[dict]],
    ticks_by_pair: Dict[str, List[QuoteTick]],
    *,
    commission_pips_roundtrip: float = 0.5,
    stress_multiplier: float = 2.0,
    min_oos_trades: int = 40,
    min_profit_factor: float = 1.20,
    start_capital: float = 50.0,
) -> dict:
    train_trades: List[dict] = []
    oos_trades: List[dict] = []
    stress_trades: List[dict] = []
    train_bars: Dict[str, List[dict]] = {}
    test_bars: Dict[str, List[dict]] = {}
    per_pair: Dict[str, dict] = {}

    for pair, bars in bars_by_pair.items():
        ticks = ticks_by_pair.get(pair) or []
        if len(bars) < 100 or not ticks:
            continue

        split = max(2, int(len(bars) * 0.70))
        tr_bars = bars[:split]
        te_bars = bars[split:]
        train_bars[pair] = tr_bars
        test_bars[pair] = te_bars

        train_signals = generate_signals(candidate, pair, tr_bars)
        oos_signals = generate_signals(candidate, pair, te_bars)
        train_exec = execute_signals(
            train_signals, ticks, pair, commission_pips_roundtrip
        )
        oos_exec = execute_signals(
            oos_signals, ticks, pair, commission_pips_roundtrip
        )
        stress_exec = execute_signals(
            oos_signals, ticks, pair,
            commission_pips_roundtrip * stress_multiplier,
        )

        train_trades.extend(train_exec)
        oos_trades.extend(oos_exec)
        stress_trades.extend(stress_exec)
        pair_days = {
            str(x.get("t") or "")[:10] for x in te_bars if x.get("t")
        }
        per_pair[pair] = {
            "signals_oos": len(oos_signals),
            "fill_rate_pct": round(
                100.0 * len(oos_exec) / max(1, len(oos_signals)), 2
            ),
            "oos": strategy_metrics(
                oos_exec, trading_days=pair_days, start_capital=start_capital
            ),
            "stress_oos": strategy_metrics(
                stress_exec, trading_days=pair_days, start_capital=start_capital
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
    if stress_metrics["expectancy_r"] <= 0:
        reasons.append("negative expectancy under stressed commission")

    positive_pairs = sum(
        1 for row in per_pair.values()
        if row["oos"]["trades"] > 0 and row["oos"]["expectancy_r"] > 0
    )
    pair_count = max(1, len(per_pair))
    positive_pair_ratio = positive_pairs / pair_count
    avg_fill = mean(
        [float(row["fill_rate_pct"]) for row in per_pair.values()]
    ) if per_pair else 0.0

    score = 0.0
    score += min(30.0, max(0.0, oos_metrics["expectancy_r"] * 30.0))
    score += min(25.0, max(0.0, (oos_metrics["profit_factor"] - 1.0) * 25.0))
    score += 15.0 * min(1.0, positive_pair_ratio)
    score += 10.0 * min(1.0, oos_metrics["trades"] / max(1, min_oos_trades))
    score += min(10.0, max(0.0, stress_metrics["expectancy_r"] * 10.0))
    score += min(10.0, max(0.0, avg_fill / 10.0))
    score = round(score, 2)

    if not reasons:
        stage = "precision_deep_search"
    elif (
        oos_metrics["trades"] >= 15
        and oos_metrics["expectancy_r"] > 0
        and oos_metrics["profit_factor"] >= 1.05
    ):
        stage = "precision_incubator"
    else:
        stage = "rejected"

    return {
        "strategy": candidate.strategy,
        "family": candidate.strategy,
        "lab_type": "precision",
        "phase": str(candidate.params.get("_phase", "precision_discovery")),
        "status": stage,
        "funnel_stage": stage,
        "funnel_score": score,
        "params": candidate.params,
        "timeframe_min": 1,
        "pairs": list(per_pair.keys()),
        "dataset": {
            "pairs": list(per_pair.keys()),
            "bars": {pair: len(bars) for pair, bars in bars_by_pair.items()},
            "tick_counts": {
                pair: len(ticks_by_pair.get(pair) or [])
                for pair in bars_by_pair
            },
            "split": "70/30 chronological signal bars; bid/ask tick execution",
        },
        "risk_model": {
            "start_capital_eur": start_capital,
            "fixed_risk_eur": float(candidate.params.get("risk_eur", 0.75)),
            "stop_pips": float(candidate.params["stop_pips"]),
            "target_pips": float(candidate.params["target_pips"]),
            "planned_reward_risk": round(
                float(candidate.params["target_pips"]) /
                float(candidate.params["stop_pips"]), 3
            ),
            "note": (
                "Fixed-risk research model. Broker minimum lot, margin and exact "
                "cash risk are validated against the Fusion cTrader demo account."
            ),
        },
        "execution_model": {
            "pending_orders": True,
            "bid_ask_ticks": True,
            "spread_embedded_in_fills": True,
            "commission_pips_roundtrip": commission_pips_roundtrip,
            "stress_commission_multiplier": stress_multiplier,
            "one_position_per_pair": True,
        },
        "train": train_metrics,
        "oos": oos_metrics,
        "stress_oos": stress_metrics,
        "per_pair": per_pair,
        "positive_pair_ratio": round(positive_pair_ratio, 3),
        "positive_pairs": positive_pairs,
        "pair_count": len(per_pair),
        "avg_fill_rate_pct": round(avg_fill, 2),
        "promoted": False,
        "rejection_reasons": reasons,
        # Keep raw trades for the dedicated trade store. The result-store layer
        # strips these before writing its summary JSON.
        "_train_trades": train_trades,
        "_oos_trades": oos_trades,
        "_stress_trades": stress_trades,
    }
