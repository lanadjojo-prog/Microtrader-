from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

from market_filters import active_session, relative_volume

ADAPTIVE_CONTEXT_VERSION = "adaptive-context-v1"


def _atr(bars: List[dict], i: int, window: int = 14) -> float:
    vals: List[float] = []
    for j in range(max(1, i - int(window)), i):
        high = float(bars[j]["h"])
        low = float(bars[j]["l"])
        prev = float(bars[j - 1]["c"])
        vals.append(max(high - low, abs(high - prev), abs(low - prev)))
    return mean(vals) if vals else 0.0


def _rolling_vwap(bars: List[dict], a: int, b: int) -> float:
    sample = bars[max(0, a):max(0, b)]
    if not sample:
        return 0.0
    pv = 0.0
    vol = 0.0
    for row in sample:
        v = float(row.get("v") or 0.0)
        typ = (float(row["h"]) + float(row["l"]) + float(row["c"])) / 3.0
        pv += typ * v
        vol += v
    return pv / vol if vol > 0 else mean(float(row["c"]) for row in sample)


def context_key(context: dict) -> str:
    regime = str(context.get("regime") or "unknown")
    session = str(context.get("session") or "off_session")
    return f"{regime}|{session}"


def market_context(
    bars: List[dict],
    i: int,
    params: dict | None = None,
) -> dict:
    """Return a causal context vector using price data strictly before i.

    Timestamp/session metadata for the proposed entry bar may be read, but no
    OHLCV data from that bar is used in the regime decision.
    """
    params = params or {}
    atr_short_n = int(params.get("regime_atr_short", 14))
    atr_long_n = int(params.get("regime_atr_long", 50))
    fast = int(params.get("fast", 8))
    slow = int(params.get("slow", 30))
    vwap_window = int(params.get("vwap_window", 60))
    need = max(atr_long_n + 1, slow, vwap_window)
    if i < need or i < 2:
        return {
            "valid": False,
            "regime": "warmup",
            "session": "",
            "key": "warmup|off_session",
        }

    atr_short = _atr(bars, i, atr_short_n)
    atr_long = _atr(bars, i, atr_long_n)
    if atr_short <= 0 or atr_long <= 0:
        return {
            "valid": False,
            "regime": "invalid",
            "session": "",
            "key": "invalid|off_session",
        }

    closes = [float(row["c"]) for row in bars[i - slow:i]]
    fast_n = max(2, min(fast, len(closes)))
    fast_ma = mean(closes[-fast_n:])
    slow_ma = mean(closes)
    trend_strength = abs(fast_ma - slow_ma) / atr_long
    vol_ratio = atr_short / atr_long

    vol_threshold = float(params.get("regime_vol_ratio", 1.20))
    trend_threshold = float(params.get("regime_trend_atr", 0.55))

    if vol_ratio >= vol_threshold:
        regime = "expansion"
    elif trend_strength >= trend_threshold:
        regime = "trend"
    else:
        regime = "range"

    if trend_strength < trend_threshold * 0.5:
        trend_direction = "flat"
    else:
        trend_direction = "up" if fast_ma > slow_ma else "down"

    if vol_ratio >= vol_threshold:
        volatility = "high"
    elif vol_ratio <= max(0.01, 2.0 - vol_threshold):
        volatility = "low"
    else:
        volatility = "normal"

    entry_time = None
    if 0 <= i < len(bars):
        entry_time = bars[i].get("t")
    elif bars:
        entry_time = bars[i - 1].get("t")
    session = active_session(entry_time)

    vol_ratio_rel = relative_volume(
        bars,
        min(i, len(bars) - 1),
        int(params.get("volume_window", 50)),
    )
    vw = _rolling_vwap(bars, i - vwap_window, i)
    last_close = float(bars[i - 1]["c"])
    vwap_distance_bps = ((last_close / vw) - 1.0) * 10_000.0 if vw > 0 else 0.0

    context = {
        "valid": True,
        "regime": regime,
        "session": session,
        "volatility": volatility,
        "trend_direction": trend_direction,
        "trend_strength_atr": round(trend_strength, 4),
        "atr_ratio": round(vol_ratio, 4),
        "relative_volume": round(float(vol_ratio_rel), 4) if vol_ratio_rel is not None else None,
        "vwap_distance_bps": round(vwap_distance_bps, 3),
    }
    context["key"] = context_key(context)
    return context


def _entry_params_for_model(model: str, source: dict) -> dict:
    source = dict(source or {})
    keys = {
        "momentum": ("fast", "slow", "entry_bps"),
        "mean_reversion": ("window", "z_entry", "z_exit"),
        "breakout": ("window", "buffer_bps"),
        "extreme_reversal": ("window", "shock_z"),
        "volatility_breakout": ("window", "vol_mult"),
        "trend_pullback": ("fast", "slow", "pullback_z"),
        "vwap_reversion": ("window", "vwap_window", "z_entry"),
        "vwap_momentum": ("window", "vwap_window", "buffer_bps"),
        "asymmetric_breakout": ("window",),
    }.get(str(model), ())
    return {key: source[key] for key in keys if key in source}


def entry_signal(model: str, params: dict, bars: List[dict], i: int) -> int:
    """Causal entry signal shared by backtest and forward paper routing."""
    model = str(model)
    p = dict(params or {})
    if i < 2:
        return 0

    if model == "momentum":
        slow = int(p.get("slow", 16))
        fast = int(p.get("fast", 4))
        if i < slow:
            return 0
        history = [float(x["c"]) for x in bars[i - slow:i]]
        fast_ma = mean(history[-fast:])
        slow_ma = mean(history)
        edge_bps = ((fast_ma / slow_ma) - 1.0) * 10_000 if slow_ma else 0.0
        threshold = float(p.get("entry_bps", 8.0))
        return 1 if edge_bps >= threshold else (-1 if edge_bps <= -threshold else 0)

    if model == "mean_reversion":
        window = int(p.get("window", 20))
        if i < window:
            return 0
        history = [float(x["c"]) for x in bars[i - window:i]]
        mu = mean(history)
        sigma = pstdev(history)
        if sigma <= 0:
            return 0
        z = (history[-1] - mu) / sigma
        threshold = float(p.get("z_entry", 1.5))
        return 1 if z <= -threshold else (-1 if z >= threshold else 0)

    if model in {"breakout", "asymmetric_breakout"}:
        window = int(p.get("window", p.get("breakout_window", 20)))
        if i < window + 1:
            return 0
        prior = bars[i - window - 1:i - 1]
        if not prior:
            return 0
        buf = 0.0 if model == "asymmetric_breakout" else float(p.get("buffer_bps", 2.0)) / 10_000.0
        px = float(bars[i - 1]["c"])
        hi = max(float(x["h"]) for x in prior) * (1.0 + buf)
        lo = min(float(x["l"]) for x in prior) * (1.0 - buf)
        return 1 if px > hi else (-1 if px < lo else 0)

    if model == "extreme_reversal":
        window = int(p.get("window", 12))
        if i < window + 1:
            return 0
        rets = [
            float(bars[j]["c"]) / float(bars[j - 1]["c"]) - 1.0
            for j in range(i - window, i)
        ]
        sigma = pstdev(rets) if len(rets) > 1 else 0.0
        if sigma <= 0:
            return 0
        threshold = float(p.get("shock_z", 2.0)) * sigma
        return 1 if rets[-1] <= -threshold else (-1 if rets[-1] >= threshold else 0)

    if model == "volatility_breakout":
        window = int(p.get("window", 20))
        if i < window + 1:
            return 0
        ranges = [float(x["h"]) - float(x["l"]) for x in bars[i - window:i]]
        avg_range = mean(ranges) if ranges else 0.0
        cur = float(bars[i - 1]["h"]) - float(bars[i - 1]["l"])
        if avg_range <= 0 or cur < avg_range * float(p.get("vol_mult", 1.6)):
            return 0
        op = float(bars[i - 1]["o"])
        cl = float(bars[i - 1]["c"])
        return 1 if cl > op else (-1 if cl < op else 0)

    if model == "trend_pullback":
        slow = int(p.get("slow", 30))
        fast = int(p.get("fast", 8))
        if i < slow:
            return 0
        closes = [float(x["c"]) for x in bars[i - slow:i]]
        fast_ma = mean(closes[-fast:])
        slow_ma = mean(closes)
        sigma = pstdev(closes)
        z = (closes[-1] - fast_ma) / sigma if sigma > 0 else 0.0
        threshold = float(p.get("pullback_z", 1.0))
        return 1 if (fast_ma > slow_ma and z <= -threshold) else (
            -1 if (fast_ma < slow_ma and z >= threshold) else 0
        )

    if model == "vwap_reversion":
        window = int(p.get("window", p.get("vwap_window", 60)))
        if i < window:
            return 0
        vw = _rolling_vwap(bars, i - window, i)
        closes = [float(x["c"]) for x in bars[i - window:i]]
        sigma = pstdev(closes)
        z = (closes[-1] - vw) / sigma if sigma > 0 else 0.0
        threshold = float(p.get("z_entry", 1.4))
        return 1 if z <= -threshold else (-1 if z >= threshold else 0)

    if model == "vwap_momentum":
        window = int(p.get("window", p.get("vwap_window", 60)))
        if i < window:
            return 0
        vw = _rolling_vwap(bars, i - window, i)
        px = float(bars[i - 1]["c"])
        edge = ((px / vw) - 1.0) * 10_000 if vw > 0 else 0.0
        threshold = float(p.get("buffer_bps", 8.0))
        return 1 if edge >= threshold else (-1 if edge <= -threshold else 0)

    return 0


def default_exit_profile(regime: str) -> dict:
    if regime == "expansion":
        return {"name": "expansion_atr_3r", "stop_atr": 0.8, "target_r": 3.0, "max_hold": 36}
    if regime == "trend":
        return {"name": "trend_atr_2_5r", "stop_atr": 1.0, "target_r": 2.5, "max_hold": 32}
    return {"name": "range_atr_1_5r", "stop_atr": 0.9, "target_r": 1.5, "max_hold": 18}


def _default_entries(regime: str) -> List[dict]:
    if regime == "expansion":
        return [
            {"entry_model": "breakout", "entry_params": {"window": 20, "buffer_bps": 2.0}, "evidence_score": 0.0},
            {"entry_model": "volatility_breakout", "entry_params": {"window": 20, "vol_mult": 1.6}, "evidence_score": 0.0},
        ]
    if regime == "trend":
        return [
            {"entry_model": "trend_pullback", "entry_params": {"fast": 8, "slow": 30, "pullback_z": 1.0}, "evidence_score": 0.0},
            {"entry_model": "momentum", "entry_params": {"fast": 4, "slow": 16, "entry_bps": 8.0}, "evidence_score": 0.0},
        ]
    return [
        {"entry_model": "vwap_reversion", "entry_params": {"window": 60, "z_entry": 1.4}, "evidence_score": 0.0},
        {"entry_model": "mean_reversion", "entry_params": {"window": 20, "z_entry": 1.5, "z_exit": 0.25}, "evidence_score": 0.0},
        {"entry_model": "extreme_reversal", "entry_params": {"window": 12, "shock_z": 2.0}, "evidence_score": 0.0},
    ]


def adaptive_signal(
    params: dict,
    bars: List[dict],
    i: int,
) -> Tuple[int, str, dict, dict, dict]:
    """Return direction, entry-model, context, exit-profile and route metadata."""
    context = market_context(bars, i, params)
    if not context.get("valid"):
        return 0, "none", context, default_exit_profile("range"), {}

    policy = dict(params.get("adaptive_policy") or {})
    routes = dict(policy.get("routes") or {})
    key = context_key(context)
    candidates = list(routes.get(key) or routes.get(f"{context['regime']}|*") or [])
    if not candidates:
        candidates = _default_entries(str(context["regime"]))

    exit_profiles = dict(policy.get("exit_profiles") or {})
    exit_profile = dict(
        exit_profiles.get(key)
        or exit_profiles.get(f"{context['regime']}|*")
        or default_exit_profile(str(context["regime"]))
    )

    target_scale = max(0.25, float(params.get("adaptive_exit_target_scale", 1.0)))
    hold_scale = max(0.25, float(params.get("adaptive_exit_hold_scale", 1.0)))
    stop_scale = max(0.25, float(params.get("adaptive_exit_stop_scale", 1.0)))
    exit_profile["target_r"] = round(float(exit_profile.get("target_r", 2.0)) * target_scale, 4)
    exit_profile["max_hold"] = max(2, int(round(float(exit_profile.get("max_hold", 24)) * hold_scale)))
    exit_profile["stop_atr"] = round(float(exit_profile.get("stop_atr", 1.0)) * stop_scale, 4)
    exit_profile.setdefault("name", f"{context['regime']}_adaptive_rr")

    for route in candidates:
        model = str(route.get("entry_model") or route.get("strategy") or "")
        entry_params = {
            **_entry_params_for_model(model, params),
            **dict(route.get("entry_params") or {}),
        }
        direction = entry_signal(model, entry_params, bars, i)
        if direction:
            meta = {
                "context_key": key,
                "evidence_score": round(float(route.get("evidence_score") or 0.0), 4),
                "source_strategy": str(route.get("source_strategy") or model),
                "source_trades": int(route.get("source_trades") or 0),
                "source_expectancy_bps": float(route.get("source_expectancy_bps") or 0.0),
                "source_profit_factor": float(route.get("source_profit_factor") or 0.0),
            }
            return direction, model, context, exit_profile, meta

    return 0, "none", context, exit_profile, {"context_key": key}


def _split_combo_key(raw: str) -> tuple[str, str]:
    if "||" not in str(raw):
        return str(raw), ""
    return tuple(str(raw).split("||", 1))  # type: ignore[return-value]


def _evidence_score(metrics: dict, stress: dict, funnel_score: float) -> float:
    trades = int(metrics.get("trades") or 0)
    exp = float(metrics.get("expectancy_bps") or 0.0)
    pf = float(metrics.get("profit_factor") or 0.0)
    stress_exp = float(stress.get("expectancy_bps") or 0.0)
    stress_pf = float(stress.get("profit_factor") or 0.0)
    loss_streak = int(metrics.get("max_loss_streak") or 0)
    sample = min(1.0, trades / 25.0)
    robustness = max(-10.0, min(25.0, exp)) + max(-8.0, min(20.0, stress_exp))
    quality = 4.0 * max(-0.5, min(2.0, pf - 1.0)) + 2.0 * max(-0.5, min(2.0, stress_pf - 1.0))
    return round(robustness + quality + sample * 8.0 + float(funnel_score) * 0.04 - loss_streak * 0.4, 4)


def execution_policy(policy: dict) -> dict:
    """Return only routing fields that can change trading decisions.

    Evidence counters/scores stay in the Research Agent state, but are excluded
    from candidate params so harmless score updates do not create a brand-new
    backtest signature.
    """
    routes = {}
    for ctx, rows in dict(policy.get("routes") or {}).items():
        compact = []
        for row in list(rows or []):
            compact.append({
                "entry_model": str(row.get("entry_model") or ""),
                "entry_params": dict(row.get("entry_params") or {}),
                "source_strategy": str(
                    row.get("source_strategy") or row.get("entry_model") or ""
                ),
            })
        if compact:
            routes[str(ctx)] = compact
    return {
        "version": str(policy.get("version") or ADAPTIVE_CONTEXT_VERSION),
        "routes": routes,
        "exit_profiles": {
            str(key): dict(value or {})
            for key, value in dict(policy.get("exit_profiles") or {}).items()
        },
    }


def build_adaptive_policy(
    results: List[dict],
    *,
    min_context_trades: int = 8,
    max_entries_per_context: int = 3,
) -> dict:
    """Build a specialist policy from context-labelled OOS evidence.

    Entry specialists and exit profiles are ranked independently. This function
    never sees individual future outcomes while routing a trade; it only builds
    policy from completed research rows.
    """
    entry_candidates: Dict[str, List[dict]] = {}
    exit_candidates: Dict[str, List[dict]] = {}
    evidence_rows = 0

    for row in results:
        # The adaptive policy is a research artifact. Never let final deep-search
        # holdout outcomes leak back into routing decisions that will later be
        # measured on that same holdout.
        phase = str((row.get("params") or {}).get("_phase") or "discovery")
        if phase not in {"discovery", "incubator"}:
            continue
        diag = dict(row.get("adaptive_diagnostics") or {})
        entry_breakdown = dict(diag.get("context_entry_breakdown") or {})
        stress_entry = dict(diag.get("stress_context_entry_breakdown") or {})
        if entry_breakdown:
            evidence_rows += 1
        params = dict(row.get("params") or {})
        funnel = float(row.get("funnel_score") or 0.0)
        source_strategy = str(row.get("strategy") or "")

        for combo, met in entry_breakdown.items():
            ctx, model = _split_combo_key(str(combo))
            regime = ctx.split("|", 1)[0]
            if regime not in {"expansion", "trend", "range"}:
                continue
            met = dict(met or {})
            stress = dict(stress_entry.get(combo) or {})
            trades = int(met.get("trades") or 0)
            exp = float(met.get("expectancy_bps") or 0.0)
            pf = float(met.get("profit_factor") or 0.0)
            stress_exp = float(stress.get("expectancy_bps") or 0.0)
            stress_pf = float(stress.get("profit_factor") or 0.0)
            stress_trades = int(stress.get("trades") or 0)
            if trades < int(min_context_trades) or exp <= 0.0 or pf < 1.02:
                continue
            if (
                stress
                and stress_trades >= int(min_context_trades)
                and (stress_exp <= 0.0 or stress_pf < 1.0)
            ):
                continue
            entry_candidates.setdefault(ctx, []).append({
                "entry_model": model or source_strategy,
                "entry_params": _entry_params_for_model(model or source_strategy, params),
                "evidence_score": _evidence_score(met, stress, funnel),
                "source_strategy": source_strategy,
                "source_trades": trades,
                "source_expectancy_bps": round(exp, 4),
                "source_profit_factor": round(pf, 4),
                "stress_expectancy_bps": round(stress_exp, 4),
            })

        if source_strategy != "adaptive_router":
            continue
        exit_breakdown = dict(diag.get("context_exit_breakdown") or {})
        stress_exit = dict(diag.get("stress_context_exit_breakdown") or {})
        profiles = dict(diag.get("exit_profiles") or {})
        for combo, met in exit_breakdown.items():
            ctx, label = _split_combo_key(str(combo))
            regime = ctx.split("|", 1)[0]
            if regime not in {"expansion", "trend", "range"}:
                continue
            met = dict(met or {})
            stress = dict(stress_exit.get(combo) or {})
            trades = int(met.get("trades") or 0)
            exp = float(met.get("expectancy_bps") or 0.0)
            pf = float(met.get("profit_factor") or 0.0)
            stress_exp = float(stress.get("expectancy_bps") or 0.0)
            stress_pf = float(stress.get("profit_factor") or 0.0)
            stress_trades = int(stress.get("trades") or 0)
            if trades < int(min_context_trades) or exp <= 0.0 or pf < 1.02:
                continue
            if (
                stress
                and stress_trades >= int(min_context_trades)
                and (stress_exp <= 0.0 or stress_pf < 1.0)
            ):
                continue
            profile = dict(profiles.get(label) or {})
            if not profile:
                continue
            exit_candidates.setdefault(ctx, []).append({
                "profile": profile,
                "score": _evidence_score(met, stress, funnel),
                "trades": trades,
            })

    routes: Dict[str, List[dict]] = {}
    for ctx, rows in entry_candidates.items():
        rows.sort(
            key=lambda x: (
                float(x.get("evidence_score") or 0.0),
                int(x.get("source_trades") or 0),
            ),
            reverse=True,
        )
        chosen: List[dict] = []
        used_models: set[str] = set()
        for row in rows:
            model = str(row.get("entry_model") or "")
            if not model or model in used_models:
                continue
            used_models.add(model)
            chosen.append(row)
            if len(chosen) >= int(max_entries_per_context):
                break
        if chosen:
            routes[ctx] = chosen

    exit_profiles: Dict[str, dict] = {}
    for ctx, rows in exit_candidates.items():
        rows.sort(key=lambda x: (float(x.get("score") or 0.0), int(x.get("trades") or 0)), reverse=True)
        if rows:
            exit_profiles[ctx] = dict(rows[0]["profile"])

    for ctx in list(routes):
        regime = ctx.split("|", 1)[0]
        exit_profiles.setdefault(ctx, default_exit_profile(regime))

    total_specialists = sum(len(rows) for rows in routes.values())
    return {
        "version": ADAPTIVE_CONTEXT_VERSION,
        "routes": routes,
        "exit_profiles": exit_profiles,
        "evidence_rows": evidence_rows,
        "contexts": len(routes),
        "specialists": total_specialists,
        "min_context_trades": int(min_context_trades),
        "max_entries_per_context": int(max_entries_per_context),
        "evidence_phases": ["discovery", "incubator"],
        "final_holdout_excluded": True,
    }
