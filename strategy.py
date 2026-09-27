from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import List, Optional


@dataclass
class Signal:
    action: str  # BUY, SELL, HOLD
    reason: str
    fast: Optional[float] = None
    slow: Optional[float] = None
    edge_bps: Optional[float] = None
    last_price: Optional[float] = None
    bar_time: Optional[str] = None


def moving_average_momentum(
    bars: List[dict],
    fast_window: int,
    slow_window: int,
    entry_edge_bps: float,
    exit_edge_bps: float,
    has_position: bool,
) -> Signal:
    need = max(fast_window, slow_window)
    if len(bars) < need:
        return Signal("HOLD", f"need {need} bars, got {len(bars)}")

    closes = [float(b["c"]) for b in bars[-need:]]
    fast = mean(closes[-fast_window:])
    slow = mean(closes[-slow_window:])
    edge_bps = ((fast / slow) - 1.0) * 10_000
    last = closes[-1]
    bar_time = bars[-1].get("t")

    if not has_position and edge_bps >= entry_edge_bps:
        return Signal("BUY", "fast MA above slow MA", fast, slow, edge_bps, last, bar_time)
    if has_position and edge_bps <= exit_edge_bps:
        return Signal("SELL", "momentum faded", fast, slow, edge_bps, last, bar_time)
    return Signal("HOLD", "no threshold crossed", fast, slow, edge_bps, last, bar_time)
