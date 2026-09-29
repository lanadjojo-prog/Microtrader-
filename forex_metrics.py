from __future__ import annotations

from datetime import datetime
from statistics import mean, median
from typing import Iterable, Optional


def _dt(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None


def _streaks(values: list[float]) -> tuple[int, int]:
    max_wins = max_losses = wins = losses = 0
    for value in values:
        if value > 0:
            wins += 1
            losses = 0
        elif value < 0:
            losses += 1
            wins = 0
        else:
            wins = losses = 0
        max_wins = max(max_wins, wins)
        max_losses = max(max_losses, losses)
    return max_wins, max_losses


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def strategy_metrics(
    trades: list[dict],
    *,
    trading_days: Optional[Iterable[str]] = None,
    start_capital: float = 50.0,
) -> dict:
    """Comprehensive metrics retained for every strategy evaluation.

    A trade may contain:
      net_return, pnl, r_multiple, entry_time, exit_time, pair/symbol.

    trading_days should contain every date present in the evaluated market
    window, including dates on which the strategy placed no trade. This keeps
    average-trades/day honest.
    """
    ordered = sorted(trades, key=lambda x: str(x.get("exit_time") or ""))
    returns = [float(t.get("net_return") or 0.0) for t in ordered]
    pnl_values = [
        float(t["pnl"]) for t in ordered
        if t.get("pnl") is not None
    ]
    r_values = [
        float(t["r_multiple"]) for t in ordered
        if t.get("r_multiple") is not None
    ]

    dates = {str(x)[:10] for x in (trading_days or []) if str(x)}
    if not dates:
        dates = {
            str(t.get("entry_time") or "")[:10]
            for t in ordered if t.get("entry_time")
        }
    days = len(dates)

    if not ordered:
        return {
            "trades": 0,
            "trading_days": days,
            "avg_trades_per_day": 0.0,
            "win_rate_pct": 0.0,
            "loss_rate_pct": 0.0,
            "breakeven_trades": 0,
            "expectancy_bps": 0.0,
            "expectancy_return_pct": 0.0,
            "profit_factor": 0.0,
            "avg_win_bps": 0.0,
            "avg_loss_bps": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "payoff_ratio": 0.0,
            "expectancy_r": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "best_trade_bps": 0.0,
            "worst_trade_bps": 0.0,
            "best_trade_r": 0.0,
            "worst_trade_r": 0.0,
            "median_trade_bps": 0.0,
            "p10_trade_bps": 0.0,
            "p90_trade_bps": 0.0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
            "compounded_trade_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_duration_minutes": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_pnl": 0.0,
            "avg_win_pnl": 0.0,
            "avg_loss_pnl": 0.0,
            "best_trade_pnl": 0.0,
            "worst_trade_pnl": 0.0,
            "start_capital": round(start_capital, 2),
            "ending_capital_from_pnl": round(start_capital, 2),
        }

    wins = [x for x in returns if x > 0]
    losses = [x for x in returns if x < 0]
    zeros = [x for x in returns if x == 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    pf = gross_profit / gross_loss if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    avg_win = mean(wins) if wins else 0.0
    avg_loss = abs(mean(losses)) if losses else 0.0
    payoff = avg_win / avg_loss if avg_loss > 0 else (999.0 if avg_win > 0 else 0.0)

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= max(0.000001, 1.0 + r)
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak else 0.0)

    max_wins, max_losses = _streaks(returns)
    ordered_returns = sorted(returns)

    durations: list[float] = []
    for trade in ordered:
        a = _dt(trade.get("entry_time"))
        b = _dt(trade.get("exit_time"))
        if a and b and b >= a:
            durations.append((b - a).total_seconds() / 60.0)

    r_wins = [x for x in r_values if x > 0]
    r_losses = [x for x in r_values if x < 0]
    pnl_wins = [x for x in pnl_values if x > 0]
    pnl_losses = [x for x in pnl_values if x < 0]

    return {
        "trades": len(returns),
        "trading_days": days,
        "avg_trades_per_day": round(len(returns) / days, 3) if days else 0.0,
        "win_rate_pct": round(100.0 * len(wins) / len(returns), 2),
        "loss_rate_pct": round(100.0 * len(losses) / len(returns), 2),
        "breakeven_trades": len(zeros),
        "expectancy_bps": round(mean(returns) * 10_000.0, 3),
        "expectancy_return_pct": round(mean(returns) * 100.0, 4),
        "profit_factor": round(min(pf, 999.0), 3),
        "avg_win_bps": round(avg_win * 10_000.0, 3),
        "avg_loss_bps": round(avg_loss * 10_000.0, 3),
        "avg_win_pct": round(avg_win * 100.0, 4),
        "avg_loss_pct": round(avg_loss * 100.0, 4),
        "payoff_ratio": round(min(payoff, 999.0), 3),
        "expectancy_r": round(mean(r_values), 3) if r_values else 0.0,
        "avg_win_r": round(mean(r_wins), 3) if r_wins else 0.0,
        "avg_loss_r": round(abs(mean(r_losses)), 3) if r_losses else 0.0,
        "best_trade_bps": round(max(returns) * 10_000.0, 3),
        "worst_trade_bps": round(min(returns) * 10_000.0, 3),
        "best_trade_r": round(max(r_values), 3) if r_values else 0.0,
        "worst_trade_r": round(min(r_values), 3) if r_values else 0.0,
        "median_trade_bps": round(median(returns) * 10_000.0, 3),
        "p10_trade_bps": round(_percentile(ordered_returns, 0.10) * 10_000.0, 3),
        "p90_trade_bps": round(_percentile(ordered_returns, 0.90) * 10_000.0, 3),
        "max_consecutive_wins": max_wins,
        "max_consecutive_losses": max_losses,
        "compounded_trade_return_pct": round((equity - 1.0) * 100.0, 3),
        "max_drawdown_pct": round(max_dd * 100.0, 3),
        "avg_duration_minutes": round(mean(durations), 2) if durations else 0.0,
        "gross_profit": round(sum(pnl_wins), 4) if pnl_values else 0.0,
        "gross_loss": round(abs(sum(pnl_losses)), 4) if pnl_values else 0.0,
        "net_pnl": round(sum(pnl_values), 4) if pnl_values else 0.0,
        "avg_win_pnl": round(mean(pnl_wins), 4) if pnl_wins else 0.0,
        "avg_loss_pnl": round(abs(mean(pnl_losses)), 4) if pnl_losses else 0.0,
        "best_trade_pnl": round(max(pnl_values), 4) if pnl_values else 0.0,
        "worst_trade_pnl": round(min(pnl_values), 4) if pnl_values else 0.0,
        "start_capital": round(start_capital, 2),
        "ending_capital_from_pnl": round(start_capital + sum(pnl_values), 2) if pnl_values else round(start_capital, 2),
    }
