from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean
from typing import List, Tuple
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")
NEW_YORK = ZoneInfo("America/New_York")


def _dt(raw) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def active_session(raw_time) -> str:
    """Return the active entry session, DST-aware, or an empty string.

    Entries are allowed during liquid London and New York daytime windows.
    Risk management/exits should continue outside these entry windows.
    """
    dt = _dt(raw_time)
    if not dt or dt.weekday() >= 5:
        return ""

    london = dt.astimezone(LONDON)
    ny = dt.astimezone(NEW_YORK)
    london_open = 7 <= london.hour < 16
    ny_open = 8 <= ny.hour < 16

    if london_open and ny_open:
        return "London+New York"
    if london_open:
        return "London"
    if ny_open:
        return "New York"
    return ""


def relative_volume(
    bars: List[dict],
    entry_idx: int,
    window: int = 50,
) -> float | None:
    """Use only completed bars BEFORE the proposed entry to avoid look-ahead."""
    if entry_idx < 2:
        return None

    current = float(bars[entry_idx - 1].get("v") or 0.0)
    hist = [
        float(x.get("v") or 0.0)
        for x in bars[max(0, entry_idx - window - 1): entry_idx - 1]
        if float(x.get("v") or 0.0) > 0
    ]
    if current <= 0 or len(hist) < 10:
        # Some providers do not expose usable tick volume. In that case the
        # session filter still applies, but volume does not falsely reject all.
        return None
    baseline = mean(hist)
    return (current / baseline) if baseline > 0 else None


def entry_allowed(
    bars: List[dict],
    entry_idx: int,
    *,
    min_volume_ratio: float = 0.70,
    volume_window: int = 50,
) -> Tuple[bool, str, float | None]:
    if entry_idx < 0 or entry_idx >= len(bars):
        return False, "", None

    session = active_session(bars[entry_idx].get("t"))
    if not session:
        return False, "", None

    ratio = relative_volume(bars, entry_idx, volume_window)
    if ratio is not None and ratio < float(min_volume_ratio):
        return False, session, ratio

    return True, session, ratio
