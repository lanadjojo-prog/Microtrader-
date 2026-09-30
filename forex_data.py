from __future__ import annotations

import asyncio
import lzma
import logging
import struct
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Tuple

import httpx

from config import Settings

log = logging.getLogger("microtrader.forex_data")

QuoteTick = Tuple[int, float, float]


class ForexDataError(RuntimeError):
    pass


@dataclass
class ForexDataStats:
    bars_requests: int = 0
    tick_requests: int = 0
    dukascopy_files_loaded: int = 0
    dukascopy_missing_files: int = 0
    twelve_requests: int = 0
    last_source: str = ""
    last_error: str = ""


class ExternalForexData:
    """Read-only market data used only for research/backtests.

    Dukascopy is the zero-key historical source and provides bid/ask ticks.
    Twelve Data is an optional recent-bar overlay when TWELVE_DATA_API_KEY is set.
    Nothing in this class can place or modify broker orders.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.stats = ForexDataStats()
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=20.0),
            follow_redirects=True,
            headers={"User-Agent": "MicroTrader-research/1.0"},
        )
        self._bars_cache: Dict[tuple, List[dict]] = {}
        self._ticks_cache: Dict[tuple, List[QuoteTick]] = {}
        self._dukascopy_day_cache: Dict[tuple, List[QuoteTick]] = {}

    async def close(self) -> None:
        await self._client.aclose()

    def public_state(self) -> dict:
        return {
            "mode": "research-only",
            "forex_provider": self.settings.forex_data_provider,
            "precision_provider": self.settings.precision_data_provider,
            "twelve_data_configured": bool(self.settings.twelve_data_api_key),
            "dukascopy_configured": bool(self.settings.dukascopy_base_urls),
            "bars_requests": self.stats.bars_requests,
            "tick_requests": self.stats.tick_requests,
            "dukascopy_files_loaded": self.stats.dukascopy_files_loaded,
            "dukascopy_missing_files": self.stats.dukascopy_missing_files,
            "twelve_requests": self.stats.twelve_requests,
            "last_source": self.stats.last_source,
            "last_error": self.stats.last_error,
        }

    async def historical_bars(
        self,
        pair: str,
        *,
        max_bars: int,
        lookback_days: int,
    ) -> List[dict]:
        pair = pair.upper().strip()
        max_bars = max(1, int(max_bars))
        lookback_days = max(1, int(lookback_days))
        key = (pair, max_bars, lookback_days, self.settings.forex_data_provider)
        if key in self._bars_cache:
            return list(self._bars_cache[key])

        self.stats.bars_requests += 1
        mode = self.settings.forex_data_provider
        if mode not in {"external", "auto", "dukascopy", "twelve"}:
            raise ForexDataError(f"Unsupported FOREX_DATA_PROVIDER={mode}")

        bars: List[dict] = []
        errors: List[str] = []

        if mode in {"external", "auto", "dukascopy"}:
            try:
                bars = await self._dukascopy_bars(
                    pair,
                    max_bars=max_bars,
                    lookback_days=lookback_days,
                )
                if bars:
                    self.stats.last_source = "dukascopy"
            except Exception as exc:
                errors.append(f"Dukascopy: {exc}")
                log.warning("Dukascopy bars failed for %s: %s", pair, exc)

        if mode in {"external", "auto", "twelve"} and self.settings.twelve_data_api_key:
            try:
                recent = await self._twelve_bars(pair, max_bars=min(max_bars, 5000))
                if recent:
                    bars = merge_bars(bars, recent)
                    self.stats.last_source = (
                        "dukascopy+twelve" if bars and mode != "twelve" else "twelve"
                    )
            except Exception as exc:
                errors.append(f"Twelve Data: {exc}")
                log.warning("Twelve Data bars failed for %s: %s", pair, exc)

        if mode == "twelve" and not self.settings.twelve_data_api_key:
            errors.append("TWELVE_DATA_API_KEY is not configured")

        bars = sorted(bars, key=lambda row: str(row.get("t") or ""))[-max_bars:]
        if not bars:
            message = "; ".join(errors) or "No external FX bars were returned"
            self.stats.last_error = message
            raise ForexDataError(message)

        self.stats.last_error = ""
        self._bars_cache[key] = list(bars)
        return bars

    async def historical_quote_ticks(
        self,
        pair: str,
        *,
        lookback_days: int,
        max_ticks: int,
    ) -> List[QuoteTick]:
        pair = pair.upper().strip()
        lookback_days = max(1, int(lookback_days))
        max_ticks = max(1, int(max_ticks))
        key = (pair, lookback_days, max_ticks)
        if key in self._ticks_cache:
            return list(self._ticks_cache[key])

        self.stats.tick_requests += 1
        ticks = await self._dukascopy_ticks(
            pair,
            lookback_days=lookback_days,
            max_ticks=max_ticks,
        )
        if not ticks:
            raise ForexDataError(f"No Dukascopy ticks were returned for {pair}")
        self.stats.last_source = "dukascopy-ticks"
        self._ticks_cache[key] = list(ticks)
        return ticks

    async def _twelve_bars(self, pair: str, *, max_bars: int) -> List[dict]:
        self.stats.twelve_requests += 1
        response = await self._client.get(
            self.settings.twelve_data_base_url.rstrip("/") + "/time_series",
            params={
                "symbol": pair,
                "interval": "1min",
                "outputsize": min(max(1, max_bars), 5000),
                "order": "asc",
                "timezone": "UTC",
                "format": "JSON",
                "apikey": self.settings.twelve_data_api_key,
            },
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "error":
            raise ForexDataError(str(payload.get("message") or payload))
        values = payload.get("values") or []
        out: List[dict] = []
        for row in values:
            raw_time = str(row.get("datetime") or "").strip()
            if not raw_time:
                continue
            try:
                dt = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                out.append({
                    "t": dt.isoformat(),
                    "o": float(row["open"]),
                    "h": float(row["high"]),
                    "l": float(row["low"]),
                    "c": float(row["close"]),
                    "v": float(row.get("volume") or 0),
                })
            except (KeyError, TypeError, ValueError):
                continue
        return out

    async def _dukascopy_bars(
        self,
        pair: str,
        *,
        max_bars: int,
        lookback_days: int,
    ) -> List[dict]:
        collected: List[dict] = []
        today = datetime.now(timezone.utc).date()
        for offset in range(0, lookback_days + 1):
            day = today - timedelta(days=offset)
            if day.weekday() >= 5:
                continue
            day_ticks = await self._dukascopy_day_ticks(pair, day)
            if not day_ticks:
                continue
            collected.extend(bars_from_quote_ticks(day_ticks))
            if len(collected) >= max_bars + 1500:
                break
        return sorted(collected, key=lambda row: str(row.get("t") or ""))[-max_bars:]

    async def _dukascopy_ticks(
        self,
        pair: str,
        *,
        lookback_days: int,
        max_ticks: int,
    ) -> List[QuoteTick]:
        collected: List[QuoteTick] = []
        today = datetime.now(timezone.utc).date()
        for offset in range(0, lookback_days + 1):
            day = today - timedelta(days=offset)
            if day.weekday() >= 5:
                continue
            day_ticks = await self._dukascopy_day_ticks(pair, day)
            if day_ticks:
                collected.extend(day_ticks)
            if len(collected) >= max_ticks:
                break
        collected.sort(key=lambda item: item[0])
        return collected[-max_ticks:]

    async def _dukascopy_day_ticks(self, pair: str, day: date) -> List[QuoteTick]:
        key = (pair.upper(), day.isoformat())
        if key in self._dukascopy_day_cache:
            return list(self._dukascopy_day_cache[key])

        # Dukascopy tick history is stored in 24 hourly files, not one daily file:
        # .../{PAIR}/{year}/{zero_based_month}/{day}/{hour}h_ticks.bi5
        # Fetch a day with bounded concurrency so Precision can build a coherent
        # bid/ask stream without hammering the source.
        semaphore = asyncio.Semaphore(6)

        async def load_hour(hour: int) -> List[QuoteTick]:
            async with semaphore:
                blob = await self._dukascopy_hour(pair, day, hour)
            if not blob:
                return []
            return decode_dukascopy_ticks(blob, pair, day, hour)

        chunks = await asyncio.gather(*(load_hour(hour) for hour in range(24)))
        ticks = [tick for chunk in chunks for tick in chunk]
        ticks.sort(key=lambda item: item[0])
        self._dukascopy_day_cache[key] = list(ticks)
        # Keep the tiny in-memory cache bounded on long-running workers.
        if len(self._dukascopy_day_cache) > 20:
            first_key = next(iter(self._dukascopy_day_cache))
            self._dukascopy_day_cache.pop(first_key, None)
        return ticks

    async def _dukascopy_hour(self, pair: str, day: date, hour: int) -> bytes:
        instrument = pair.replace("/", "").upper()
        month_zero_based = day.month - 1
        relative = (
            f"{instrument}/{day.year}/{month_zero_based:02d}/"
            f"{day.day:02d}/{int(hour):02d}h_ticks.bi5"
        )
        last_error = ""
        for base in self.settings.dukascopy_base_urls:
            url = base.rstrip("/") + "/" + relative
            try:
                response = await self._client.get(url)
                if response.status_code == 404:
                    continue
                response.raise_for_status()
                if response.content:
                    self.stats.dukascopy_files_loaded += 1
                    return response.content
            except Exception as exc:
                last_error = str(exc)
                continue
        self.stats.dukascopy_missing_files += 1
        if last_error:
            log.debug("Dukascopy file unavailable %s: %s", relative, last_error)
        return b""


def decode_dukascopy_ticks(blob: bytes, pair: str, day: date, hour: int = 0) -> List[QuoteTick]:
    try:
        raw = lzma.decompress(blob)
    except lzma.LZMAError as exc:
        raise ForexDataError(f"Unable to decompress Dukascopy .bi5 data: {exc}") from exc

    record_size = 20
    if len(raw) < record_size:
        return []
    usable = len(raw) - (len(raw) % record_size)
    scale = 1000.0 if pair.upper().endswith("/JPY") or pair.upper().endswith("JPY") else 100000.0
    hour_start_ms = int(datetime(day.year, day.month, day.day, int(hour), tzinfo=timezone.utc).timestamp() * 1000)
    out: List[QuoteTick] = []
    for offset in range(0, usable, record_size):
        ms, ask_i, bid_i, _ask_vol, _bid_vol = struct.unpack(">IIIff", raw[offset:offset + record_size])
        ask = ask_i / scale
        bid = bid_i / scale
        if ask <= 0 or bid <= 0:
            continue
        out.append((hour_start_ms + int(ms), bid, ask))
    return out


def bars_from_quote_ticks(ticks: List[QuoteTick]) -> List[dict]:
    buckets: Dict[int, dict] = {}
    for ts_ms, bid, ask in ticks:
        minute_ms = (int(ts_ms) // 60000) * 60000
        mid = (float(bid) + float(ask)) / 2.0
        row = buckets.get(minute_ms)
        if row is None:
            buckets[minute_ms] = {
                "t": datetime.fromtimestamp(minute_ms / 1000, tz=timezone.utc).isoformat(),
                "o": mid,
                "h": mid,
                "l": mid,
                "c": mid,
                "v": 1.0,
            }
        else:
            row["h"] = max(float(row["h"]), mid)
            row["l"] = min(float(row["l"]), mid)
            row["c"] = mid
            row["v"] = float(row.get("v") or 0) + 1.0
    return [buckets[key] for key in sorted(buckets)]


def merge_bars(older: List[dict], newer: List[dict]) -> List[dict]:
    merged: Dict[str, dict] = {}
    for row in older:
        key = str(row.get("t") or "")
        if key:
            merged[key] = dict(row)
    for row in newer:
        key = str(row.get("t") or "")
        if key:
            merged[key] = dict(row)
    return [merged[key] for key in sorted(merged)]
