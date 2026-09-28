from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
import httpx

from config import Settings


class AlpacaError(RuntimeError):
    pass


class AlpacaClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = httpx.AsyncClient(timeout=30.0)

    @property
    def headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.settings.api_key,
            "APCA-API-SECRET-KEY": self.settings.api_secret,
            "Content-Type": "application/json",
        }

    async def close(self):
        await self._client.aclose()

    async def _request(self, method: str, url: str, **kwargs):
        response = await self._client.request(method, url, headers=self.headers, **kwargs)
        if response.status_code >= 400:
            raise AlpacaError(f"{method} {url} -> {response.status_code}: {response.text[:500]}")
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    async def account(self) -> dict:
        return await self._request("GET", f"{self.settings.trading_base_url}/v2/account")

    async def clock(self) -> dict:
        return await self._request("GET", f"{self.settings.trading_base_url}/v2/clock")

    async def positions(self) -> List[dict]:
        return await self._request("GET", f"{self.settings.trading_base_url}/v2/positions")

    async def orders_today(self) -> List[dict]:
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        params = {"status": "all", "after": start, "direction": "desc", "limit": 500}
        return await self._request("GET", f"{self.settings.trading_base_url}/v2/orders", params=params)

    async def get_order(self, order_id: str) -> dict:
        return await self._request("GET", f"{self.settings.trading_base_url}/v2/orders/{order_id}")

    async def submit_market_buy(self, symbol: str, notional: float) -> dict:
        payload = {
            "symbol": symbol,
            "notional": round(float(notional), 2),
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": f"micro-{symbol.lower()}-{int(datetime.now(timezone.utc).timestamp())}",
        }
        return await self._request("POST", f"{self.settings.trading_base_url}/v2/orders", json=payload)

    async def close_position(self, symbol: str) -> Optional[dict]:
        return await self._request("DELETE", f"{self.settings.trading_base_url}/v2/positions/{symbol}")

    async def cancel_all_orders(self):
        return await self._request("DELETE", f"{self.settings.trading_base_url}/v2/orders")

    async def close_all_positions(self):
        return await self._request("DELETE", f"{self.settings.trading_base_url}/v2/positions", params={"cancel_orders": "true"})

    async def latest_bars(self, symbols: List[str], limit: int) -> Dict[str, List[dict]]:
        params = {
            "symbols": ",".join(symbols),
            "timeframe": "1Min",
            "limit": max(limit * len(symbols), 100),
            "adjustment": "raw",
            "feed": self.settings.data_feed,
            "sort": "desc",
        }
        data = await self._request("GET", f"{self.settings.data_base_url}/v2/stocks/bars", params=params)
        bars = data.get("bars", {}) if isinstance(data, dict) else {}
        normalized: Dict[str, List[dict]] = {}
        for symbol in symbols:
            items = list(bars.get(symbol, []))[:limit]
            items.sort(key=lambda x: x.get("t", ""))
            normalized[symbol] = items
        return normalized

    async def historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "5Min",
        max_bars: int = 5000,
    ) -> List[dict]:
        """Fetch chronologically ordered historical bars with Alpaca pagination."""
        url = f"{self.settings.data_base_url}/v2/stocks/{symbol}/bars"
        page_token: Optional[str] = None
        bars: List[dict] = []

        while len(bars) < max_bars:
            params = {
                "timeframe": timeframe,
                "start": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "end": end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "limit": min(10000, max_bars - len(bars)),
                "adjustment": "raw",
                "feed": self.settings.data_feed,
                "sort": "asc",
            }
            if page_token:
                params["page_token"] = page_token

            data = await self._request("GET", url, params=params)
            page = list((data or {}).get("bars", []))
            bars.extend(page)
            page_token = (data or {}).get("next_page_token")
            if not page_token or not page:
                break

        bars.sort(key=lambda x: x.get("t", ""))
        return bars[:max_bars]
