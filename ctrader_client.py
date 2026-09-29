from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlencode
from uuid import uuid4

import httpx
import websockets

from config import Settings


class CTraderError(RuntimeError):
    pass


class CTraderClient:
    """Small asyncio JSON transport for cTrader Open API.

    The connector is safe to ship before credentials exist: status/OAuth URL
    work with configuration only, while socket authentication is attempted only
    when explicitly requested.
    """

    AUTH_BASE = "https://id.ctrader.com/my/settings/openapi/grantingaccess/"
    TOKEN_URL = "https://openapi.ctrader.com/apps/token"

    def __init__(self, settings: Settings):
        self.settings = settings
        self._http = httpx.AsyncClient(timeout=30.0)
        self._ws = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self.connected = False
        self.application_authenticated = False
        self.account_authenticated = False
        self.account_id: Optional[int] = None
        self.last_error: Optional[str] = None

    @property
    def endpoint(self) -> str:
        host = "demo.ctraderapi.com" if self.settings.ctrader_environment == "demo" else "live.ctraderapi.com"
        return f"wss://{host}:5036"

    @property
    def oauth_ready(self) -> bool:
        return bool(
            self.settings.ctrader_client_id
            and self.settings.ctrader_client_secret
            and self.settings.ctrader_redirect_uri
        )

    @property
    def api_ready(self) -> bool:
        return bool(self.oauth_ready and self.settings.ctrader_access_token)

    def public_state(self) -> dict:
        return {
            "environment": self.settings.ctrader_environment,
            "endpoint": self.endpoint,
            "oauth_ready": self.oauth_ready,
            "access_token_configured": bool(self.settings.ctrader_access_token),
            "refresh_token_configured": bool(self.settings.ctrader_refresh_token),
            "account_id_configured": bool(self.settings.ctrader_account_id),
            "connected": self.connected,
            "application_authenticated": self.application_authenticated,
            "account_authenticated": self.account_authenticated,
            "account_id": self.account_id,
            "last_error": self.last_error,
        }

    def authorization_url(self, scope: str = "trading") -> str:
        if not self.settings.ctrader_client_id or not self.settings.ctrader_redirect_uri:
            raise CTraderError("CTRADER_CLIENT_ID and CTRADER_REDIRECT_URI are required")
        if scope not in {"accounts", "trading"}:
            raise CTraderError("scope must be 'accounts' or 'trading'")
        return self.AUTH_BASE + "?" + urlencode({
            "client_id": self.settings.ctrader_client_id,
            "redirect_uri": self.settings.ctrader_redirect_uri,
            "scope": scope,
            "product": "web",
        })

    async def exchange_code(self, code: str) -> dict:
        if not self.oauth_ready:
            raise CTraderError("cTrader OAuth application credentials are not configured")
        response = await self._http.get(self.TOKEN_URL, params={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.settings.ctrader_redirect_uri,
            "client_id": self.settings.ctrader_client_id,
            "client_secret": self.settings.ctrader_client_secret,
        })
        response.raise_for_status()
        data = response.json()
        if data.get("errorCode"):
            raise CTraderError(f"{data.get('errorCode')}: {data.get('description')}")
        return data

    async def refresh_access_token(self, refresh_token: Optional[str] = None) -> dict:
        token = refresh_token or self.settings.ctrader_refresh_token
        if not self.oauth_ready or not token:
            raise CTraderError("cTrader refresh token or application credentials are missing")
        response = await self._http.post(self.TOKEN_URL, params={
            "grant_type": "refresh_token",
            "refresh_token": token,
            "client_id": self.settings.ctrader_client_id,
            "client_secret": self.settings.ctrader_client_secret,
        })
        response.raise_for_status()
        data = response.json()
        if data.get("errorCode"):
            raise CTraderError(f"{data.get('errorCode')}: {data.get('description')}")
        return data

    async def _send(self, payload_type: int, payload: Optional[dict] = None) -> str:
        if not self._ws:
            raise CTraderError("cTrader socket is not connected")
        client_msg_id = str(uuid4())
        await self._ws.send(json.dumps({
            "clientMsgId": client_msg_id,
            "payloadType": payload_type,
            "payload": payload or {},
        }))
        return client_msg_id

    async def _recv_until(self, expected_payload_type: int, timeout: float = 15.0) -> dict:
        if not self._ws:
            raise CTraderError("cTrader socket is not connected")
        while True:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=timeout)
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            message = json.loads(raw)
            payload_type = int(message.get("payloadType") or 0)
            if payload_type == 50:
                payload = message.get("payload") or {}
                raise CTraderError(
                    f"{payload.get('errorCode', 'OPEN_API_ERROR')}: "
                    f"{payload.get('description', '')}"
                )
            if payload_type == expected_payload_type:
                return message

    async def connect_and_authenticate(self, access_token: Optional[str] = None) -> dict:
        token = access_token or self.settings.ctrader_access_token
        if not self.oauth_ready or not token:
            raise CTraderError("cTrader client credentials and access token are required")

        await self.disconnect()
        self.last_error = None
        try:
            self._ws = await websockets.connect(
                self.endpoint,
                open_timeout=15,
                close_timeout=5,
                ping_interval=None,
            )
            self.connected = True

            await self._send(2100, {
                "clientId": self.settings.ctrader_client_id,
                "clientSecret": self.settings.ctrader_client_secret,
            })
            await self._recv_until(2101)
            self.application_authenticated = True

            await self._send(2149, {"accessToken": token})
            accounts_message = await self._recv_until(2150)
            accounts = (accounts_message.get("payload") or {}).get("ctidTraderAccount") or []
            want_live = self.settings.ctrader_environment == "live"
            eligible_accounts = [
                row for row in accounts
                if row.get("ctidTraderAccountId") is not None
                and bool(row.get("isLive", False)) == want_live
            ]
            ids = [int(row["ctidTraderAccountId"]) for row in eligible_accounts]
            if not ids:
                raise CTraderError(
                    f"No {self.settings.ctrader_environment} cTrader accounts are authorized for this token"
                )

            configured = int(self.settings.ctrader_account_id) if self.settings.ctrader_account_id else None
            if configured is not None and configured not in ids:
                raise CTraderError("Configured CTRADER_ACCOUNT_ID is not authorized by this token")
            self.account_id = configured or ids[0]

            await self._send(2102, {
                "ctidTraderAccountId": self.account_id,
                "accessToken": token,
            })
            await self._recv_until(2103)
            self.account_authenticated = True
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(), name="ctrader-heartbeat"
            )
            return self.public_state()
        except Exception as exc:
            self.last_error = str(exc)
            await self.disconnect()
            raise

    async def _heartbeat_loop(self) -> None:
        try:
            while self.connected and self._ws:
                await asyncio.sleep(10)
                await self._send(51, {})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_error = str(exc)
            self.connected = False

    async def symbols(self) -> list[dict]:
        if not self.account_authenticated or not self.account_id:
            await self.connect_and_authenticate()
        await self._send(2114, {
            "ctidTraderAccountId": self.account_id,
            "includeArchivedSymbols": False,
        })
        message = await self._recv_until(2115)
        return list((message.get("payload") or {}).get("symbol") or [])

    async def symbol_details(self, symbol_ids: list[int]) -> list[dict]:
        if not self.account_authenticated or not self.account_id:
            await self.connect_and_authenticate()
        if not symbol_ids:
            return []
        await self._send(2116, {
            "ctidTraderAccountId": self.account_id,
            "symbolId": [int(x) for x in symbol_ids],
        })
        message = await self._recv_until(2117)
        return list((message.get("payload") or {}).get("symbol") or [])

    async def resolve_symbol(self, pair: str) -> dict:
        wanted = pair.upper().replace("/", "").replace("-", "").replace("_", "")
        symbols = await self.symbols()
        for row in symbols:
            name = str(row.get("symbolName") or "")
            normalized = name.upper().replace("/", "").replace("-", "").replace("_", "")
            if normalized == wanted:
                details = await self.symbol_details([int(row["symbolId"])])
                detail = details[0] if details else {}
                return {**row, **detail}
        raise CTraderError(f"Symbol {pair} is not available on this cTrader account")

    async def historical_bars(
        self,
        pair: str,
        *,
        timeframe_min: int = 1,
        max_bars: int = 5000,
        lookback_days: int = 90,
    ) -> list[dict]:
        """Load cTrader trendbars and normalize them to MicroTrader OHLC format."""
        period_map = {
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 10: 6, 15: 7,
            30: 8, 60: 9, 240: 10, 720: 11, 1440: 12,
        }
        if timeframe_min not in period_map:
            raise CTraderError(f"Unsupported cTrader timeframe: {timeframe_min} minutes")
        if not self.account_authenticated or not self.account_id:
            await self.connect_and_authenticate()

        symbol = await self.resolve_symbol(pair)
        symbol_id = int(symbol["symbolId"])
        digits = int(symbol.get("digits") or 5)
        period = period_map[timeframe_min]

        now = datetime.now(timezone.utc)
        from_ms = int((now - timedelta(days=max(1, lookback_days))).timestamp() * 1000)
        cursor_to = int(now.timestamp() * 1000)
        out: list[dict] = []

        while len(out) < max_bars:
            remaining = max_bars - len(out)
            await self._send(2137, {
                "ctidTraderAccountId": self.account_id,
                "fromTimestamp": from_ms,
                "toTimestamp": cursor_to,
                "period": period,
                "symbolId": symbol_id,
                "count": min(5000, remaining),
            })
            message = await self._recv_until(2138, timeout=30.0)
            payload = message.get("payload") or {}
            trendbars = list(payload.get("trendbar") or [])
            if not trendbars:
                break

            batch: list[dict] = []
            for row in trendbars:
                low_raw = int(row.get("low") or 0)
                low = round(low_raw / 100000.0, digits)
                open_px = round((low_raw + int(row.get("deltaOpen") or 0)) / 100000.0, digits)
                high = round((low_raw + int(row.get("deltaHigh") or 0)) / 100000.0, digits)
                close = round((low_raw + int(row.get("deltaClose") or 0)) / 100000.0, digits)
                minute_ts = int(row.get("utcTimestampInMinutes") or 0)
                ts = datetime.fromtimestamp(minute_ts * 60, tz=timezone.utc).isoformat()
                batch.append({
                    "t": ts,
                    "o": open_px,
                    "h": high,
                    "l": low,
                    "c": close,
                    "v": float(row.get("volume") or 0),
                    "pair": pair,
                })

            batch.sort(key=lambda x: x["t"])
            existing = {x["t"] for x in out}
            new_rows = [x for x in batch if x["t"] not in existing]
            out = new_rows + out
            out.sort(key=lambda x: x["t"])
            if not payload.get("hasMore") or not new_rows:
                break
            oldest = datetime.fromisoformat(new_rows[0]["t"])
            next_cursor = int((oldest - timedelta(milliseconds=1)).timestamp() * 1000)
            if next_cursor >= cursor_to:
                break
            cursor_to = next_cursor
            await asyncio.sleep(0.22)

        return out[-max_bars:]

    async def disconnect(self) -> None:
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        self._heartbeat_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self.connected = False
        self.application_authenticated = False
        self.account_authenticated = False
        self.account_id = None

    async def close(self) -> None:
        await self.disconnect()
        await self._http.aclose()
