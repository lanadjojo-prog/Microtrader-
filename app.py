from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse

from alpaca_client import AlpacaClient
from config import settings
from engine import TradingEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

client = AlpacaClient(settings)
engine = TradingEngine(settings, client)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.auto_start:
        await engine.start()
    yield
    await engine.stop()
    await client.close()


app = FastAPI(title="MicroTrader", version="0.1.0", lifespan=lifespan)


def require_token(authorization: str | None):
    if not settings.dashboard_token:
        raise HTTPException(status_code=503, detail="DASHBOARD_TOKEN is not configured")
    expected = f"Bearer {settings.dashboard_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
async def health():
    return {"ok": True, "paper": settings.paper, "engine_running": engine.state.running}


@app.get("/api/status")
async def status(authorization: str | None = Header(default=None)):
    require_token(authorization)
    payload = engine.public_state()
    payload.update({
        "paper": settings.paper,
        "live_trading_enabled": settings.live_trading_enabled,
        "can_trade": settings.can_trade,
        "symbols": settings.symbols,
        "trade_notional": settings.trade_notional,
        "max_open_positions": settings.max_open_positions,
        "max_trades_per_day": settings.max_trades_per_day,
        "max_daily_loss": settings.max_daily_loss,
    })
    if settings.api_key and settings.api_secret:
        try:
            acct = await client.account()
            payload["account"] = {
                "status": acct.get("status"),
                "equity": acct.get("equity"),
                "cash": acct.get("cash"),
                "buying_power": acct.get("buying_power"),
                "trading_blocked": acct.get("trading_blocked"),
            }
            payload["positions"] = await client.positions()
        except Exception as exc:
            payload["broker_error"] = str(exc)
    return payload


@app.get("/api/executions")
async def executions(authorization: str | None = Header(default=None)):
    require_token(authorization)
    return {"executions": engine.executions()}


@app.post("/api/start")
async def start(authorization: str | None = Header(default=None)):
    require_token(authorization)
    await engine.start()
    return {"ok": True, "running": True}


@app.post("/api/stop")
async def stop(authorization: str | None = Header(default=None)):
    require_token(authorization)
    await engine.stop()
    return {"ok": True, "running": False}


@app.post("/api/flatten")
async def flatten(authorization: str | None = Header(default=None)):
    require_token(authorization)
    if not settings.can_trade:
        raise HTTPException(status_code=403, detail="Live safety block is active")
    await client.cancel_all_orders()
    await client.close_all_positions()
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD)


DASHBOARD = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>MicroTrader</title>
<style>
body{font-family:system-ui,-apple-system,sans-serif;background:#0e1116;color:#e9eef5;margin:0;padding:28px;max-width:1000px;margin:auto}
.card{background:#171c24;border:1px solid #2a3442;border-radius:16px;padding:18px;margin:14px 0}.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}button,input{font:inherit;border-radius:10px;border:1px solid #3a4658;padding:10px 13px;background:#0f141b;color:#fff}button{cursor:pointer}button.danger{border-color:#8a3841}.muted{color:#96a4b5}pre{white-space:pre-wrap;word-break:break-word}h1{margin-bottom:0}.pill{padding:5px 9px;border-radius:999px;background:#242d39}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}.metric{background:#111720;padding:12px;border-radius:12px}.metric b{display:block;font-size:1.3rem;margin-top:4px}</style>
</head>
<body>
<h1>MicroTrader</h1><p class="muted">Alpaca micro-trading engine · safe-by-default</p>
<div class="card">
  <div class="row"><input id="token" type="password" placeholder="Dashboard token" style="min-width:260px"><button onclick="loadStatus()">Connect</button><button onclick="action('start')">Start</button><button onclick="action('stop')">Stop</button><button class="danger" onclick="action('flatten')">Flatten</button></div>
</div>
<div class="card"><div id="headline" class="row"></div><div id="metrics" class="grid" style="margin-top:12px"></div></div>
<div class="card"><h3>Positions</h3><pre id="positions">Not connected.</pre></div>
<div class="card"><h3>Recent executions</h3><pre id="executions">Not connected.</pre></div>
<div class="card"><h3>Raw status</h3><pre id="raw"></pre></div>
<script>
const token=()=>document.getElementById('token').value;
async function api(path,method='GET'){
 const r=await fetch('/api/'+path,{method,headers:{Authorization:'Bearer '+token()}});
 const j=await r.json(); if(!r.ok) throw new Error(j.detail||JSON.stringify(j)); return j;
}
async function loadStatus(){
 try{const s=await api('status');
  document.getElementById('headline').innerHTML=`<span class="pill">${s.paper?'PAPER':'LIVE'}</span><span class="pill">engine ${s.running?'RUNNING':'STOPPED'}</span><span class="pill">orders ${s.submitted_orders}</span>`;
  const a=s.account||{}; const vals={Equity:a.equity||'-',Cash:a.cash||'-','Trade size':'$'+s.trade_notional,'Daily loss cap':'$'+s.max_daily_loss,'Trade cap':s.max_trades_per_day,'Open-position cap':s.max_open_positions};
  document.getElementById('metrics').innerHTML=Object.entries(vals).map(([k,v])=>`<div class="metric"><span class="muted">${k}</span><b>${v}</b></div>`).join('');
  document.getElementById('positions').textContent=JSON.stringify(s.positions||[],null,2);
  const ex=await api('executions'); document.getElementById('executions').textContent=JSON.stringify(ex.executions||[],null,2);
  document.getElementById('raw').textContent=JSON.stringify(s,null,2);
 }catch(e){alert(e.message)}
}
async function action(x){try{await api(x,'POST');await loadStatus()}catch(e){alert(e.message)}}
setInterval(()=>{if(token()) loadStatus()},15000);
</script>
</body></html>'''
