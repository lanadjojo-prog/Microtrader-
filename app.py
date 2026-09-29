from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse

from alpaca_client import AlpacaClient
from config import settings
from engine import TradingEngine
from strategy_lab import StrategyLab
from research_labs import ResearchLabs
from crypto_lab import CryptoMicrostructureLab
from research_agent import ResearchAgent
from research_coordinator import ResearchCoordinator
from ctrader_client import CTraderClient, CTraderError
from forex_lab import ForexStrategyLab
from precision_lab import PrecisionStrategyLab

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

client = AlpacaClient(settings)
engine = TradingEngine(settings, client)
lab = StrategyLab(settings, client)
research = ResearchLabs(settings, client)
crypto_lab = CryptoMicrostructureLab(settings)
research_agent = ResearchAgent(settings, lab)
coordinator = ResearchCoordinator(lab, research)
ctrader = CTraderClient(settings)
forex_lab = ForexStrategyLab(settings, ctrader)
precision_lab = PrecisionStrategyLab(settings, ctrader)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # FOREX_FIRST keeps the legacy stock/crypto code available but prevents it
    # from consuming CPU while the forex research stack owns the instance.
    if not settings.forex_first:
        if settings.auto_start:
            await engine.start()
        if settings.lab_auto_start and settings.api_key and settings.api_secret:
            await lab.start()
        await coordinator.start()
        if settings.crypto_lab_auto_start:
            await crypto_lab.start()
        if settings.research_agent_auto_start:
            await research_agent.start()

    if settings.forex_lab_auto_start and ctrader.api_ready:
        await forex_lab.start()
    elif settings.precision_lab_auto_start and ctrader.api_ready:
        await precision_lab.start()

    yield

    await precision_lab.stop()
    await forex_lab.stop()
    await ctrader.close()
    await coordinator.stop()
    await research_agent.stop()
    await engine.stop()
    await lab.stop()
    await research.stop()
    await crypto_lab.close()
    await client.close()


app = FastAPI(title="MicroTrader", version="0.2.0", lifespan=lifespan)


def require_token(authorization: str | None):
    if not settings.dashboard_token:
        raise HTTPException(status_code=503, detail="DASHBOARD_TOKEN is not configured")
    expected = f"Bearer {settings.dashboard_token}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
async def health():
    return {
        "ok": True,
        "mode": "forex-first" if settings.forex_first else "legacy-mixed",
        "paper": settings.paper,
        "engine_running": engine.state.running,
        "forex_lab_running": forex_lab.state.running,
        "precision_lab_running": precision_lab.state.running,
        "ctrader_ready": ctrader.api_ready,
    }


@app.get("/api/status")
async def status(authorization: str | None = Header(default=None)):
    require_token(authorization)
    payload = engine.public_state()
    payload.update({
        "mode": "forex-first" if settings.forex_first else "legacy-mixed",
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


@app.get("/api/lab/status")
async def lab_status(authorization: str | None = Header(default=None)):
    require_token(authorization)
    return lab.public_state()


@app.get("/api/lab/results")
async def lab_results(authorization: str | None = Header(default=None)):
    require_token(authorization)
    return {"results": lab.results(), "state": lab.public_state()}


@app.post("/api/lab/start")
async def lab_start(authorization: str | None = Header(default=None)):
    require_token(authorization)
    if settings.forex_first:
        raise HTTPException(status_code=409, detail="Stock Strategy Lab is disabled in FOREX_FIRST mode")
    if not settings.api_key or not settings.api_secret:
        raise HTTPException(status_code=503, detail="Alpaca API credentials are not configured")
    await lab.start()
    return {"ok": True, "running": True}


@app.post("/api/lab/stop")
async def lab_stop(authorization: str | None = Header(default=None)):
    require_token(authorization)
    await lab.stop()
    return {"ok": True, "running": False}


@app.get("/api/coordinator/status")
async def coordinator_status(authorization: str | None = Header(default=None)):
    require_token(authorization)
    return coordinator.public_state()


@app.get("/api/research/status")
async def research_status(authorization: str | None = Header(default=None)):
    require_token(authorization)
    return research.public_state()


@app.post("/api/research/start")
async def research_start(authorization: str | None = Header(default=None)):
    require_token(authorization)
    if settings.forex_first:
        raise HTTPException(status_code=409, detail="Legacy stock Research Labs are disabled in FOREX_FIRST mode")
    if not settings.api_key or not settings.api_secret:
        raise HTTPException(status_code=503, detail="Alpaca API credentials are not configured")
    await research.start()
    return {"ok": True, "running": True}


@app.post("/api/research/stop")
async def research_stop(authorization: str | None = Header(default=None)):
    require_token(authorization)
    await research.stop()
    return {"ok": True, "running": False}


@app.get("/api/agent/status")
async def agent_status(authorization: str | None = Header(default=None)):
    require_token(authorization)
    return research_agent.public_state()


@app.post("/api/agent/start")
async def agent_start(authorization: str | None = Header(default=None)):
    require_token(authorization)
    if settings.forex_first:
        raise HTTPException(status_code=409, detail="Legacy stock Research Agent is disabled in FOREX_FIRST mode")
    await research_agent.start()
    return {"ok": True, "running": True}


@app.post("/api/agent/stop")
async def agent_stop(authorization: str | None = Header(default=None)):
    require_token(authorization)
    await research_agent.stop()
    return {"ok": True, "running": False}


@app.post("/api/agent/cycle")
async def agent_cycle(authorization: str | None = Header(default=None)):
    require_token(authorization)
    if settings.forex_first:
        raise HTTPException(status_code=409, detail="Legacy stock Research Agent is disabled in FOREX_FIRST mode")
    await research_agent.cycle()
    return research_agent.public_state()


@app.get("/api/crypto-lab/status")
async def crypto_lab_status(authorization: str | None = Header(default=None)):
    require_token(authorization)
    return crypto_lab.public_state()


@app.post("/api/crypto-lab/start")
async def crypto_lab_start(authorization: str | None = Header(default=None)):
    require_token(authorization)
    if settings.forex_first:
        raise HTTPException(status_code=409, detail="Crypto Lab is disabled in FOREX_FIRST mode")
    await crypto_lab.start()
    return {"ok": True, "running": True}


@app.post("/api/crypto-lab/stop")
async def crypto_lab_stop(authorization: str | None = Header(default=None)):
    require_token(authorization)
    await crypto_lab.stop()
    return {"ok": True, "running": False}


@app.post("/api/start")
async def start(authorization: str | None = Header(default=None)):
    require_token(authorization)
    if settings.forex_first:
        raise HTTPException(status_code=409, detail="Alpaca trading engine is disabled in FOREX_FIRST mode")
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


@app.get("/api/ctrader/status")
async def ctrader_status(authorization: str | None = Header(default=None)):
    require_token(authorization)
    return ctrader.public_state()


@app.get("/api/ctrader/oauth-url")
async def ctrader_oauth_url(authorization: str | None = Header(default=None)):
    require_token(authorization)
    try:
        return {"url": ctrader.authorization_url()}
    except CTraderError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/api/ctrader/connect")
async def ctrader_connect(authorization: str | None = Header(default=None)):
    require_token(authorization)
    try:
        return await ctrader.connect_and_authenticate()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/api/ctrader/diagnostics")
async def ctrader_diagnostics(authorization: str | None = Header(default=None)):
    require_token(authorization)
    try:
        pairs = list(dict.fromkeys(settings.forex_pairs + settings.precision_pairs))
        return await ctrader.diagnostics(pairs)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/ctrader/callback", response_class=HTMLResponse)
async def ctrader_callback(code: str = ""):
    if not code:
        return HTMLResponse(
            "<h3>cTrader authorization failed</h3><p>No authorization code was returned.</p>",
            status_code=400,
        )
    try:
        token_data = await ctrader.exchange_code(code)
        expires = token_data.get("expiresIn")
        return HTMLResponse(
            "<h3>cTrader demo authorization complete</h3>"
            "<p>MicroTrader has loaded the access token into this running instance. "
            "You can close this window and test the connection from the dashboard.</p>"
            f"<p>Token lifetime reported by cTrader: {expires or '-'} seconds.</p>"
            "<p>No token or client secret is displayed on this page.</p>"
        )
    except Exception as exc:
        return HTMLResponse(
            "<h3>cTrader authorization failed</h3>"
            f"<p>{str(exc)}</p>",
            status_code=503,
        )


@app.get("/api/forex-lab/status")
async def forex_lab_status(authorization: str | None = Header(default=None)):
    require_token(authorization)
    return forex_lab.public_state()


@app.get("/api/forex-lab/results")
async def forex_lab_results(authorization: str | None = Header(default=None)):
    require_token(authorization)
    return {"results": forex_lab.results(), "state": forex_lab.public_state()}


@app.post("/api/forex-lab/start")
async def forex_lab_start(authorization: str | None = Header(default=None)):
    require_token(authorization)
    if precision_lab.state.running:
        await precision_lab.stop()
    await forex_lab.start()
    if not forex_lab.state.running and forex_lab.state.stage == "waiting_credentials":
        raise HTTPException(status_code=503, detail=forex_lab.state.message)
    return forex_lab.public_state()


@app.post("/api/forex-lab/stop")
async def forex_lab_stop(authorization: str | None = Header(default=None)):
    require_token(authorization)
    await forex_lab.stop()
    return forex_lab.public_state()


@app.get("/api/precision-lab/status")
async def precision_lab_status(authorization: str | None = Header(default=None)):
    require_token(authorization)
    return precision_lab.public_state()


@app.get("/api/precision-lab/results")
async def precision_lab_results(authorization: str | None = Header(default=None)):
    require_token(authorization)
    return {
        "results": precision_lab.results(),
        "state": precision_lab.public_state(),
    }


@app.post("/api/precision-lab/start")
async def precision_lab_start(authorization: str | None = Header(default=None)):
    require_token(authorization)
    if forex_lab.state.running:
        await forex_lab.stop()
    await precision_lab.start()
    if (
        not precision_lab.state.running
        and precision_lab.state.stage == "waiting_credentials"
    ):
        raise HTTPException(status_code=503, detail=precision_lab.state.message)
    return precision_lab.public_state()


@app.post("/api/precision-lab/stop")
async def precision_lab_stop(authorization: str | None = Header(default=None)):
    require_token(authorization)
    await precision_lab.stop()
    return precision_lab.public_state()


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
body{font-family:system-ui,-apple-system,sans-serif;background:#0e1116;color:#e9eef5;margin:0;padding:28px;max-width:1100px;margin:auto}
.card{background:#171c24;border:1px solid #2a3442;border-radius:16px;padding:18px;margin:14px 0}.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}button,input{font:inherit;border-radius:10px;border:1px solid #3a4658;padding:10px 13px;background:#0f141b;color:#fff}button{cursor:pointer}button.danger{border-color:#8a3841}.muted{color:#96a4b5}pre{white-space:pre-wrap;word-break:break-word}h1{margin-bottom:0}.pill{padding:5px 9px;border-radius:999px;background:#242d39}.pill.good{background:#17351f}.pill.bad{background:#3a1c22}.pill.run{background:#17314a}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}.metric{background:#111720;padding:12px;border-radius:12px}.metric b{display:block;font-size:1.3rem;margin-top:4px}.progress{height:12px;background:#0f141b;border:1px solid #2a3442;border-radius:999px;overflow:hidden;margin:12px 0}.progress>div{height:100%;background:#e9eef5;width:0%;transition:width .25s ease}.lab-head{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap}.lab-table{width:100%;border-collapse:collapse;margin-top:14px;font-size:.92rem}.lab-table th,.lab-table td{text-align:left;padding:9px 8px;border-bottom:1px solid #2a3442;vertical-align:top}.lab-table th{color:#96a4b5;font-weight:600}.ok{color:#8de39e}.no{color:#ff9da8}.small{font-size:.85rem}.scroll{overflow-x:auto}.overview{position:sticky;top:8px;z-index:20;background:#171c24ee;backdrop-filter:blur(8px)}.lab-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-top:12px}.lab-mini{background:#111720;border:1px solid #2a3442;border-radius:12px;padding:12px}.lab-mini b{display:block;margin-bottom:6px}.lab-mini .state{font-size:.8rem;font-weight:700}.state.done{color:#8de39e}.state.running{color:#8ecbff}.state.waiting{color:#96a4b5}.state.error{color:#ff9da8}details{margin-top:12px}summary{cursor:pointer;color:#96a4b5}.nav{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}.nav a{color:#dbe7f5;text-decoration:none;background:#111720;border:1px solid #2a3442;padding:9px 12px;border-radius:10px}.section-title{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap}.mode-banner{padding:10px 12px;border-radius:12px;background:#132318;border:1px solid #275335;color:#a7e6b3;margin:10px 0}.signal-on{color:#8de39e;font-weight:700}.signal-off{color:#738195}.crypto-card{background:#111720;border:1px solid #2a3442;border-radius:12px;padding:14px}.crypto-market{font-size:1.05rem;font-weight:700;margin-bottom:8px}</style>
</head>
<body>
<h1>MicroTrader</h1><p class="muted">Trading + research dashboard · safe-by-default</p>
<div class="nav"><a href="#engine">Trading Engine</a><a href="#forex-lab">Forex Lab</a><a href="#precision-lab">Precision Lab</a><a href="#agent">Research Agent</a><a href="#stock-lab">Stock Lab</a><a href="#crypto-lab">Crypto Lab</a><a href="#research-labs">Research Labs</a></div>
<div class="card">
  <div class="row"><input id="token" type="password" placeholder="Dashboard token" style="min-width:260px"><button onclick="loadStatus()">Connect</button><button onclick="action('start')">Start</button><button onclick="action('stop')">Stop</button><button class="danger" onclick="action('flatten')">Flatten</button></div>
</div>
<div class="card overview" id="engine">
  <h3 style="margin-top:0">Live Overview</h3>
  <div id="overallStatus" class="row"><span class="muted">Connecting…</span></div>
  <div id="overallMetrics" class="grid" style="margin-top:12px"></div>
</div>
<div class="card"><div id="headline" class="row"></div><div id="metrics" class="grid" style="margin-top:12px"></div></div>
<div class="card"><h3>Positions</h3><pre id="positions">Not connected.</pre></div>
<div class="card"><h3>Recent executions</h3><pre id="executions">Not connected.</pre></div>
<div class="card" id="forex-lab">
  <div class="section-title"><h3>Forex Strategy Lab</h3><span class="pill good">FUSION / cTRADER · DEMO FIRST</span></div>
  <div class="mode-banner"><b>Asymmetry-first.</b> Elke strategie-run wordt append-only opgeslagen met parameters, dataset, OOS/stress-resultaten, per-pair resultaten, risk model en uitgebreide trade-statistieken.</div>
  <div class="row"><button onclick="forexAction('start')">Run Forex Lab</button><button onclick="forexAction('stop')">Stop Forex Lab</button><button onclick="loadForex()">Refresh</button><button onclick="openCTraderAuth()">Authorize cTrader</button></div>
  <div id="forexStatus" class="row" style="margin-top:14px"><span class="pill">IDLE</span></div>
  <div id="forexMessage" class="muted" style="margin-top:8px">Waiting for configuration.</div>
  <div id="forexMetrics" class="grid" style="margin-top:12px"></div>
  <div id="forexResults" class="scroll"></div>
</div>
<div class="card" id="precision-lab">
  <div class="section-title"><h3>Precision Lab</h3><span class="pill good">TIGHT STOP · TICK EXECUTION</span></div>
  <div class="mode-banner"><b>Aparte tester.</b> 3–4 pip stops, 5–10 pip targets, pending limit entries en echte bid/ask-tickuitvoering. Draait niet tegelijk met het normale Forex Lab om CPU en cTrader-requests te sparen.</div>
  <div class="row"><button onclick="precisionAction('start')">Run Precision Lab</button><button onclick="precisionAction('stop')">Stop Precision Lab</button><button onclick="loadPrecision()">Refresh</button></div>
  <div id="precisionStatus" class="row" style="margin-top:14px"><span class="pill">IDLE</span></div>
  <div id="precisionMessage" class="muted" style="margin-top:8px">Waiting for cTrader credentials.</div>
  <div id="precisionMetrics" class="grid" style="margin-top:12px"></div>
  <div id="precisionResults" class="scroll"></div>
</div>
<div class="card" id="agent">
  <div class="section-title"><h3>Research Agent</h3><span class="pill good">RESEARCH ONLY</span></div>
  <div class="mode-banner"><b>Autonoom, begrensd.</b> De agent mag research prioriteren en vervolgexperimenten sturen, maar kan geen orders plaatsen of live trading inschakelen.</div>
  <div class="row"><button onclick="agentAction('start')">Start Agent</button><button onclick="agentAction('stop')">Stop Agent</button><button onclick="agentAction('cycle')">Run Cycle Now</button><button onclick="loadAgent()">Refresh</button></div>
  <div id="agentStatus" class="row" style="margin-top:14px"><span class="pill">IDLE</span></div>
  <div id="agentMessage" class="muted" style="margin-top:8px">Nog niet geladen.</div>
  <div id="agentMetrics" class="grid" style="margin-top:12px"></div>
  <h4 style="margin-bottom:6px">Current Research Direction</h4>
  <div id="agentFocus" class="row"></div>
  <h4 style="margin-bottom:6px">Hypotheses</h4>
  <div id="agentHypotheses" class="lab-grid"></div>
  <details><summary>Laatste agent-besluit / technische details</summary><pre id="agentRaw" class="small">Not loaded.</pre></details>
</div>
<div class="card" id="stock-lab">
  <div class="section-title"><h3>Stock Strategy Lab</h3><span class="pill">ALPACA · RESEARCH</span></div>
  <p class="muted">Funnel: <b>Discovery</b> zoekt breed → <b>Incubator</b> maakt varianten rond kansrijke kandidaten → <b>Deep Search</b> valideert op volledige historie → <b>Promoted</b> haalt alle harde filters.</p>
  <div class="row"><button onclick="labAction('start')">Run Strategy Lab</button><button onclick="labAction('stop')">Stop Lab</button><button onclick="loadLab(true)">Refresh Lab</button></div>
  <div class="lab-head" style="margin-top:14px">
    <div id="labStatus"><span class="pill">IDLE</span></div>
    <div id="labUpdated" class="muted small"></div>
  </div>
  <div class="progress"><div id="labProgress"></div></div>
  <div id="labMessage" class="muted">Not run yet.</div>
  <div id="labWork" class="metric" style="margin-top:12px"><span class="muted">Current work</span><b>Waiting…</b></div>
  <div id="labMetrics" class="grid" style="margin-top:12px"></div>
  <h4 style="margin-bottom:6px">Promising pipeline</h4>
  <div id="labPromising" class="lab-grid"></div>
  <div id="labResults" class="scroll"></div>
</div>
<div class="card" id="crypto-lab">
  <div class="section-title"><h3>Crypto / Microstructure Lab</h3><span class="pill good">SIMULATION ONLY</span></div>
  <div class="mode-banner"><b>Geen echt geld.</b> Dit lab leest alleen publieke Bitvavo marktdata en plaatst geen orders.</div>
  <p class="muted">Vergelijkt mean reversion, spread/micro-market-making, order-book imbalance en een hybride signaal op live top-of-book data.</p>
  <div class="row"><button onclick="cryptoAction('start')">Start Crypto Lab</button><button onclick="cryptoAction('stop')">Stop Crypto Lab</button><button onclick="loadCrypto()">Refresh</button></div>
  <div id="cryptoStatus" class="row" style="margin-top:14px"><span class="pill">IDLE</span></div>
  <div id="cryptoMessage" class="muted" style="margin-top:8px">Nog niet gestart.</div>
  <div id="cryptoMetrics" class="grid" style="margin-top:12px"></div>
  <h4 style="margin-bottom:6px">Strategy Scoreboard</h4>
  <div id="cryptoScoreboard" class="scroll"></div>
  <details><summary>Live Market Signals tonen</summary><div id="cryptoMarkets" class="lab-grid"></div></details>
  <details><summary>Strategiestatistieken / technische details</summary><pre id="cryptoRaw" class="small">Not loaded.</pre></details>
</div>
<div class="card" id="research-labs">
  <h3>Research Labs</h3>
  <p class="muted">17 specialised labs for robustness, sizing, compounding, risk and aggressive-growth research.</p>
  <div class="row"><button onclick="researchAction('start')">Run Research Labs</button><button onclick="researchAction('stop')">Stop Research Labs</button><button onclick="loadResearch()">Refresh</button></div>
  <div id="researchStatus" class="row" style="margin-top:14px"></div>
  <div id="researchMetrics" class="grid" style="margin-top:12px"></div>
  <div id="researchLabGrid" class="lab-grid"></div>
  <details><summary>Technische details / ruwe resultaten</summary><pre id="researchResults" class="small">Not loaded.</pre></details>
</div>
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
 }catch(e){
  const msg=e.message==='Unauthorized'
    ? 'Dashboard token ontbreekt of is ongeldig.'
    : e.message;
  document.getElementById('raw').textContent=msg;
 }
}
async function action(x){
 try{await api(x,'POST');await loadStatus()}
 catch(e){
  const msg=e.message==='Unauthorized'
    ? 'Dashboard token ontbreekt of is ongeldig.'
    : e.message;
  document.getElementById('raw').textContent=msg;
 }
}
function fmt(v,d=2){return (v===null||v===undefined)?'-':Number(v).toFixed(d)}
function esc(s){return String(s??'').replace(/[&<>"']/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]))}
function renderLab(s,results){
 const done=Number(s.progress||0), total=Number(s.total||0);
 const symbolDone=Number(s.symbols_loaded||0), symbolTotal=Number(s.symbols_total||0);
 let pct=0;
 if(s.stage==='loading_data' && symbolTotal) pct=Math.round((symbolDone/symbolTotal)*35);
 else if(total) pct=Math.round(35+(done/total)*65);
 if(s.stage==='completed') pct=100;
 document.getElementById('labProgress').style.width=Math.max(0,Math.min(100,pct))+'%';
 const cls=s.running?'run':(s.last_error?'bad':(s.completed_at?'good':''));
 const label=s.running?'RUNNING':(s.last_error?'ERROR':(s.completed_at?'COMPLETED':'IDLE'));
 document.getElementById('labStatus').innerHTML=`<span class="pill ${cls}">${label}</span> <span class="muted small">${esc(s.stage||'idle')}</span>`;
 document.getElementById('labMessage').textContent=s.message||'';
 document.getElementById('labUpdated').textContent=s.completed_at?('Completed '+new Date(s.completed_at).toLocaleString()):(s.started_at?('Started '+new Date(s.started_at).toLocaleString()):'');
 const sum=s.summary||{};
 const fc=sum.funnel_counts||{};
 const cand=s.current_candidate||'';
 const cp=s.current_params||{};
 const symTotal=Number(s.current_symbol_total||0), symIdx=Number(s.current_symbol_index||0);
 const symbolPct=symTotal?Math.round((symIdx/symTotal)*100):0;
 const currentBits=[];
 if(cand) currentBits.push(prettyLabName(cand));
 if(cp.timeframe_min) currentBits.push(cp.timeframe_min+'m');
 if(s.current_symbol) currentBits.push('symbol '+s.current_symbol+' '+symIdx+'/'+symTotal);
 if(s.candidate_seconds) currentBits.push(Math.round(Number(s.candidate_seconds))+'s');
 document.getElementById('labWork').innerHTML=
   '<span class="muted">Current work</span><b>'+esc(currentBits.join(' · ')||prettyLabName(s.stage||'idle'))+'</b>'+
   (symTotal?'<div class="progress" style="margin:8px 0 0"><div style="width:'+symbolPct+'%"></div></div>':'')+
   '<div class="small muted">'+esc(s.last_completed_candidate?('Last completed: '+prettyLabName(s.last_completed_candidate)+(s.candidate_seconds?' · '+fmt(s.candidate_seconds,1)+'s':'')):'No candidate completed in this run yet.')+'</div>'+
   (s.last_persist_error?'<div class="small no">Persistence delayed: '+esc(s.last_persist_error)+'</div>':'');
 const vals={
   'Source data':sum.source_timeframe||'1Min',
   'Current stage':prettyLabName(s.stage||'idle'),
   'Tested':sum.candidates_tested??s.tested_total??results.length,
   'Incubator':fc.incubator??0,
   'Deep Search':fc.deep_search??0,
   'Promoted':(fc.promoted??sum.promoted_count??s.promoted_total??0)+'/'+(sum.target_promoted??s.target_promoted??'-'),
   'Persistence':prettyLabName(s.persistence_status||'idle')+(s.persistence_pending?(' · '+s.persistence_pending+' queued'):'')
 };
 document.getElementById('labMetrics').innerHTML=Object.entries(vals).map(([k,v])=>`<div class="metric"><span class="muted">${k}</span><b>${v}</b></div>`).join('');
 const promising=results.filter(x=>['incubator','deep_search','promoted'].includes(x.funnel_stage)).slice(0,6);
 document.getElementById('labPromising').innerHTML=promising.length?promising.map(x=>{
   const o=x.oos||{}, stage=x.promoted?'PROMOTED':(x.funnel_stage==='deep_search'?'DEEP SEARCH':'INCUBATOR');
   const cls=x.promoted?'done':(x.funnel_stage==='deep_search'?'running':'waiting');
   return '<div class="lab-mini"><b>'+esc(prettyLabName(x.strategy))+' · '+esc((x.params||{}).timeframe_min||'-')+'m</b>'+
     '<span class="state '+cls+'">'+stage+'</span>'+
     '<div class="small muted" style="margin-top:7px">Score '+fmt(x.funnel_score,1)+' · PF '+fmt(o.profit_factor,2)+' · Exp '+fmt(o.expectancy_bps,2)+' bps · '+(o.trades??0)+' trades</div></div>';
 }).join(''):'<div class="lab-mini"><b>Nog geen promising kandidaat</b><span class="state waiting">DISCOVERY</span><div class="small muted" style="margin-top:7px">Het lab zoekt breed totdat een kandidaat Incubator of Deep Search haalt.</div></div>';
 if(!results.length){
   document.getElementById('labResults').innerHTML='<p class="muted">No results yet. During data loading this is normal.</p>';
   return;
 }
 const promoted=results.filter(x=>x.promoted);
 const deep=results.filter(x=>x.funnel_stage==='deep_search' && !x.promoted);
 const incubator=results.filter(x=>x.funnel_stage==='incubator');
 const rejected=results.filter(x=>!x.promoted && !['deep_search','incubator'].includes(x.funnel_stage));
 function labRows(items){
   return items.map((x,i)=>{
     const o=x.oos||{}, st=x.stress_oos||{};
     const status=x.promoted?'<span class="ok">PROMOTED</span>':(x.funnel_stage==='deep_search'?'<span class="ok">DEEP SEARCH</span>':(x.funnel_stage==='incubator'?'<span class="pill run">INCUBATOR</span>':'<span class="no">REJECTED</span>'));
     const why=x.promoted?'—':esc((x.rejection_reasons||[]).join('; '));
     const tf=(x.params||{}).timeframe_min||'-';
     return `<tr><td>${i+1}</td><td>${esc(x.strategy)}</td><td>${tf}m</td><td>${status}</td><td>${fmt(x.funnel_score,1)}</td><td>${o.trades??0}</td><td>${fmt(o.win_rate_pct,1)}%</td><td>${fmt(o.payoff_ratio,2)}x</td><td>${fmt(o.expectancy_r,2)}R</td><td>${fmt(o.expectancy_bps,3)} bps</td><td>${fmt(o.profit_factor,2)}</td><td>${fmt(st.expectancy_bps,3)} bps</td><td class="small">${why}</td></tr>`;
   }).join('');
 }
 const head='<table class="lab-table"><thead><tr><th>#</th><th>Family</th><th>TF</th><th>Stage</th><th>Score</th><th>OOS trades</th><th>Winrate</th><th>Payoff</th><th>Exp. R</th><th>OOS exp.</th><th>PF</th><th>Stress exp.</th><th>Why not promoted?</th></tr></thead><tbody>';
 let html='<div class="row small" style="margin:10px 0"><span class="pill good">'+promoted.length+' promoted</span><span class="pill run">'+deep.length+' deep search</span><span class="pill">'+incubator.length+' incubator</span><span class="pill">'+rejected.length+' rejected</span></div>';
 const near=(sum.near_misses||[]).slice(0,5);
 if(near.length){
   html+='<h4 style="margin-bottom:6px">Most promising / near misses</h4>'+head+labRows(near)+'</tbody></table>';
 }
 if(promoted.length){
   html+='<h4 style="margin-bottom:6px">Promoted</h4>'+head+labRows(promoted)+'</tbody></table>';
 }
 if(deep.length){
   html+='<h4 style="margin-bottom:6px">Deep Search</h4>'+head+labRows(deep.slice(0,10))+'</tbody></table>';
 }
 if(incubator.length){
   html+='<details open><summary>Incubator ('+incubator.length+')</summary>'+head+labRows(incubator.slice(0,15))+'</tbody></table></details>';
 }
 if(rejected.length){
   html+='<details><summary>Rejected strategieën tonen ('+rejected.length+')</summary>'+head+labRows(rejected)+'</tbody></table></details>';
 }
 document.getElementById('labResults').innerHTML=html;
}
async function loadLab(manual=false){
 try{
  const [s,r]=await Promise.all([api('lab/status'),api('lab/results')]);
  renderLab(s,r.results||[]);
  lastStrategyState=s;
  updateOverview(lastStrategyState,lastResearchState,lastCryptoState,lastCoordinatorState);
 }catch(e){
  document.getElementById('labMessage').textContent='Lab error: '+e.message;
 }
}
async function labAction(x){
 try{await api('lab/'+x,'POST');await loadLab(true)}
 catch(e){
  const msg=e.message==='Unauthorized'
    ? 'Dashboard token ontbreekt of is ongeldig.'
    : e.message;
  document.getElementById('labMessage').textContent=msg;
 }
}
async function loadForex(){
 try{
  const [s,r]=await Promise.all([api('forex-lab/status'),api('forex-lab/results')]);
  const broker=s.broker||{};
  const cls=s.running?'run':(s.last_error?'bad':(broker.account_authenticated?'good':''));
  const label=s.running?'RUNNING':(s.last_error?'ERROR':(broker.account_authenticated?'READY':'WAITING'));
  document.getElementById('forexStatus').innerHTML=
    '<span class="pill '+cls+'">'+label+'</span>'+
    '<span class="pill">'+esc(s.stage||'idle')+'</span>'+
    '<span class="pill">'+esc(broker.environment||'demo').toUpperCase()+'</span>';
  document.getElementById('forexMessage').textContent=s.message||'';
  const vals={
    'Pairs':(s.pairs||[]).join(', ')||'-',
    'Tested':s.tested_total||0,
    'Start capital':'€'+fmt(s.start_capital_eur,2),
    'Risk / trade':'€'+fmt(s.risk_eur,2),
    'cTrader':broker.account_authenticated?'AUTHENTICATED':(broker.access_token_configured?'TOKEN READY':'NEEDS CREDENTIALS'),
    'Progress':(s.progress||0)+'/'+(s.total||0)
  };
  document.getElementById('forexMetrics').innerHTML=Object.entries(vals).map(([k,v])=>'<div class="metric"><span class="muted">'+esc(k)+'</span><b>'+esc(v)+'</b></div>').join('');
  const rows=(r.results||[]).slice(0,30);
  if(!rows.length){
    document.getElementById('forexResults').innerHTML='<p class="muted">Nog geen forex strategie-resultaten.</p>';
    return;
  }
  const head='<table class="lab-table"><thead><tr><th>#</th><th>Strategy</th><th>TF</th><th>Target</th><th>Stage</th><th>Trades</th><th>/day</th><th>Win%</th><th>Avg win</th><th>Avg loss</th><th>Payoff</th><th>PF</th><th>Exp R</th><th>Best/Worst</th><th>Loss streak</th></tr></thead><tbody>';
  const body=rows.map((x,i)=>{
    const m=x.oos||{},p=x.params||{};
    return '<tr><td>'+(i+1)+'</td><td>'+esc(prettyLabName(x.strategy))+'</td><td>'+esc(x.timeframe_min||p.timeframe_min||'-')+'m</td><td>'+fmt(p.target_r,1)+'R</td><td>'+esc(prettyLabName(x.status||x.funnel_stage||''))+'</td><td>'+(m.trades||0)+'</td><td>'+fmt(m.avg_trades_per_day,2)+'</td><td>'+fmt(m.win_rate_pct,1)+'%</td><td>'+fmt(m.avg_win_r,2)+'R</td><td>'+fmt(m.avg_loss_r,2)+'R</td><td>'+fmt(m.payoff_ratio,2)+'x</td><td>'+fmt(m.profit_factor,2)+'</td><td>'+fmt(m.expectancy_r,3)+'R</td><td>'+fmt(m.best_trade_r,2)+' / '+fmt(m.worst_trade_r,2)+'R</td><td>'+(m.max_consecutive_losses||0)+'</td></tr>';
  }).join('');
  document.getElementById('forexResults').innerHTML=head+body+'</tbody></table>';
 }catch(e){
  document.getElementById('forexMessage').textContent=e.message==='Unauthorized'?'Dashboard token ontbreekt of is ongeldig.':e.message;
 }
}
async function forexAction(x){
 try{await api('forex-lab/'+x,'POST');await loadForex()}
 catch(e){document.getElementById('forexMessage').textContent=e.message}
}
async function openCTraderAuth(){
 try{
  const x=await api('ctrader/oauth-url');
  window.open(x.url,'_blank','noopener');
 }catch(e){document.getElementById('forexMessage').textContent=e.message}
}

async function loadPrecision(){
 try{
  const [s,r]=await Promise.all([api('precision-lab/status'),api('precision-lab/results')]);
  const broker=s.broker||{};
  const cls=s.running?'run':(s.last_error?'bad':(broker.account_authenticated?'good':''));
  const label=s.running?'RUNNING':(s.last_error?'ERROR':(broker.account_authenticated?'READY':'WAITING'));
  document.getElementById('precisionStatus').innerHTML=
    '<span class="pill '+cls+'">'+label+'</span>'+
    '<span class="pill">'+esc(s.stage||'idle')+'</span>'+
    '<span class="pill">3–4p → 5–10p</span>';
  document.getElementById('precisionMessage').textContent=s.message||'';
  const vals={
    'Pairs':(s.pairs||[]).join(', ')||'-',
    'Tested':s.tested_total||0,
    'Deep search':s.deep_search_total||0,
    'Bars loaded':s.bars_loaded||0,
    'Quote ticks':s.quote_ticks_loaded||0,
    'Risk / trade':'€'+fmt(s.risk_eur,2),
    'Commission model':fmt(s.commission_pips_roundtrip,2)+' pip RT',
    'Progress':(s.progress||0)+'/'+(s.total||0)
  };
  document.getElementById('precisionMetrics').innerHTML=Object.entries(vals).map(([k,v])=>'<div class="metric"><span class="muted">'+esc(k)+'</span><b>'+esc(v)+'</b></div>').join('');
  const rows=(r.results||[]).slice(0,40);
  if(!rows.length){
    document.getElementById('precisionResults').innerHTML='<p class="muted">Nog geen Precision Lab-resultaten.</p>';
    return;
  }
  const head='<table class="lab-table"><thead><tr><th>#</th><th>Setup</th><th>Stop</th><th>Target</th><th>Stage</th><th>Trades</th><th>/day</th><th>Win%</th><th>Fill%</th><th>Avg win</th><th>Avg loss</th><th>Payoff</th><th>PF</th><th>Exp R</th><th>DD</th><th>Loss streak</th></tr></thead><tbody>';
  const body=rows.map((x,i)=>{
    const m=x.oos||{},p=x.params||{};
    return '<tr><td>'+(i+1)+'</td><td>'+esc(prettyLabName(x.strategy))+'</td><td>'+fmt(p.stop_pips,1)+'p</td><td>'+fmt(p.target_pips,1)+'p</td><td>'+esc(prettyLabName(x.status||x.funnel_stage||''))+'</td><td>'+(m.trades||0)+'</td><td>'+fmt(m.avg_trades_per_day,2)+'</td><td>'+fmt(m.win_rate_pct,1)+'%</td><td>'+fmt(x.avg_fill_rate_pct,1)+'%</td><td>'+fmt(m.avg_win_r,2)+'R</td><td>'+fmt(m.avg_loss_r,2)+'R</td><td>'+fmt(m.payoff_ratio,2)+'x</td><td>'+fmt(m.profit_factor,2)+'</td><td>'+fmt(m.expectancy_r,3)+'R</td><td>'+fmt(m.max_drawdown_pct,1)+'%</td><td>'+(m.max_consecutive_losses||0)+'</td></tr>';
  }).join('');
  document.getElementById('precisionResults').innerHTML=head+body+'</tbody></table>';
 }catch(e){
  document.getElementById('precisionMessage').textContent=e.message==='Unauthorized'?'Dashboard token ontbreekt of is ongeldig.':e.message;
 }
}
async function precisionAction(x){
 try{await api('precision-lab/'+x,'POST');await loadPrecision();await loadForex()}
 catch(e){document.getElementById('precisionMessage').textContent=e.message}
}

async function loadAgent(){
 try{
  const s=await api('agent/status');
  const cls=s.running?'run':(s.last_error?'bad':'good');
  const label=s.running?'RUNNING':(s.last_error?'ERROR':'STOPPED');
  document.getElementById('agentStatus').innerHTML=
    '<span class="pill '+cls+'">'+label+'</span>'+
    '<span class="pill good">NO ORDER PERMISSION</span>'+
    '<span class="muted small">'+esc(s.stage||'idle')+'</span>';
  document.getElementById('agentMessage').textContent=s.message||'';
  const vals={
    'Cycles':s.cycles||0,
    'Last cycle':s.last_cycle_at?new Date(s.last_cycle_at).toLocaleTimeString():'-',
    'Focus families':(s.focus_families||[]).length,
    'Focus timeframes':(s.focus_timeframes||[]).map(x=>x+'m').join(', ')||'-'
  };
  document.getElementById('agentMetrics').innerHTML=Object.entries(vals).map(([k,v])=>'<div class="metric"><span class="muted">'+esc(k)+'</span><b>'+esc(v)+'</b></div>').join('');
  const fam=s.focus_families||[], tfs=s.focus_timeframes||[];
  document.getElementById('agentFocus').innerHTML=
    fam.map(x=>'<span class="pill run">'+esc(prettyLabName(x))+'</span>').join('')+
    tfs.map(x=>'<span class="pill">'+esc(x)+'m</span>').join('');
  const hy=s.hypotheses||[];
  document.getElementById('agentHypotheses').innerHTML=hy.length?hy.map(h=>
    '<div class="lab-mini"><b>'+esc(prettyLabName(h.family||''))+' · '+esc(h.timeframe_min||'-')+'m</b>'+
    '<div class="small muted">Score '+fmt(h.funnel_score,1)+' · Exp '+fmt(h.expectancy_bps,3)+' bps · PF '+fmt(h.profit_factor,2)+' · Payoff '+fmt(h.payoff_ratio,2)+'x</div>'+
    '<div class="small" style="margin-top:7px">'+esc(h.thesis||'')+'</div></div>'
  ).join(''):'<p class="muted">Nog geen hypotheses. De agent wacht op nieuwe funnel-resultaten.</p>';
  document.getElementById('agentRaw').textContent=JSON.stringify(s,null,2);
 }catch(e){
  document.getElementById('agentMessage').textContent=e.message==='Unauthorized'?'Dashboard token ontbreekt of is ongeldig.':e.message;
 }
}
async function agentAction(x){
 try{await api('agent/'+x,'POST');await loadAgent();await loadLab(true)}
 catch(e){document.getElementById('agentMessage').textContent=e.message}
}

async function loadCrypto(){
 try{
  const s=await api('crypto-lab/status');
  lastCryptoState=s;
  const cls=s.running?'run':(s.last_error?'bad':((s.observations||0)>0?'good':''));
  const label=s.running?'COLLECTING':(s.last_error?'ERROR':((s.observations||0)>0?'PAUSED':'IDLE'));
  document.getElementById('cryptoStatus').innerHTML=
    '<span class="pill '+cls+'">'+label+'</span>'+
    '<span class="pill good">NO LIVE ORDERS</span>'+
    '<span class="muted small">'+esc(s.venue||'')+'</span>';
  document.getElementById('cryptoMessage').textContent=s.message||'';
  const vals={
    'Markets':(s.markets||[]).length+'/'+(s.symbols||[]).length,
    'Observations':s.observations||0,
    'Book events':s.book_events||0,
    'Trade events':s.trade_events||0,
    'Reconnects':s.reconnects||0,
    'Book depth':s.book_depth||'-',
    'Flow window':(s.flow_window_seconds||'-')+' sec',
    'Last update':s.last_update?new Date(s.last_update).toLocaleTimeString():'-'
  };
  document.getElementById('cryptoMetrics').innerHTML=Object.entries(vals).map(function(kv){
    return '<div class="metric"><span class="muted">'+esc(kv[0])+'</span><b>'+esc(kv[1])+'</b></div>';
  }).join('');
  const paper=(s.paper_simulation||{});
  const strategies=(paper.strategies||{});
  const strategyOrder=['mean_reversion','market_maker','top_imbalance','full_book_imbalance','order_flow','liquidity_vacuum','flow_reversal','momentum','hybrid','adaptive'];
  const ranked=strategyOrder.map(function(name){
    return {name:name,x:strategies[name]||{}};
  }).sort(function(a,b){return Number(b.x.expectancy_bps||0)-Number(a.x.expectancy_bps||0)});
  const rows=ranked.map(function(row,idx){
    const name=row.name,x=row.x;
    const pnl=Number(x.net_pnl_quote||x.net_pnl_eur||0);
    const pnlClass=pnl>0?'ok':(pnl<0?'no':'muted');
    return '<tr>'+
      '<td>'+(idx+1)+'</td>'+
      '<td>'+esc(prettyLabName(name))+'</td>'+
      '<td>'+(x.trades||0)+'</td>'+
      '<td>'+fmt(x.win_rate_pct,1)+'%</td>'+
      '<td class="'+pnlClass+'">'+fmt(pnl,4)+'</td>'+
      '<td>'+fmt(x.expectancy_bps,3)+' bps</td>'+
      '<td>'+fmt(x.max_drawdown_quote||x.max_drawdown_eur,4)+'</td>'+
      '<td>'+((x.open_positions||0)+(x.pending_entries||0))+'</td>'+
    '</tr>';
  }).join('');
  const qs=paper.quote_summary||{};
  document.getElementById('cryptoScoreboard').innerHTML=
    '<div class="small muted" style="margin:6px 0 10px">Realtime WebSocket · '+esc(s.stream||'')+
    ' · '+fmt(paper.notional_eur_per_trade,2)+' quote units per paper trade · maker fees per markt inbegrepen</div>'+
    '<div class="row small" style="margin-bottom:8px"><span class="pill">EUR: '+(qs.EUR?.trades||0)+' trades · '+fmt(qs.EUR?.expectancy_bps,3)+' bps</span><span class="pill">USDC: '+(qs.USDC?.trades||0)+' trades · '+fmt(qs.USDC?.expectancy_bps,3)+' bps</span></div>'+
    '<table class="lab-table"><thead><tr><th>#</th><th>Strategy</th><th>Trades</th><th>Winrate</th><th>Net P&L*</th><th>Expectancy</th><th>Max DD*</th><th>Open/Pending</th></tr></thead><tbody>'+rows+'</tbody></table>'+
    '<div class="small muted">* P&L staat in de quote currency van de markt; EUR en USDC worden niet als exact dezelfde valuta opgeteld voor financiële conclusies.</div>';
  const markets=s.markets||[];
  document.getElementById('cryptoMarkets').innerHTML=markets.length?markets.map(function(m){
    function sig(on,label){return '<span class="'+(on?'signal-on':'signal-off')+'">'+(on?'●':'○')+' '+label+'</span>';}
    return '<div class="crypto-card">'+
      '<div class="crypto-market">'+esc(m.market)+'</div>'+
      '<div class="small muted">Mid €'+fmt(m.mid,2)+' · Spread '+fmt(m.spread_bps,2)+' bps</div>'+
      '<div class="small muted">Z '+fmt(m.zscore,2)+' · Top '+fmt(m.top_imbalance,2)+' · Full book '+fmt(m.full_book_imbalance,2)+' · Flow '+fmt(m.flow_imbalance,2)+'</div>'+
      '<div class="small" style="margin-top:8px">'+Object.entries(m.signals||{}).filter(function(kv){return kv[1]}).map(function(kv){return sig(true,prettyLabName(kv[0]));}).join('<br>')+'</div>'+
    '</div>';
  }).join(''):'<p class="muted">Start het lab om live marktdata te verzamelen.</p>';
  document.getElementById('cryptoRaw').textContent=JSON.stringify({paper_simulation:s.paper_simulation,strategy_stats:s.strategy_stats,markets:s.markets},null,2);
  updateOverview(lastStrategyState,lastResearchState,lastCryptoState,lastCoordinatorState);
 }catch(e){
  document.getElementById('cryptoMessage').textContent=e.message==='Unauthorized'?'Dashboard token ontbreekt of is ongeldig.':e.message;
 }
}
async function cryptoAction(x){
 try{await api('crypto-lab/'+x,'POST');await loadCrypto()}
 catch(e){document.getElementById('cryptoMessage').textContent=e.message}
}
const researchNames=['market','session','regime','high_frequency','walk_forward','parameter_stability','cost_stress','monte_carlo','position_sizing','compounding','leverage','risk_of_ruin','recovery','portfolio','capital_allocation','aggressive_growth','master'];
function prettyLabName(x){return String(x||'').split('_').map(w=>w.charAt(0).toUpperCase()+w.slice(1)).join(' ')}
function updateOverview(strategyState,researchState,cryptoState,coordinatorState){
 const strategyRunning=!!strategyState?.running, researchRunning=!!researchState?.running, cryptoRunning=!!cryptoState?.running;
 document.getElementById('overallStatus').innerHTML=
   `<span class="pill ${strategyRunning?'run':'good'}">Strategy Lab: ${strategyRunning?'RUNNING':'STOPPED'}</span>`+
   `<span class="pill ${researchRunning?'run':'good'}">Research Labs: ${researchRunning?'RUNNING':'STOPPED'}</span>`+
   `<span class="pill ${cryptoRunning?'run':'good'}">Crypto Lab: ${cryptoRunning?'COLLECTING':'STOPPED'}</span>`+
   `<span class="pill run">Scheduler: ${esc(prettyLabName(coordinatorState?.mode||'idle'))}</span>`+
   (researchState?.current_lab?`<span class="muted">Current: ${esc(prettyLabName(researchState.current_lab))}</span>`:'');
 const vals={
   'Strategies tested':strategyState?.tested_total??'-',
   'Strategies promoted':(strategyState?.promoted_total??0)+'/'+(strategyState?.target_promoted??'-'),
   'Research progress':(researchState?.completed_labs??0)+'/'+(researchState?.total_labs??17),
   'Active research':researchState?.current_lab?prettyLabName(researchState.current_lab):'-',
   'Crypto observations':cryptoState?.observations??'-',
   'Scheduler':coordinatorState?.message||'-'
 };
 document.getElementById('overallMetrics').innerHTML=Object.entries(vals).map(([k,v])=>`<div class="metric"><span class="muted">${k}</span><b>${esc(v)}</b></div>`).join('');
}
let lastStrategyState=null,lastResearchState=null,lastCryptoState=null,lastCoordinatorState=null;
async function loadResearch(){
 try{
  const s=await api('research/status');
  lastResearchState=s;
  document.getElementById('researchStatus').innerHTML=`<span class="pill ${s.running?'run':'good'}">${s.running?'RUNNING':'IDLE'}</span><span class="muted small">${esc(prettyLabName(s.current_lab||''))}</span>`;
  const vals={'Completed labs':(s.completed_labs||0)+'/'+(s.total_labs||0),'Current':s.current_lab?prettyLabName(s.current_lab):'-','Error':s.last_error||'none'};
  document.getElementById('researchMetrics').innerHTML=Object.entries(vals).map(([k,v])=>`<div class="metric"><span class="muted">${k}</span><b>${esc(v)}</b></div>`).join('');
  const completed=new Set(Object.keys(s.labs||{}));
  document.getElementById('researchLabGrid').innerHTML=researchNames.map(name=>{
    let state='WAITING',cls='waiting';
    if(completed.has(name)){state='DONE';cls='done'}
    if(s.running && s.current_lab===name){state='RUNNING';cls='running'}
    if(s.last_error && s.current_lab===name){state='ERROR';cls='error'}
    return `<div class="lab-mini"><b>${esc(prettyLabName(name))}</b><span class="state ${cls}">${state}</span></div>`;
  }).join('');
  document.getElementById('researchResults').textContent=JSON.stringify(s.labs||{},null,2);
  updateOverview(lastStrategyState,lastResearchState,lastCryptoState,lastCoordinatorState);
 }catch(e){
  document.getElementById('researchResults').textContent=e.message==='Unauthorized'?'Dashboard token ontbreekt of is ongeldig.':e.message;
 }
}
async function loadCoordinator(){
 try{
  lastCoordinatorState=await api('coordinator/status');
  updateOverview(lastStrategyState,lastResearchState,lastCryptoState,lastCoordinatorState);
 }catch(e){}
}
async function researchAction(x){try{await api('research/'+x,'POST');await loadResearch()}catch(e){document.getElementById('researchResults').textContent=e.message}}
const savedToken=sessionStorage.getItem('microtraderDashboardToken')||'';
document.getElementById('token').value=savedToken;
document.getElementById('token').addEventListener('input',e=>sessionStorage.setItem('microtraderDashboardToken',e.target.value));
setInterval(()=>{if(token()){loadStatus();loadForex();loadPrecision();loadAgent();loadLab();loadCrypto();loadResearch();loadCoordinator()}},5000);
if(token()){loadStatus();loadForex();loadPrecision();loadAgent();loadLab();loadCrypto();loadResearch();loadCoordinator();}
</script>
</body></html>'''
