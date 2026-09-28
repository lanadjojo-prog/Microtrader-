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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

client = AlpacaClient(settings)
engine = TradingEngine(settings, client)
lab = StrategyLab(settings, client)
research = ResearchLabs(settings, client)
crypto_lab = CryptoMicrostructureLab(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.auto_start:
        await engine.start()
    if settings.lab_auto_start and settings.api_key and settings.api_secret:
        await lab.start()
    if settings.research_auto_start and settings.api_key and settings.api_secret:
        await research.start()
    if settings.crypto_lab_auto_start:
        await crypto_lab.start()
    yield
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
    if not settings.api_key or not settings.api_secret:
        raise HTTPException(status_code=503, detail="Alpaca API credentials are not configured")
    await lab.start()
    return {"ok": True, "running": True}


@app.post("/api/lab/stop")
async def lab_stop(authorization: str | None = Header(default=None)):
    require_token(authorization)
    await lab.stop()
    return {"ok": True, "running": False}


@app.get("/api/research/status")
async def research_status(authorization: str | None = Header(default=None)):
    require_token(authorization)
    return research.public_state()


@app.post("/api/research/start")
async def research_start(authorization: str | None = Header(default=None)):
    require_token(authorization)
    if not settings.api_key or not settings.api_secret:
        raise HTTPException(status_code=503, detail="Alpaca API credentials are not configured")
    await research.start()
    return {"ok": True, "running": True}


@app.post("/api/research/stop")
async def research_stop(authorization: str | None = Header(default=None)):
    require_token(authorization)
    await research.stop()
    return {"ok": True, "running": False}


@app.get("/api/crypto-lab/status")
async def crypto_lab_status(authorization: str | None = Header(default=None)):
    require_token(authorization)
    return crypto_lab.public_state()


@app.post("/api/crypto-lab/start")
async def crypto_lab_start(authorization: str | None = Header(default=None)):
    require_token(authorization)
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
body{font-family:system-ui,-apple-system,sans-serif;background:#0e1116;color:#e9eef5;margin:0;padding:28px;max-width:1100px;margin:auto}
.card{background:#171c24;border:1px solid #2a3442;border-radius:16px;padding:18px;margin:14px 0}.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}button,input{font:inherit;border-radius:10px;border:1px solid #3a4658;padding:10px 13px;background:#0f141b;color:#fff}button{cursor:pointer}button.danger{border-color:#8a3841}.muted{color:#96a4b5}pre{white-space:pre-wrap;word-break:break-word}h1{margin-bottom:0}.pill{padding:5px 9px;border-radius:999px;background:#242d39}.pill.good{background:#17351f}.pill.bad{background:#3a1c22}.pill.run{background:#17314a}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px}.metric{background:#111720;padding:12px;border-radius:12px}.metric b{display:block;font-size:1.3rem;margin-top:4px}.progress{height:12px;background:#0f141b;border:1px solid #2a3442;border-radius:999px;overflow:hidden;margin:12px 0}.progress>div{height:100%;background:#e9eef5;width:0%;transition:width .25s ease}.lab-head{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap}.lab-table{width:100%;border-collapse:collapse;margin-top:14px;font-size:.92rem}.lab-table th,.lab-table td{text-align:left;padding:9px 8px;border-bottom:1px solid #2a3442;vertical-align:top}.lab-table th{color:#96a4b5;font-weight:600}.ok{color:#8de39e}.no{color:#ff9da8}.small{font-size:.85rem}.scroll{overflow-x:auto}.overview{position:sticky;top:8px;z-index:20;background:#171c24ee;backdrop-filter:blur(8px)}.lab-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-top:12px}.lab-mini{background:#111720;border:1px solid #2a3442;border-radius:12px;padding:12px}.lab-mini b{display:block;margin-bottom:6px}.lab-mini .state{font-size:.8rem;font-weight:700}.state.done{color:#8de39e}.state.running{color:#8ecbff}.state.waiting{color:#96a4b5}.state.error{color:#ff9da8}details{margin-top:12px}summary{cursor:pointer;color:#96a4b5}.nav{display:flex;gap:8px;flex-wrap:wrap;margin:16px 0}.nav a{color:#dbe7f5;text-decoration:none;background:#111720;border:1px solid #2a3442;padding:9px 12px;border-radius:10px}.section-title{display:flex;justify-content:space-between;gap:10px;align-items:center;flex-wrap:wrap}.mode-banner{padding:10px 12px;border-radius:12px;background:#132318;border:1px solid #275335;color:#a7e6b3;margin:10px 0}.signal-on{color:#8de39e;font-weight:700}.signal-off{color:#738195}.crypto-card{background:#111720;border:1px solid #2a3442;border-radius:12px;padding:14px}.crypto-market{font-size:1.05rem;font-weight:700;margin-bottom:8px}</style>
</head>
<body>
<h1>MicroTrader</h1><p class="muted">Trading + research dashboard · safe-by-default</p>
<div class="nav"><a href="#engine">Trading Engine</a><a href="#stock-lab">Stock Lab</a><a href="#crypto-lab">Crypto Lab</a><a href="#research-labs">Research Labs</a></div>
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
<div class="card" id="stock-lab">
  <div class="section-title"><h3>Stock Strategy Lab</h3><span class="pill">ALPACA · RESEARCH</span></div>
  <p class="muted">Backtests candidate strategies on historical Alpaca bars with a chronological holdout and stressed transaction costs.</p>
  <div class="row"><button onclick="labAction('start')">Run Strategy Lab</button><button onclick="labAction('stop')">Stop Lab</button><button onclick="loadLab(true)">Refresh Lab</button></div>
  <div class="lab-head" style="margin-top:14px">
    <div id="labStatus"><span class="pill">IDLE</span></div>
    <div id="labUpdated" class="muted small"></div>
  </div>
  <div class="progress"><div id="labProgress"></div></div>
  <div id="labMessage" class="muted">Not run yet.</div>
  <div id="labMetrics" class="grid" style="margin-top:12px"></div>
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
  <h4 style="margin-bottom:6px">Live Market Signals</h4>
  <div id="cryptoMarkets" class="lab-grid"></div>
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
 const vals={
   'Data loaded': symbolTotal?symbolDone+'/'+symbolTotal:'-',
   'Candidates': total?done+'/'+total:'-',
   'Tested': sum.candidates_tested??results.length,
   'Promoted': (sum.promoted_count??s.promoted_total??0)+'/'+(sum.target_promoted??s.target_promoted??'-')
 };
 document.getElementById('labMetrics').innerHTML=Object.entries(vals).map(([k,v])=>`<div class="metric"><span class="muted">${k}</span><b>${v}</b></div>`).join('');
 if(!results.length){
   document.getElementById('labResults').innerHTML='<p class="muted">No results yet. During data loading this is normal.</p>';
   return;
 }
 const promoted=results.filter(x=>x.promoted);
 const rejected=results.filter(x=>!x.promoted);
 function labRows(items){
   return items.map((x,i)=>{
     const o=x.oos||{}, st=x.stress_oos||{};
     const status=x.promoted?'<span class="ok">PROMOTED</span>':'<span class="no">REJECTED</span>';
     const why=x.promoted?'—':esc((x.rejection_reasons||[]).join('; '));
     return `<tr><td>${i+1}</td><td>${esc(x.strategy)}</td><td class="small">${esc(JSON.stringify(x.params))}</td><td>${status}</td><td>${o.trades??0}</td><td>${fmt(o.win_rate_pct)}%</td><td>${fmt(o.expectancy_bps,3)} bps</td><td>${fmt(o.profit_factor,3)}</td><td>${fmt(o.max_drawdown_pct,3)}%</td><td>${fmt(st.expectancy_bps,3)} bps</td><td class="small">${why}</td></tr>`;
   }).join('');
 }
 const head='<table class="lab-table"><thead><tr><th>#</th><th>Strategy</th><th>Parameters</th><th>Status</th><th>OOS trades</th><th>Win rate</th><th>OOS expectancy</th><th>PF</th><th>Drawdown</th><th>Stress expectancy</th><th>Reason</th></tr></thead><tbody>';
 let html='<div class="small muted" style="margin:10px 0">'+promoted.length+' promoted · '+rejected.length+' rejected</div>';
 if(promoted.length) html+=head+labRows(promoted)+'</tbody></table>';
 else html+='<p class="muted">Nog geen promoted strategieën.</p>';
 if(rejected.length) html+='<details><summary>Rejected strategieën tonen ('+rejected.length+')</summary>'+head+labRows(rejected)+'</tbody></table></details>';
 document.getElementById('labResults').innerHTML=html;
}
async function loadLab(manual=false){
 try{
  const [s,r]=await Promise.all([api('lab/status'),api('lab/results')]);
  renderLab(s,r.results||[]);
  lastStrategyState=s;
  updateOverview(lastStrategyState,lastResearchState,lastCryptoState);
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
    'Cycles':s.cycles||0,
    'Polling':(s.poll_seconds||'-')+' sec',
    'Maker fee assumption':fmt(s.maker_fee_bps_one_way,1)+' bps / side',
    'Last update':s.last_update?new Date(s.last_update).toLocaleTimeString():'-'
  };
  document.getElementById('cryptoMetrics').innerHTML=Object.entries(vals).map(function(kv){
    return '<div class="metric"><span class="muted">'+esc(kv[0])+'</span><b>'+esc(kv[1])+'</b></div>';
  }).join('');
  const paper=(s.paper_simulation||{});
  const strategies=(paper.strategies||{});
  const strategyOrder=['mean_reversion','market_maker','imbalance','hybrid'];
  const rows=strategyOrder.map(function(name){
    const x=strategies[name]||{};
    const pnl=Number(x.net_pnl_eur||0);
    const pnlClass=pnl>0?'ok':(pnl<0?'no':'muted');
    return '<tr>'+
      '<td>'+esc(prettyLabName(name))+'</td>'+
      '<td>'+(x.trades||0)+'</td>'+
      '<td>'+fmt(x.win_rate_pct,1)+'%</td>'+
      '<td class="'+pnlClass+'">€'+fmt(pnl,4)+'</td>'+
      '<td>'+fmt(x.expectancy_bps,3)+' bps</td>'+
      '<td>€'+fmt(x.max_drawdown_eur,4)+'</td>'+
      '<td>'+((x.open_positions||0)+(x.pending_entries||0))+'</td>'+
    '</tr>';
  }).join('');
  document.getElementById('cryptoScoreboard').innerHTML=
    '<div class="small muted" style="margin:6px 0 10px">Paper model: '+esc(paper.fill_model||'-')+
    ' · €'+fmt(paper.notional_eur_per_trade,2)+' per trade · maker fees inbegrepen</div>'+
    '<table class="lab-table"><thead><tr><th>Strategy</th><th>Trades</th><th>Winrate</th><th>Net P&L</th><th>Expectancy</th><th>Max DD</th><th>Open/Pending</th></tr></thead><tbody>'+rows+'</tbody></table>';
  const markets=s.markets||[];
  document.getElementById('cryptoMarkets').innerHTML=markets.length?markets.map(function(m){
    function sig(on,label){return '<span class="'+(on?'signal-on':'signal-off')+'">'+(on?'●':'○')+' '+label+'</span>';}
    return '<div class="crypto-card">'+
      '<div class="crypto-market">'+esc(m.market)+'</div>'+
      '<div class="small muted">Mid €'+fmt(m.mid,2)+' · Spread '+fmt(m.spread_bps,2)+' bps</div>'+
      '<div class="small muted">Z-score '+fmt(m.zscore,2)+' · Imbalance '+fmt(m.imbalance,2)+'</div>'+
      '<div class="small" style="margin-top:8px">'+sig(m.mean_reversion_signal,'Mean reversion')+'<br>'+sig(m.market_maker_signal,'Maker spread')+'<br>'+sig(m.imbalance_signal,'Order-book imbalance')+'<br>'+sig(m.hybrid_signal,'Hybrid')+'</div>'+
    '</div>';
  }).join(''):'<p class="muted">Start het lab om live marktdata te verzamelen.</p>';
  document.getElementById('cryptoRaw').textContent=JSON.stringify({paper_simulation:s.paper_simulation,strategy_stats:s.strategy_stats,markets:s.markets},null,2);
  updateOverview(lastStrategyState,lastResearchState,lastCryptoState);
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
function updateOverview(strategyState,researchState,cryptoState){
 const strategyRunning=!!strategyState?.running, researchRunning=!!researchState?.running, cryptoRunning=!!cryptoState?.running;
 document.getElementById('overallStatus').innerHTML=
   `<span class="pill ${strategyRunning?'run':'good'}">Strategy Lab: ${strategyRunning?'RUNNING':'STOPPED'}</span>`+
   `<span class="pill ${researchRunning?'run':'good'}">Research Labs: ${researchRunning?'RUNNING':'STOPPED'}</span>`+
   `<span class="pill ${cryptoRunning?'run':'good'}">Crypto Lab: ${cryptoRunning?'COLLECTING':'STOPPED'}</span>`+
   (researchState?.current_lab?`<span class="muted">Current: ${esc(prettyLabName(researchState.current_lab))}</span>`:'');
 const vals={
   'Strategies tested':strategyState?.tested_total??'-',
   'Strategies promoted':(strategyState?.promoted_total??0)+'/'+(strategyState?.target_promoted??'-'),
   'Research progress':(researchState?.completed_labs??0)+'/'+(researchState?.total_labs??17),
   'Active research':researchState?.current_lab?prettyLabName(researchState.current_lab):'-',
   'Crypto observations':cryptoState?.observations??'-'
 };
 document.getElementById('overallMetrics').innerHTML=Object.entries(vals).map(([k,v])=>`<div class="metric"><span class="muted">${k}</span><b>${esc(v)}</b></div>`).join('');
}
let lastStrategyState=null,lastResearchState=null,lastCryptoState=null;
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
  updateOverview(lastStrategyState,lastResearchState,lastCryptoState);
 }catch(e){
  document.getElementById('researchResults').textContent=e.message==='Unauthorized'?'Dashboard token ontbreekt of is ongeldig.':e.message;
 }
}
async function researchAction(x){try{await api('research/'+x,'POST');await loadResearch()}catch(e){document.getElementById('researchResults').textContent=e.message}}
const savedToken=sessionStorage.getItem('microtraderDashboardToken')||'';
document.getElementById('token').value=savedToken;
document.getElementById('token').addEventListener('input',e=>sessionStorage.setItem('microtraderDashboardToken',e.target.value));
setInterval(()=>{if(token()){loadStatus();loadLab();loadCrypto();loadResearch()}},5000);
if(token()){loadStatus();loadLab();loadCrypto();loadResearch();}
</script>
</body></html>'''
