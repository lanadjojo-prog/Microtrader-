from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from config import settings
from ctrader_client import CTraderClient, CTraderError
from strategy_lab import StrategyLab
from research_labs import ResearchLabs
from research_agent import ResearchAgent
from research_coordinator import ResearchCoordinator
from forex_lab import ForexStrategyLab
from precision_lab import PrecisionStrategyLab
from paper_trader import PaperTradingEngine
from oauth_store import CTraderTokenStore

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
ctrader=CTraderClient(settings)
research_ctrader=CTraderClient(settings)
validation_ctrader=CTraderClient(settings)
forex_ctrader=CTraderClient(settings)
precision_ctrader=CTraderClient(settings)
paper_ctrader=CTraderClient(settings)
strategy_lab=StrategyLab(settings,research_ctrader)
research_labs=ResearchLabs(settings,validation_ctrader)
research_agent=ResearchAgent(settings,strategy_lab)
research_coordinator=ResearchCoordinator(strategy_lab,research_labs)
forex_lab=ForexStrategyLab(settings,forex_ctrader)
precision_lab=PrecisionStrategyLab(settings,precision_ctrader)
paper_engine=PaperTradingEngine(settings,paper_ctrader)
token_store=CTraderTokenStore(settings.database_url)

async def restore_ctrader():
    await token_store.init()
    saved=await token_store.load()
    access=str(saved.get("access_token") or "")
    refresh=str(saved.get("refresh_token") or "")
    if not access and not refresh:
        return False
    ctrader.set_runtime_tokens(access,refresh)
    research_ctrader.set_runtime_tokens(access,refresh)
    validation_ctrader.set_runtime_tokens(access,refresh)
    forex_ctrader.set_runtime_tokens(access,refresh)
    precision_ctrader.set_runtime_tokens(access,refresh)
    paper_ctrader.set_runtime_tokens(access,refresh)
    try:
        await ctrader.connect_and_authenticate()
    except Exception:
        if not refresh:
            return False
        try:
            fresh=await ctrader.refresh_access_token(refresh)
            access=str(fresh.get("accessToken") or ctrader.active_access_token)
            refresh=str(fresh.get("refreshToken") or ctrader.active_refresh_token)
            await token_store.save(access,refresh)
            research_ctrader.set_runtime_tokens(access,refresh)
            validation_ctrader.set_runtime_tokens(access,refresh)
            forex_ctrader.set_runtime_tokens(access,refresh)
            precision_ctrader.set_runtime_tokens(access,refresh)
            paper_ctrader.set_runtime_tokens(access,refresh)
            await ctrader.connect_and_authenticate()
        except Exception:
            logging.getLogger("microtrader").exception("Could not restore cTrader session")
            return False
    return True

async def autostart():
    if settings.lab_auto_start: await strategy_lab.start()
    if settings.research_agent_auto_start: await research_agent.start()
    if settings.research_auto_start: await research_coordinator.start()
    if settings.forex_lab_auto_start: await forex_lab.start()
    if settings.precision_lab_auto_start: await precision_lab.start()
    if settings.paper_trading_auto_start: await paper_engine.start()
    logging.getLogger("microtrader").info(
        "WORKERS research=%s agent=%s coordinator=%s forex=%s precision=%s paper=%s "
        "research_stage=%s forex_stage=%s precision_stage=%s paper_stage=%s",
        strategy_lab.state.running,
        research_agent.state.running,
        research_coordinator.state.running,
        forex_lab.state.running,
        precision_lab.state.running,
        paper_engine.state.running,
        strategy_lab.state.stage,
        forex_lab.state.stage,
        precision_lab.state.stage,
        paper_engine.state.stage,
    )

@asynccontextmanager
async def lifespan(app: FastAPI):
    await restore_ctrader()
    await autostart(); yield
    await paper_engine.stop(); await precision_lab.stop(); await forex_lab.stop()
    await research_coordinator.stop(); await research_agent.stop(); await research_labs.stop(); await strategy_lab.stop()
    await paper_ctrader.close(); await precision_ctrader.close(); await forex_ctrader.close()
    await validation_ctrader.close(); await research_ctrader.close(); await ctrader.close()

app=FastAPI(title="ForexTrader Research",lifespan=lifespan)
def auth(a):
    if not settings.dashboard_token or a!=f"Bearer {settings.dashboard_token}": raise HTTPException(401,"Unauthorized")

@app.get('/health')
async def health(): return {'ok':True,'mode':'forex-research-and-paper-only','live_trading_enabled':False,'research_market':'forex','research_data_source':'ctrader','research_running':strategy_lab.state.running,'research_stage':strategy_lab.state.stage,'research_validation_running':research_labs.state.running,'research_coordinator_mode':research_coordinator.state.mode,'research_agent_running':research_agent.state.running,'research_agent_stage':research_agent.state.stage,'forex_lab_running':forex_lab.state.running,'forex_stage':forex_lab.state.stage,'precision_lab_running':precision_lab.state.running,'precision_stage':precision_lab.state.stage,'paper_running':paper_engine.state.running,'paper_stage':paper_engine.state.stage,'ctrader_ready':ctrader.api_ready}
@app.get('/api/research/status')
async def research_status(authorization:str|None=Header(None)): auth(authorization); return {'strategy':strategy_lab.public_state(),'validation':research_labs.public_state(),'coordinator':research_coordinator.public_state(),'agent':asdict(research_agent.state)}
@app.get('/api/research/results')
async def research_results(authorization:str|None=Header(None)): auth(authorization); return {'state':strategy_lab.public_state(),'results':strategy_lab.results(),'validation':research_labs.public_state(),'coordinator':research_coordinator.public_state(),'agent':asdict(research_agent.state)}
@app.post('/api/research/start')
async def research_start(authorization:str|None=Header(None)): auth(authorization); await strategy_lab.start(); await research_agent.start(); await research_coordinator.start(); return {'strategy':strategy_lab.public_state(),'coordinator':research_coordinator.public_state(),'agent':asdict(research_agent.state)}
@app.post('/api/research/stop')
async def research_stop(authorization:str|None=Header(None)): auth(authorization); await research_coordinator.stop(); await research_agent.stop(); await research_labs.stop(); await strategy_lab.stop(); return {'strategy':strategy_lab.public_state(),'coordinator':research_coordinator.public_state(),'agent':asdict(research_agent.state)}
@app.get('/api/forex-lab/status')
async def fs(authorization:str|None=Header(None)): auth(authorization); return forex_lab.public_state()
@app.get('/api/forex-lab/results')
async def fr(authorization:str|None=Header(None)): auth(authorization); return {'state':forex_lab.public_state(),'results':forex_lab.results()}
@app.post('/api/forex-lab/start')
async def fst(authorization:str|None=Header(None)): auth(authorization); await forex_lab.start(); return forex_lab.public_state()
@app.post('/api/forex-lab/stop')
async def fsp(authorization:str|None=Header(None)): auth(authorization); await forex_lab.stop(); return forex_lab.public_state()
@app.get('/api/precision-lab/status')
async def ps(authorization:str|None=Header(None)): auth(authorization); return precision_lab.public_state()
@app.get('/api/precision-lab/results')
async def pr(authorization:str|None=Header(None)): auth(authorization); return {'state':precision_lab.public_state(),'results':precision_lab.results()}
@app.post('/api/precision-lab/start')
async def pst(authorization:str|None=Header(None)): auth(authorization); await precision_lab.start(); return precision_lab.public_state()
@app.post('/api/precision-lab/stop')
async def psp(authorization:str|None=Header(None)): auth(authorization); await precision_lab.stop(); return precision_lab.public_state()
@app.get('/api/paper/status')
async def paper_status(authorization:str|None=Header(None)): auth(authorization); return paper_engine.public_state()
async def _paper_results_with_marks():
    data = await paper_engine.results()
    positions = list(data.get("positions") or [])
    cache = {}
    for row in positions:
        try:
            params = dict(row.get("params") or {})
            tf = int(params.get("timeframe_min") or 1)
            pair = str(row.get("pair") or "")
            key = (pair, tf)
            if key not in cache:
                bars = await paper_ctrader.historical_bars(
                    pair, timeframe_min=tf, max_bars=5, lookback_days=1
                )
                cache[key] = float(bars[-1]["c"]) if bars else None
            mark = cache.get(key)
            if mark is None:
                continue
            entry = float(row.get("entry_price") or 0)
            risk_distance = float(row.get("risk_distance") or 0)
            direction = int(row.get("direction") or 0)
            risk_eur = float(row.get("risk_eur") or 0)
            gross = direction * ((mark / entry) - 1.0) if entry > 0 else 0.0
            net = gross - (2.0 * float(settings.forex_cost_bps) / 10_000.0)
            risk_pct = risk_distance / entry if entry > 0 else 0.0
            open_r = net / risk_pct if risk_pct > 0 else 0.0
            row["current_price"] = mark
            row["unrealized_r"] = round(open_r, 4)
            row["unrealized_pnl"] = round(open_r * risk_eur, 4)
        except Exception:
            continue
    data["positions"] = positions
    return data

@app.get('/api/paper/results')
async def paper_results(authorization:str|None=Header(None)):
    auth(authorization)
    return await _paper_results_with_marks()
@app.post('/api/paper/start')
async def paper_start(authorization:str|None=Header(None)): auth(authorization); await paper_engine.start(); return paper_engine.public_state()
@app.post('/api/paper/stop')
async def paper_stop(authorization:str|None=Header(None)): auth(authorization); await paper_engine.stop(); return paper_engine.public_state()
@app.get('/api/ctrader/oauth-url')
async def oauth(authorization:str|None=Header(None)):
    auth(authorization)
    try:return {'url':ctrader.authorization_url()}
    except CTraderError as e:raise HTTPException(503,str(e))
@app.get('/ctrader/start')
async def cs():
    try:return RedirectResponse(ctrader.authorization_url())
    except CTraderError as e:return HTMLResponse(f'<h3>cTrader niet gereed</h3><p>{e}</p>',503)
@app.get('/ctrader/callback',response_class=HTMLResponse)
async def cb(code:str=''):
    if not code:return HTMLResponse('<h3>Geen authorization code.</h3>',400)
    try:
        await ctrader.exchange_code(code)
        await token_store.init()
        await token_store.save(ctrader.active_access_token,ctrader.active_refresh_token)
        research_ctrader.set_runtime_tokens(ctrader.active_access_token, ctrader.active_refresh_token)
        validation_ctrader.set_runtime_tokens(ctrader.active_access_token, ctrader.active_refresh_token)
        forex_ctrader.set_runtime_tokens(ctrader.active_access_token, ctrader.active_refresh_token)
        precision_ctrader.set_runtime_tokens(ctrader.active_access_token, ctrader.active_refresh_token)
        paper_ctrader.set_runtime_tokens(ctrader.active_access_token, ctrader.active_refresh_token)
        state=await ctrader.connect_and_authenticate()
        await autostart()
        return HTMLResponse(f"<h3>Fusion Markets demo connected</h3><p>{state.get('environment','-')} · account {state.get('account_id','-')}</p><p>Research labs zijn gestart waar mogelijk. Live orders staan uit.</p>")
    except Exception as e:return HTMLResponse(f'<h3>cTrader authorization failed</h3><p>{e}</p>',503)
@app.get('/',response_class=HTMLResponse)
async def home():
    return HTMLResponse(
        DASHBOARD,
        headers={
            "Cache-Control":"no-store, no-cache, must-revalidate, max-age=0",
            "Pragma":"no-cache",
            "Expires":"0",
        },
    )

DASHBOARD=r'''<!doctype html><html lang="nl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ForexTrader</title><style>
*{box-sizing:border-box}body{font-family:system-ui,-apple-system,sans-serif;background:#0e1116;color:#edf2f7;margin:0;padding:24px;max-width:1000px;margin:auto}h1{margin:0}.muted{color:#96a4b5}.top,.row{display:flex;gap:9px;flex-wrap:wrap;align-items:center}.top{justify-content:space-between;margin-bottom:18px}.card{background:#171c24;border:1px solid #2a3442;border-radius:16px;padding:18px;margin:14px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(135px,1fr));gap:9px;margin-top:12px}.metric{background:#111720;padding:11px;border-radius:11px}.metric b{display:block;font-size:1.1rem;margin-top:4px}.pill{padding:5px 9px;border-radius:999px;background:#242d39;font-size:.82rem}.good{background:#17351f;color:#9ee8ad}.run{background:#17314a;color:#a8d8ff}.bad{background:#3a1c22;color:#ffadb5}button,input{font:inherit;border-radius:10px;border:1px solid #3a4658;padding:10px 12px;background:#0f141b;color:#fff}button{cursor:pointer}.progress{height:9px;background:#0f141b;border-radius:99px;overflow:hidden;margin:12px 0}.progress div{height:100%;background:#e9eef5;width:0}.results{overflow:auto;margin-top:12px;max-height:420px;border:1px solid #263241;border-radius:10px}.results table{min-width:620px}table{width:100%;border-collapse:collapse;font-size:.84rem}th,td{text-align:left;padding:8px;border-bottom:1px solid #2a3442}th{color:#96a4b5;position:sticky;top:0;background:#171c24;z-index:2}.overview{position:static}.config{font-size:.9rem;line-height:1.5}.pipeline-head{display:flex;justify-content:space-between;align-items:end;gap:10px;margin-top:18px}.pipeline-head b{font-size:.92rem}.pipeline-head span{font-size:.76rem;color:#96a4b5}.phases{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:10px;margin-top:9px}.phase{position:relative;background:linear-gradient(180deg,#131a24,#10161f);border:1px solid #2b3746;border-radius:14px;padding:12px;min-height:128px;overflow:hidden}.phase:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:#4b5a6b}.phase.promoted:before{background:#62d985}.phase.deep_search:before,.phase.precision_deep_search:before{background:#72b8ff}.phase.incubator:before{background:#d9b962}.phase.discovery:before,.phase.precision_discovery:before{background:#9b8cff}.phase.rejected{opacity:.72}.phase.rejected:before{background:#d66b76}.phase-top{display:flex;justify-content:space-between;gap:8px;align-items:flex-start}.phase h4{margin:0;font-size:.82rem;text-transform:uppercase;letter-spacing:.04em}.phase .count{font-size:1.55rem;font-weight:800;line-height:1}.phase .sub{color:#96a4b5;font-size:.72rem;margin-top:4px}.phase .names{display:flex;flex-wrap:wrap;gap:5px;margin-top:10px;max-height:76px;overflow:auto}.phase .tag{font-size:.72rem;line-height:1.2;padding:5px 7px;border-radius:999px;background:#202a36;color:#cbd6e3;border:1px solid #2f3a48}.phase-empty{grid-column:1/-1;background:#111720;border:1px dashed #2e3a49;border-radius:12px;padding:14px;color:#96a4b5}.rules{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:9px;margin-top:10px}.rule{background:#111720;border:1px solid #263241;border-radius:12px;padding:11px}.rule b{display:block;margin-top:3px}details{margin-top:12px;border-top:1px solid #263241;padding-top:10px}summary{cursor:pointer;color:#b7c4d3;font-weight:650}.pnlpos{color:#9ee8ad}.pnlneg{color:#ffadb5}@media(max-width:600px){body{padding:13px}.card{padding:13px;margin:10px 0}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}h1{font-size:1.5rem}.metric{padding:9px}button{padding:9px 10px}.results{max-height:320px}.phases{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body><div class="top"><div><h1>ForexTrader</h1><div class="muted">100% forex research via cTrader · geen stocks · geen live trading</div></div><span class="pill good">RESEARCH ONLY</span></div>
<div class="card"><div class="row"><input id="token" type="password" placeholder="Dashboard token"><button onclick="connect()">Connect</button><button onclick="authorize()">cTrader koppelen</button></div><div id="connection" class="muted" style="margin-top:8px">Voer je token in.</div></div>
<div class="card overview"><div class="row" style="justify-content:space-between"><div class="row"><b>Status</b><span id="rmini" class="pill">Research —</span><span id="fmini" class="pill">Forex —</span><span id="pmini" class="pill">Precision —</span><span id="papermini" class="pill">Paper —</span></div><span class="pill">DASHBOARD V4</span></div></div>
<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">Funnel overzicht</h2><span class="pill good">LIVE PER LAB</span></div><p class="muted config">Hier zie je direct hoeveel kandidaten in elke fase zitten en welke strategieën dat zijn.</p><h3>Research Lab</h3><div id="overview-research-phases" class="phases"><div class="phase-empty">Laden…</div></div><h3>Forex Lab</h3><div id="overview-forex-phases" class="phases"><div class="phase-empty">Laden…</div></div><h3>Precision Lab</h3><div id="overview-precision-phases" class="phases"><div class="phase-empty">Laden…</div></div></div>
<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">Actieve regels</h2><span class="pill good">HARD FILTERS</span></div><div class="rules"><div class="rule"><span class="muted">Timeframes</span><b>1m + 5m</b></div><div class="rule"><span class="muted">Per strategie</span><b>≥ 3 trades/dag</b></div><div class="rule"><span class="muted">Voorkeur</span><b>5–10 trades/dag</b></div><div class="rule"><span class="muted">Portfolio</span><b>≥ 10 trades/dag</b></div><div class="rule"><span class="muted">Entry-sessies</span><b>London + New York</b></div><div class="rule"><span class="muted">Volume</span><b>≥ 0,70× normaal</b></div><div class="rule"><span class="muted">Startkapitaal</span><b>€50 paper</b></div><div class="rule"><span class="muted">Live geld</span><b>UIT</b></div></div><p class="muted config">Nieuwe entries alleen tijdens liquide London/New York-uren, DST-aware. Stops, targets en positiebeheer blijven ook buiten die uren actief.</p></div>
<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">Research Lab</h2><span class="pill">FOREX DISCOVERY + 17-STAGE VALIDATION</span></div><p class="config muted"><b>Forex-only via cTrader/Fusion · 1m/5m · harde ondergrens 3 trades/dag per strategie.</b> Geen SPY/AAPL/NVDA of andere stocks meer. Dit lab ontdekt breed forex-ideeën; Forex Lab verfijnt daarna de kansrijke FX-kandidaten.</p><div class="row"><button onclick="researchAct('start')">Start</button><button onclick="researchAct('stop')">Stop</button><button onclick="loadResearch()">Refresh</button></div><div id="research-status" class="row" style="margin-top:12px"></div><div id="research-message" class="muted"></div><div id="research-metrics" class="grid"></div><div class="pipeline-head"><b>Funnel per fase</b><span>hoeveel + welke strategieën</span></div><div id="research-phases" class="phases"></div><details><summary>Alle Research-resultaten</summary><div id="research-results" class="results"></div></details></div>
<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">Forex Lab</h2><span class="pill">BREDE FUNNEL</span></div><p class="config muted"><b>HARDE EIS: gemiddeld minimaal 3 trades per handelsdag per strategie.</b> Vanaf 5–10 trades/dag krijgt frequentie extra score. Portfolio-doel: ≥10 trades/dag. Breakout · trend pullback · range reversal · exit-management varianten.</p><div class="row"><button onclick="act('forex-lab','start')">Start</button><button onclick="act('forex-lab','stop')">Stop</button><button onclick="load('forex-lab')">Refresh</button></div><div id="forex-lab-status" class="row" style="margin-top:12px"></div><div id="forex-lab-message" class="muted"></div><div class="progress"><div id="forex-lab-progress"></div></div><div id="forex-lab-metrics" class="grid"></div><div id="forex-lab-promoted"></div><div class="pipeline-head"><b>Funnel per fase</b><span>hoeveel + welke strategieën</span></div><div id="forex-lab-phases" class="phases"></div><details><summary>Alle Forex-testresultaten</summary><div id="forex-lab-results" class="results"></div></details></div>
<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">Paper Trading</h2><span class="pill good">€50 START · SIMULATED</span></div><p class="config muted">Alleen geldige promoted 1m/5m-strategieën. Geen brokerorders.</p><div class="row"><button onclick="paperAct('start')">Start</button><button onclick="paperAct('stop')">Stop</button><button onclick="loadPaper()">Refresh</button></div><div id="paper-status" class="row" style="margin-top:12px"></div><div id="paper-message" class="muted"></div><div id="paper-strategies"></div><h3>Open posities · actuele stand</h3><div id="paper-positions" class="results"></div><h3>Per dag</h3><div id="paper-daily" class="results"></div><details><summary>Gesloten trades</summary><div id="paper-trades" class="results"></div></details></div>
<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">Precision Lab</h2><span class="pill">PIP FUNNEL</span></div><p class="config muted"><b>HARDE EIS: gemiddeld minimaal 3 trades per handelsdag per strategie.</b> Frequentievoorkeur 5–10/dag; portfolio-doel ≥10/dag. €50 start · €2/€3 risico · stops 2–5 pips · targets 4–10 pips · bid/ask tick execution.</p><div class="row"><button onclick="act('precision-lab','start')">Start</button><button onclick="act('precision-lab','stop')">Stop</button><button onclick="load('precision-lab')">Refresh</button></div><div id="precision-lab-status" class="row" style="margin-top:12px"></div><div id="precision-lab-message" class="muted"></div><div class="progress"><div id="precision-lab-progress"></div></div><div id="precision-lab-metrics" class="grid"></div><div class="pipeline-head"><b>Funnel per fase</b><span>hoeveel + welke strategieën</span></div><div id="precision-lab-phases" class="phases"></div><details><summary>Alle Precision-testresultaten</summary><div id="precision-lab-results" class="results"></div></details></div>
<script>const $=x=>document.getElementById(x);let tok=localStorage.getItem('ftToken')||'';$('token').value=tok;function H(){tok=$('token').value.trim();localStorage.setItem('ftToken',tok);return{'Authorization':'Bearer '+tok}}async function req(u,o={}){o.headers={...(o.headers||{}),...H()};let r=await fetch(u,o),j=await r.json().catch(()=>({}));if(!r.ok)throw Error(j.detail||'HTTP '+r.status);return j}function M(a,b){return`<div class="metric"><span class="muted">${a}</span><b>${b??'—'}</b></div>`}function D(rs){let ps=(rs||[]).filter(r=>(r.status||r.funnel_stage)==='promoted');if(!ps.length)return'';return ps.map((r,i)=>{let m=r.oos||{},st=r.stress_oos||{},p=r.params||{};return `<div class="card" style="margin:12px 0 0;padding:14px;background:#10161f"><div class="row" style="justify-content:space-between"><b>Promoted #${i+1}: ${r.strategy||'—'}</b><span class="pill good">PROMOTED</span></div><div class="grid">${M('Timeframe',(r.timeframe_min??p.timeframe_min??'—')+'m')}${M('Risk','€'+(p.risk_eur??r.risk_model?.fixed_risk_eur??'—'))}${M('Target',(p.target_r??'—')+'R')}${M('Stop ATR',p.stop_atr??'—')}${M('OOS trades',m.trades??'—')}${M('Trades/dag',m.avg_trades_per_day??'—')}${M('Winrate',(m.win_rate_pct??'—')+'%')}${M('Profit factor',m.profit_factor??'—')}${M('Expectancy',(m.expectancy_r??'—')+'R')}${M('Max DD',(m.max_drawdown_pct??'—')+'%')}${M('Payoff',m.payoff_ratio??'—')}${M('Stress PF',st.profit_factor??'—')}${M('Stress exp',(st.expectancy_r??'—')+'R')}</div><div class="muted" style="margin-top:8px">Score: ${r.funnel_score??'—'} · Positieve pairs: ${r.positive_pairs??'—'}/${r.pair_count??'—'} · Exit: ${exitLabel(p)} · Params: ${Object.entries(p).filter(([k])=>k!=='_phase').map(([k,v])=>k+'='+v).join(' · ')}</div></div>`}).join('')}function stageLabel(k){let x={promoted:'Promoted',precision_deep_search:'Deep search',deep_search:'Deep search',incubator:'Incubator',discovery:'Discovery',precision_discovery:'Discovery',continuing:'Continuing',rejected:'Rejected',overig:'Overig'};return x[k]||String(k||'Overig').replaceAll('_',' ')}function P(rs){if(!rs?.length)return'<div class="phase-empty">Nog geen resultaten om over de fases te verdelen.</div>';let g={};for(let r of rs){let k=r.status||r.funnel_stage||r.phase||'overig';(g[k]??=[]).push(r)}let order=['promoted','precision_deep_search','deep_search','incubator','discovery','precision_discovery','continuing','rejected','overig'];let keys=[...new Set([...order,...Object.keys(g)])].filter(k=>g[k]?.length);return keys.map(k=>{let rows=g[k],freq={};for(let r of rows){let n=r.strategy||r.candidate?.strategy||'—';freq[n]=(freq[n]||0)+1}let tags=Object.entries(freq).sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])).map(([n,v])=>'<span class="tag">'+n+(v>1?' ×'+v:'')+'</span>').join('');return '<div class="phase '+k+'"><div class="phase-top"><div><h4>'+stageLabel(k)+'</h4><div class="sub">'+Object.keys(freq).length+' strategietype'+(Object.keys(freq).length===1?'':'n')+'</div></div><div class="count">'+rows.length+'</div></div><div class="names">'+tags+'</div></div>'}).join('')}function T(rs){if(!rs?.length)return'<p class="muted">Nog geen opgeslagen resultaten.</p>';let h='<table><tr><th>Strategie</th><th>Stage</th><th>Trades</th><th>Trades/dag</th><th>PF</th><th>Expectancy</th></tr>';for(let r of rs.slice(0,100)){let m=r.oos||r.metrics||{};h+=`<tr><td>${r.strategy||r.candidate?.strategy||'—'}</td><td>${r.status||r.funnel_stage||r.phase||'—'}</td><td>${m.trades??r.trade_count??'—'}</td><td>${Number(m.avg_trades_per_day||0).toFixed(1)}</td><td>${Number(m.profit_factor||0).toFixed(2)}</td><td>${Number(m.expectancy_r||0).toFixed(2)}R</td></tr>`}return h+'</table>'}async function loadResearch(){try{let d=await req('/api/research/results'),s=d.state||{},co=d.coordinator||{},ag=d.agent||{},val=d.validation||{},run=!!s.running,cl=run?'run':s.stage==='error'?'bad':'';$('research-status').innerHTML=`<span class="pill ${cl}">${run?'RUNNING':String(s.stage||'IDLE').toUpperCase()}</span><span class="pill">${String(co.mode||'COORDINATOR').toUpperCase()}</span><span class="pill">${ag.running?'AGENT ON':'AGENT OFF'}</span>`;$('research-message').textContent=s.message||co.message||ag.message||'';$('research-metrics').innerHTML=M('Getest',s.tested_total??0)+M('Promoted',s.promoted_total??0)+M('Kandidaat',s.current_candidate||'—')+M('TF',(s.current_params?.timeframe_min??'—')+'m')+M('Validaties',co.validations_completed??0)+M('17-staps lab',val.running?(val.current_lab||'RUNNING'):(val.completed_at?'COMPLETED':'IDLE'));let rows=d.results||[];$('research-phases').innerHTML=P(rows);$('overview-research-phases').innerHTML=P(rows);let h='<table><tr><th>Strategie</th><th>TF</th><th>Stage</th><th>Trades/dag</th><th>PF</th><th>Expectancy</th></tr>';for(let r of rows.slice(0,50)){let m=r.oos||{},p=r.params||{};h+=`<tr><td>${r.strategy||'—'}</td><td>${p.timeframe_min||'—'}m</td><td>${r.funnel_stage||'—'}</td><td>${Number(m.avg_trades_per_day||0).toFixed(1)}</td><td>${Number(m.profit_factor||0).toFixed(2)}</td><td>${Number(m.expectancy_bps||0).toFixed(2)} bps</td></tr>`}$('research-results').innerHTML=h+'</table>';$('rmini').className='pill '+cl;$('rmini').textContent='Research '+(run?'RUNNING':String(s.stage||'IDLE').toUpperCase())}catch(e){$('research-message').textContent=e.message}}async function researchAct(a){try{await req('/api/research/'+a,{method:'POST'});await loadResearch()}catch(e){$('research-message').textContent=e.message}}async function load(n){try{let d=await req('/api/'+n+'/results'),s=d.state||{},run=s.running,cl=run?'run':s.stage==='error'?'bad':'';$(n+'-status').innerHTML=`<span class="pill ${cl}">${run?'RUNNING':(s.stage||'IDLE').toUpperCase()}</span>`;$(n+'-message').textContent=s.message||'';$(n+'-progress').style.width=(s.total?Math.round((s.progress||0)*100/s.total):0)+'%';$(n+'-metrics').innerHTML=M('Getest',s.tested_total)+M('Voortgang',(s.progress||0)+' / '+(s.total||0))+M('Strategie',s.current_candidate||'—')+M('Pair',s.current_pair||'—')+M('Opgeslagen',s.results_loaded??d.results?.length??0)+M('Laatste',s.last_completed_candidate||'—');if(n==='forex-lab')$('forex-lab-promoted').innerHTML=D(d.results);$(n+'-phases').innerHTML=P(d.results);if(n==='forex-lab')$('overview-forex-phases').innerHTML=P(d.results);if(n==='precision-lab')$('overview-precision-phases').innerHTML=P(d.results);$(n+'-results').innerHTML=T(d.results);let id=n==='forex-lab'?'fmini':'pmini';$(id).className='pill '+cl;$(id).textContent=(n==='forex-lab'?'Forex ':'Precision ')+(run?'RUNNING':(s.stage||'IDLE').toUpperCase())}catch(e){$(n+'-message').textContent=e.message}}function money(x){return '€'+Number(x||0).toFixed(2)}function exitLabel(p){let m=String((p||{}).exit_mode||'baseline');return m==='protect_2r_025r'?'2R → +0.25R':m==='breakeven_2r'?'2R → BE':m==='lock_2r_05r'?'2R → +0.50R':'Baseline'}function stratLabel(r){return (r.strategy||'—')+' · '+exitLabel(r.params)}function paperStrategyCards(rows){if(!rows?.length)return'<p class="muted">Nog geen promoted strategieën in paper trading.</p>';return rows.map(r=>{let p=r.params||{},tr=Number(r.trades||0),wr=tr?100*Number(r.wins||0)/tr:0;return `<div class="card" style="margin:12px 0;padding:14px;background:#10161f"><div class="row" style="justify-content:space-between"><b>${stratLabel(r)}</b><span class="pill ${r.status==='active'?'run':r.status==='ruined'?'bad':''}">${String(r.status||'').toUpperCase()}</span></div><div class="grid">${M('Start',money(r.start_balance))}${M('Balans',money(r.balance))}${M('P/L',money(r.net_pnl))}${M('Trades',tr)}${M('Winrate',wr.toFixed(1)+'%')}${M('Open',r.open_positions??0)}${M('TF',(r.timeframe_min||p.timeframe_min)+'m')}${M('Risk',money(p.risk_eur))}${M('Target',(p.target_r??'—')+'R')}${M('Stop ATR',p.stop_atr??'—')}${M('Exit',exitLabel(p))}</div><div class="muted" style="margin-top:8px">Sinds ${String(r.started_at||'').replace('T',' ').slice(0,16)} · ${r.last_error?'Fout: '+r.last_error:'laatste cyclus '+String(r.last_cycle_at||'—').replace('T',' ').slice(0,16)}</div></div>`}).join('')}function dailyTable(rows){if(!rows?.length)return'<p class="muted">Nog geen dagen geregistreerd.</p>';let h='<table><tr><th>Datum</th><th>Strategie</th><th>Start</th><th>P/L</th><th>Eind</th><th>Trades</th><th>W/L</th><th>Open</th></tr>';for(let r of rows){h+=`<tr><td>${r.trade_date}</td><td>${stratLabel(r)}</td><td>${money(r.start_balance)}</td><td>${money(r.realized_pnl)}</td><td>${money(r.end_balance)}</td><td>${r.trade_count}</td><td>${r.wins}/${r.losses}</td><td>${r.open_positions}</td></tr>`}return h+'</table>'}function posTable(rows){if(!rows?.length)return'<p class="muted">Geen open posities.</p>';let h='<table><tr><th>Strategie</th><th>Pair</th><th>Side</th><th>Entry</th><th>Huidig</th><th>Open R</th><th>Open P/L</th><th>Stop</th><th>Target</th></tr>';for(let r of rows){let ur=Number(r.unrealized_r||0),up=Number(r.unrealized_pnl||0),cl=up>=0?'pnlpos':'pnlneg';h+=`<tr><td>${stratLabel(r)}</td><td>${r.pair}</td><td>${Number(r.direction)>0?'LONG':'SHORT'}</td><td>${Number(r.entry_price).toFixed(5)}</td><td>${Number(r.current_price||r.entry_price).toFixed(5)}</td><td class="${cl}">${ur.toFixed(2)}R</td><td class="${cl}">${money(up)}</td><td>${Number(r.stop_price).toFixed(5)}</td><td>${Number(r.target_price).toFixed(5)}</td></tr>`}return h+'</table>'}function tradeTable(rows){if(!rows?.length)return'<p class="muted">Nog geen gesloten paper trades.</p>';let h='<table><tr><th>Exit</th><th>Strategie</th><th>Pair</th><th>Side</th><th>Reden</th><th>R</th><th>P/L</th><th>Balans</th></tr>';for(let r of rows){h+=`<tr><td>${String(r.exit_time).replace('T',' ').slice(0,16)}</td><td>${stratLabel(r)}</td><td>${r.pair}</td><td>${r.side}</td><td>${r.exit_reason}</td><td>${Number(r.r_multiple).toFixed(2)}R</td><td>${money(r.pnl)}</td><td>${money(r.balance_after)}</td></tr>`}return h+'</table>'}async function loadPaper(){try{let d=await req('/api/paper/results'),st=d.state||{},run=st.running;let cl=run?'run':st.stage==='error'?'bad':'',eligible=(d.strategies||[]).filter(r=>['active','ruined'].includes(String(r.status||''))),ids=new Set(eligible.map(r=>r.paper_id));$('paper-status').innerHTML=`<span class="pill ${cl}">${run?'RUNNING':String(st.stage||'IDLE').toUpperCase()}</span><span class="pill ${st.portfolio_frequency_ready?'good':''}">Portfolio ${Number(st.expected_portfolio_trades_per_day||0).toFixed(1)}/${Number(st.portfolio_target_trades_per_day||10).toFixed(0)} trades/dag</span>`;$('paper-message').textContent=st.message||'';$('paper-strategies').innerHTML=paperStrategyCards(eligible);$('paper-daily').innerHTML=dailyTable((d.daily||[]).filter(r=>ids.has(r.paper_id)));$('paper-positions').innerHTML=posTable((d.positions||[]).filter(r=>ids.has(r.paper_id)));$('paper-trades').innerHTML=tradeTable((d.trades||[]).filter(r=>ids.has(r.paper_id)));$('papermini').className='pill '+cl;$('papermini').textContent='Paper '+(run?'RUNNING':String(st.stage||'IDLE').toUpperCase())}catch(e){$('paper-message').textContent=e.message}}async function paperAct(a){try{await req('/api/paper/'+a,{method:'POST'});await loadPaper()}catch(e){$('paper-message').textContent=e.message}}async function connect(){try{let h=await fetch('/health').then(r=>r.json());$('connection').textContent=`Online · live trading UIT · cTrader ${h.ctrader_ready?'ready':'niet ready'}`;await Promise.all([loadResearch(),load('forex-lab'),load('precision-lab'),loadPaper()])}catch(e){$('connection').textContent=e.message}}async function act(n,a){try{await req('/api/'+n+'/'+a,{method:'POST'});await load(n)}catch(e){$(n+'-message').textContent=e.message}}async function authorize(){try{let d=await req('/api/ctrader/oauth-url');location.href=d.url}catch(e){$('connection').textContent=e.message}}if(tok)connect();setInterval(()=>{if(tok){loadResearch();load('forex-lab');load('precision-lab');loadPaper()}},15000)</script></body></html>'''
