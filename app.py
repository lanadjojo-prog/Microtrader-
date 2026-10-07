from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from config import settings
from ctrader_client import CTraderClient, CTraderError
from strategy_lab import RESEARCH_POLICY_VERSION, StrategyLab
from research_labs import ResearchLabs
from research_agent import ResearchAgent
from research_coordinator import ResearchCoordinator
from forex_lab import ForexStrategyLab
from forex_backtest import FOREX_EVALUATION_POLICY_VERSION
from precision_lab import PrecisionStrategyLab
from precision_backtest import PRECISION_EVALUATION_POLICY_VERSION
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
mark_ctrader=CTraderClient(settings)
strategy_lab=StrategyLab(settings,research_ctrader)
research_labs=ResearchLabs(settings,validation_ctrader)
research_agent=ResearchAgent(settings,strategy_lab)
research_coordinator=ResearchCoordinator(strategy_lab,research_labs)
forex_lab=ForexStrategyLab(settings,forex_ctrader)
precision_lab=PrecisionStrategyLab(settings,precision_ctrader)
paper_engine=PaperTradingEngine(settings,paper_ctrader)
token_store=CTraderTokenStore(settings.database_url)

def runtime_invariant_errors() -> list[str]:
    errors = []
    if settings.live_trading_enabled:
        errors.append("LIVE_TRADING_ENABLED must remain false")
    if settings.ctrader_environment != "demo":
        errors.append("CTRADER_ENVIRONMENT must remain demo")
    if not settings.ctrader_demo_only:
        errors.append("CTRADER_DEMO_ONLY must remain true")
    if settings.ctrader_oauth_scope != "accounts":
        errors.append("CTRADER_OAUTH_SCOPE must remain accounts/read-only")
    if not (0 < settings.strategy_min_trades_per_day <= settings.strategy_preferred_trades_per_day <= settings.strategy_target_trades_per_day):
        errors.append("frequency policy must satisfy hard_min <= preferred <= target")
    if settings.forex_max_bars_per_pair < 20000:
        errors.append("FOREX_MAX_BARS_PER_PAIR is too small for the frozen-holdout policy")
    if not settings.forex_first:
        errors.append("FOREX_FIRST must remain true")
    if settings.forex_data_provider != "ctrader":
        errors.append("FOREX_DATA_PROVIDER must remain ctrader for the active runtime")
    if settings.precision_data_provider != "ctrader":
        errors.append("PRECISION_DATA_PROVIDER must remain ctrader for the active runtime")
    if settings.crypto_lab_auto_start:
        errors.append("CRYPTO_LAB_AUTO_START must remain false in forex-only mode")
    return errors


def assert_runtime_invariants() -> None:
    errors = runtime_invariant_errors()
    if errors:
        raise RuntimeError("Unsafe/inconsistent runtime configuration: " + "; ".join(errors))

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
    mark_ctrader.set_runtime_tokens(access,refresh)
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
            mark_ctrader.set_runtime_tokens(access,refresh)
            await ctrader.connect_and_authenticate()
        except Exception:
            logging.getLogger("microtrader").exception("Could not restore cTrader session")
            return False
    return True

async def autostart():
    if settings.lab_auto_start: await strategy_lab.start()
    if settings.research_agent_auto_start: await research_agent.start()
    if settings.research_auto_start: await research_coordinator.start()
    # Legacy Forex/Precision labs are retained for audit endpoints only.
    # The active research path is now StrategyLab → Distillation → Validation → Paper.
    if forex_lab.state.running:
        await forex_lab.stop()
    if precision_lab.state.running:
        await precision_lab.stop()
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
    assert_runtime_invariants()
    await restore_ctrader()
    await autostart(); yield
    await paper_engine.stop(); await precision_lab.stop(); await forex_lab.stop()
    await research_coordinator.stop(); await research_agent.stop(); await research_labs.stop(); await strategy_lab.stop()
    await mark_ctrader.close(); await paper_ctrader.close(); await precision_ctrader.close(); await forex_ctrader.close()
    await validation_ctrader.close(); await research_ctrader.close(); await ctrader.close()

app=FastAPI(title="ForexTrader Research",lifespan=lifespan)
def auth(a):
    if not settings.dashboard_token or a!=f"Bearer {settings.dashboard_token}": raise HTTPException(401,"Unauthorized")

@app.get('/health')
async def health(): return {'ok':not runtime_invariant_errors(),'mode':'forex-research-and-paper-only','runtime_invariant_errors':runtime_invariant_errors(),'live_trading_enabled':False,'research_market':'forex','research_data_source':'ctrader','research_running':strategy_lab.state.running,'research_stage':strategy_lab.state.stage,'research_validation_running':research_labs.state.running,'research_coordinator_mode':research_coordinator.state.mode,'research_agent_running':research_agent.state.running,'research_agent_stage':research_agent.state.stage,'forex_lab_running':forex_lab.state.running,'forex_stage':forex_lab.state.stage,'precision_lab_running':precision_lab.state.running,'precision_stage':precision_lab.state.stage,'paper_running':paper_engine.state.running,'paper_stage':paper_engine.state.stage,'ctrader_ready':ctrader.api_ready}
@app.get('/api/research/status')
async def research_status(authorization:str|None=Header(None)): auth(authorization); return {'strategy':strategy_lab.public_state(),'validation':research_labs.public_state(),'coordinator':research_coordinator.public_state(),'agent':research_agent.public_state()}
@app.get('/api/research/results')
async def research_results(authorization:str|None=Header(None)):
    auth(authorization)
    summary = await strategy_lab.store.funnel_summary(RESEARCH_POLICY_VERSION)
    return {'state':strategy_lab.public_state(),'results':strategy_lab.results(),'funnel_summary':summary,'validation':research_labs.public_state(),'coordinator':research_coordinator.public_state(),'agent':research_agent.public_state()}
@app.post('/api/research/start')
async def research_start(authorization:str|None=Header(None)): auth(authorization); await strategy_lab.start(); await research_agent.start(); await research_coordinator.start(); return {'strategy':strategy_lab.public_state(),'coordinator':research_coordinator.public_state(),'agent':research_agent.public_state()}
@app.post('/api/research/stop')
async def research_stop(authorization:str|None=Header(None)): auth(authorization); await research_coordinator.stop(); await research_agent.stop(); await research_labs.stop(); await strategy_lab.stop(); return {'strategy':strategy_lab.public_state(),'coordinator':research_coordinator.public_state(),'agent':research_agent.public_state()}
@app.get('/api/forex-lab/status')
async def fs(authorization:str|None=Header(None)): auth(authorization); return forex_lab.public_state()
@app.get('/api/forex-lab/results')
async def fr(authorization:str|None=Header(None)):
    auth(authorization)
    summary = await forex_lab.store.funnel_summary(FOREX_EVALUATION_POLICY_VERSION)
    return {'state':forex_lab.public_state(),'results':forex_lab.results(),'funnel_summary':summary}
@app.post('/api/forex-lab/start')
async def fst(authorization:str|None=Header(None)): auth(authorization); await forex_lab.start(); return forex_lab.public_state()
@app.post('/api/forex-lab/stop')
async def fsp(authorization:str|None=Header(None)): auth(authorization); await forex_lab.stop(); return forex_lab.public_state()
@app.get('/api/precision-lab/status')
async def ps(authorization:str|None=Header(None)): auth(authorization); return precision_lab.public_state()
@app.get('/api/precision-lab/results')
async def pr(authorization:str|None=Header(None)):
    auth(authorization)
    summary = await precision_lab.store.funnel_summary(PRECISION_EVALUATION_POLICY_VERSION)
    return {'state':precision_lab.public_state(),'results':precision_lab.results(),'funnel_summary':summary}
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
                bars = await mark_ctrader.historical_bars(
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
        mark_ctrader.set_runtime_tokens(ctrader.active_access_token, ctrader.active_refresh_token)
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

DASHBOARD=r'''<!doctype html>
<html lang="nl">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ForexTrader Research</title>
<style>
*{box-sizing:border-box}body{font-family:system-ui,-apple-system,sans-serif;background:#0d1117;color:#edf2f7;margin:0;max-width:980px;padding:18px;margin:auto}
h1,h2,h3{margin:0}h1{font-size:1.55rem}h2{font-size:1.12rem}.muted{color:#92a0b1}.top,.row{display:flex;gap:9px;align-items:center;flex-wrap:wrap}.top{justify-content:space-between;margin-bottom:14px}
.card{background:#161c24;border:1px solid #293443;border-radius:15px;padding:16px;margin:12px 0}.pill{font-size:.78rem;padding:5px 9px;border-radius:999px;background:#242e3a}.run{background:#153552;color:#abd9ff}.good{background:#153820;color:#9ce7ac}.bad{background:#411e25;color:#ffb0b7}
button,input{font:inherit;border:1px solid #39485b;border-radius:9px;padding:9px 11px;background:#10161e;color:#fff}button{cursor:pointer}
.pipeline{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:14px}.stage{background:#101720;border:1px solid #283545;border-radius:12px;padding:12px;min-height:110px;position:relative}.stage.active{border-color:#6aa6db}.stage.active:after{content:"";position:absolute;right:10px;top:10px;width:8px;height:8px;border-radius:50%;background:#7fc2ff;animation:pulse 1.1s infinite}.stage.done{border-color:#376443}.stage b{display:block;font-size:.88rem}.stage .n{font-size:1.5rem;font-weight:800;margin:5px 0}.stage small{color:#92a0b1;line-height:1.35}@keyframes pulse{0%,100%{opacity:.35;transform:scale(.85)}50%{opacity:1;transform:scale(1.2)}}
.now{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:8px;margin-top:12px}.metric{background:#101720;border-radius:10px;padding:10px}.metric span{display:block;color:#92a0b1;font-size:.73rem}.metric b{display:block;margin-top:4px;font-size:1.03rem;overflow-wrap:anywhere}
.criteria{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:8px;margin-top:12px}.criterion{padding:10px;background:#101720;border-radius:10px;border:1px solid #263342;font-size:.84rem}.criterion b{display:block;margin-bottom:3px}
.routes{margin-top:12px;overflow:auto}.routes table,.results table{width:100%;border-collapse:collapse;font-size:.81rem;min-width:650px}th,td{text-align:left;padding:8px;border-bottom:1px solid #293443}th{color:#92a0b1}.results{overflow:auto;margin-top:10px;max-height:380px;border:1px solid #263342;border-radius:10px}
.paper-tabs{display:flex;gap:7px;overflow-x:auto;margin-top:12px;padding-bottom:5px}.paper-tab{white-space:nowrap}.paper-tab.active{background:#263b54;border-color:#6586aa}.paper-pane{margin-top:10px;padding:13px;background:#101720;border:1px solid #263342;border-radius:12px}.paper-title{font-size:1.05rem;font-weight:800}.paper-sub{margin-top:3px;color:#92a0b1}.paper-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(125px,1fr));gap:7px;margin-top:11px}.paper-metric{background:#151e29;border-radius:9px;padding:9px}.paper-metric span{display:block;color:#92a0b1;font-size:.7rem}.paper-metric b{display:block;margin-top:3px}.pos{color:#9ce7ac}.neg{color:#ffb0b7}.empty{padding:11px;color:#92a0b1;border:1px dashed #334154;border-radius:9px;margin-top:9px}.refresh{font-size:.75rem;color:#92a0b1;margin-top:7px}
@media(max-width:650px){body{padding:11px}.pipeline{grid-template-columns:repeat(2,1fr)}.card{padding:13px}.paper-grid,.now{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div class="top"><div><h1>ForexTrader</h1><div class="muted">Research → simpele regels → validation → paper</div></div><div class="row"><span id="connection" class="pill">offline</span><button onclick="authorize()">cTrader</button></div></div>
<div id="authbox" class="card"><b>Dashboard toegang</b><div class="row" style="margin-top:9px"><input id="token" type="password" placeholder="Dashboard token"><button onclick="saveToken()">Open</button></div></div>
<div id="main" style="display:none">

<div class="card">
<div class="row" style="justify-content:space-between"><div><h2>Onderzoekspijplijn</h2><div class="muted">Je ziet alleen de hoofdfases; interne onderzoekslagen blijven op de achtergrond.</div></div><span id="research-status" class="pill">—</span></div>
<div class="pipeline">
<div id="stage-discovery" class="stage"><b>1 · Discovery</b><div id="n-discovery" class="n">0</div><small>Breed zoeken naar edge en wanneer entries werken.</small></div>
<div id="stage-distill" class="stage"><b>2 · Distillation</b><div id="n-distill" class="n">0</div><small>Tijd, volume, context, entry, management en exit terugbrengen tot simpele regels.</small></div>
<div id="stage-validation" class="stage"><b>3 · Validation</b><div id="n-validation" class="n">0</div><small>Bevroren regel: walk-forward, parameterstabiliteit en hogere kosten.</small></div>
<div id="stage-paper" class="stage"><b>4 · Paper</b><div id="n-paper" class="n">0</div><small>Forward-only. Nieuwe inzichten veranderen een lopende versie niet.</small></div>
</div>
<div class="now" id="research-now"></div>
<div id="research-message" class="refresh"></div>
<div id="last-refresh" class="refresh"></div>
</div>

<section class="card"><h3>Kritische onderzoekscontrole</h3><p id="critic-message"></p><div id="critic-hypotheses" style="white-space:pre-line"></div><p class="muted">Hypotheses op onderzoeksdata. Paper trading vereist aparte validatie.</p></section>
<div class="card">
<h2>Criteria</h2>
<div class="criteria">
<div class="criterion"><b>Frequentie</b><span id="criterion-frequency">≥3 trades/dag per strategie; voorkeur 5–10.</span></div>
<div class="criterion"><b>Contextbewijs</b>Minimaal 30 trades voordat een context-specialist inzetbaar wordt.</div>
<div class="criterion"><b>Validation</b>PF ≥1,15 + positieve expectancy, ≥3/4 positieve walk-forwardvensters, ≥50% stabiele buurparameters en positief bij 1,5× kosten.</div>
<div class="criterion"><b>Eenvoud</b>Maximaal 2 specialists per context. Geen bewezen route of exit = geen trade.</div>
<div class="criterion"><b>Management</b>Baseline, break-even/profit-lock varianten worden apart vergeleken met MFE, MAE en giveback.</div>
<div class="criterion"><b>Holdout & forward</b>Finale holdout wordt niet teruggeleerd in de router. Paper is daarna de echte nieuwe forward test.</div>
</div>
</div>

<div class="card">
<div class="row" style="justify-content:space-between"><div><h2>Uitgekristalliseerde regels</h2><div class="muted">Alleen combinaties waarvoor voldoende bewijs is gevonden.</div></div><span id="adaptive-status" class="pill">LEREN</span></div>
<div id="route-summary" class="now"></div>
<div id="routes" class="routes"></div>
</div>

<div class="card">
<div class="row" style="justify-content:space-between"><div><h2>Paper Trading</h2><div class="muted">Volledige forward-data inclusief trade management en giveback.</div></div><span id="paper-status" class="pill">—</span></div>
<div id="paper-message" class="refresh"></div>
<div id="paper-tabs" class="paper-tabs"></div>
<div id="paper-content"></div>
</div>
</div>

<script>
const $=id=>document.getElementById(id);let tok=localStorage.getItem('mt_token')||'';let activePaper=localStorage.getItem('paperTab')||'';
function saveToken(){tok=$('token').value.trim();localStorage.setItem('mt_token',tok);connect()}
async function req(url,opt={}){opt.headers={...(opt.headers||{}),Authorization:'Bearer '+tok};let r=await fetch(url,opt);if(!r.ok)throw Error(await r.text());return r.json()}
function metric(k,v){return '<div class="metric"><span>'+k+'</span><b>'+String(v??'—')+'</b></div>'}
function pm(k,v,cl=''){return '<div class="paper-metric"><span>'+k+'</span><b class="'+cl+'">'+String(v??'—')+'</b></div>'}
function money(v){return '€'+Number(v||0).toFixed(2)}
function pretty(v){return String(v||'—').replaceAll('_',' ').replace(/\b\w/g,m=>m.toUpperCase())}
function setStage(id,on,done=false){let e=$(id);e.classList.toggle('active',!!on);e.classList.toggle('done',!!done)}
function stageCounts(rows){let out={discovery:0,distill:0,validation:0,promoted:0,rejected:0};for(let r of rows||[]){let s=String(r.funnel_stage||'');if(s==='incubator')out.distill++;else if(s==='deep_search')out.validation++;else if(s==='promoted'||r.promoted)out.promoted++;else if(s==='rejected')out.rejected++;else out.discovery++}return out}
function managementLabel(p){p=p||{};let n=String(p.name||'baseline');if(n==='baseline')return'Baseline';let t=p.management_trigger_r,l=p.management_lock_net_r;if(t!=null&&l!=null)return'+'+t+'R → lock +'+l+'R';return pretty(n)}
function routeTable(dec){let tfs=(dec?.adaptive_policy||{}).timeframes||{},rows=[];for(let [tf,pol] of Object.entries(tfs)){for(let [ctx,routes] of Object.entries(pol.routes||{})){for(let r of routes||[]){let ex=r.exit_profile||{},cond=Object.entries(r.conditions||{}).map(([k,v])=>pretty(k)+': '+pretty(v)).join(' · ')||'Geen extra filter';rows.push([tf,ctx,r.entry_model||r.source_strategy,cond,managementLabel(r.management_profile),ex.name||'—',r.source_trades||'—',Number(r.source_profit_factor||0).toFixed(2)])}}}if(!rows.length)return'<div class="empty">Nog geen complete regel heeft voldoende bewijs. De correcte actie is dan: geen trade.</div>';return'<table><tr><th>TF</th><th>Context</th><th>Entry</th><th>Extra voorwaarde</th><th>Management</th><th>Exit</th><th>Evidence</th><th>PF</th></tr>'+rows.map(r=>'<tr><td>'+r[0]+'m</td><td>'+r[1]+'</td><td>'+pretty(r[2])+'</td><td>'+r[3]+'</td><td>'+r[4]+'</td><td>'+pretty(r[5])+'</td><td>'+r[6]+'</td><td>'+r[7]+'</td></tr>').join('')+'</table>'}
async function loadResearch(){let d=await req('/api/research/status'),s=d.strategy||{},ag=d.agent||{},co=d.coordinator||{},val=d.validation||{},rows=[],counts=stageCounts(rows),dec=ag.last_decision||{},aps=dec.adaptive_policy_summary||{};$('research-status').className='pill '+(s.running?'run':s.stage==='error'?'bad':'');$('research-status').textContent=s.running?'ACTIEF':String(s.stage||'IDLE').toUpperCase();$('n-discovery').textContent=s.tested_total??0;$('n-distill').textContent=(aps.specialists??0)+' specialists';$('n-validation').textContent=co.validations_completed??0;$('n-paper').textContent=counts.promoted;let phase=String(s.current_params?._phase||'discovery');setStage('stage-discovery',!!s.running&&phase==='discovery');setStage('stage-distill',!!s.running&&phase==='incubator',Number(aps.specialists||0)>0);setStage('stage-validation',String(co.mode||'')==='validation',Number(co.validations_completed||0)>0);$('research-now').innerHTML=metric('Nu getest',s.current_candidate||'—')+metric('Timeframe',(s.current_params?.timeframe_min??'—')+'m')+metric('Pair',s.current_symbol||'—')+metric('Pair voortgang',(s.current_symbol_index||0)+' / '+(s.current_symbol_total||0))+metric('Generatie',s.generation??'—')+metric('Laatste klaar',s.last_completed_candidate||'—')+metric('Contexts',aps.contexts??0)+metric('Evidence rows',aps.evidence_rows??0)+metric('Validation check',String(co.mode||'')==='validation'?(pretty(val.current_lab||'start')+' '+Number(val.completed_labs||0)+'/'+Number(val.total_labs||4)):'—');$('research-message').textContent=s.message||ag.message||'';$('route-summary').innerHTML=metric('Contexts',aps.contexts??0)+metric('Specialists',aps.specialists??0)+metric('Evidence rows',aps.evidence_rows??0)+metric('Timeframes',(aps.timeframes||[]).map(x=>x+'m').join(' · ')||'—');$('adaptive-status').className='pill '+(Number(aps.specialists||0)>0?'good':'');$('adaptive-status').textContent=Number(aps.specialists||0)>0?'DISTILLING':'LEREN';$('routes').innerHTML=routeTable(dec);let cr=dec.critic||{};$('critic-message').textContent=cr.message||'Wacht op eerste kritische analyse';$('critic-hypotheses').textContent=(cr.hypotheses||[]).map(h=>h.thesis+' '+h.rationale+' '+h.falsification).join('\n\n');$('last-refresh').textContent='Laatste beweging: '+new Date().toLocaleTimeString('nl-NL',{hour:'2-digit',minute:'2-digit',second:'2-digit'});return d}
function codeName(r){let x=String(r.paper_id||'paper-000').replace('paper-','');return pretty(r.strategy)+' · '+x.slice(-5).toUpperCase()}
function paperHeader(s,trades){let own=(trades||[]).filter(t=>t.paper_id===s.paper_id),mfe=own.length?own.reduce((a,x)=>a+Number(x.max_favorable_r||0),0)/own.length:0,mae=own.length?own.reduce((a,x)=>a+Number(x.max_adverse_r||0),0)/own.length:0,give=own.length?own.reduce((a,x)=>a+Number(x.giveback_r||0),0)/own.length:0,tr=Number(s.trades||0),wr=tr?100*Number(s.wins||0)/tr:0;return'<div class="row" style="justify-content:space-between"><div><div class="paper-title">'+codeName(s)+'</div><div class="paper-sub">'+pretty(s.strategy)+' · '+(s.timeframe_min??'—')+'m</div></div><span class="pill '+(s.status==='active'?'run':s.status==='review_pause'?'bad':'')+'">'+pretty(s.status)+'</span></div><div class="paper-grid">'+pm('Balans',money(s.balance))+pm('P/L',money(s.net_pnl),Number(s.net_pnl)>=0?'pos':'neg')+pm('Trades',tr)+pm('Winrate',wr.toFixed(1)+'%')+pm('Open',s.open_positions??0)+pm('Max loss streak',s.paper_max_loss_streak??'—')+pm('Gem. MFE',mfe.toFixed(2)+'R')+pm('Gem. MAE',mae.toFixed(2)+'R')+pm('Gem. giveback',give.toFixed(2)+'R')+'</div>'}
function openTable(s,rows){let a=(rows||[]).filter(x=>x.paper_id===s.paper_id);if(!a.length)return'<div class="empty">Geen open trades.</div>';return'<div class="results"><table><tr><th>Pair</th><th>Side</th><th>Open R</th><th>MFE</th><th>MAE</th><th>Management</th><th>Stop</th><th>Target</th></tr>'+a.map(x=>'<tr><td>'+x.pair+'</td><td>'+(Number(x.direction)>0?'LONG':'SHORT')+'</td><td>'+Number(x.unrealized_r||0).toFixed(2)+'R</td><td>'+Number(x.max_favorable_r||0).toFixed(2)+'R</td><td>'+Number(x.max_adverse_r||0).toFixed(2)+'R</td><td>'+managementLabel({name:x.management_model,management_trigger_r:x.management_trigger_r_override,management_lock_net_r:x.management_lock_net_r_override})+'</td><td>'+Number(x.stop_price||0).toFixed(5)+'</td><td>'+Number(x.target_price||0).toFixed(5)+'</td></tr>').join('')+'</table></div>'}
function closedTable(s,rows){let a=(rows||[]).filter(x=>x.paper_id===s.paper_id);if(!a.length)return'<div class="empty">Nog geen gesloten trades.</div>';return'<div class="results"><table><tr><th>Exit</th><th>Pair</th><th>Entry</th><th>Context</th><th>Management</th><th>Reden</th><th>MFE</th><th>MAE</th><th>Giveback</th><th>R</th><th>P/L</th></tr>'+a.map(x=>{let ctx=x.entry_context||{},r=Number(x.r_multiple||0),p=Number(x.pnl||0);return'<tr><td>'+String(x.exit_time||'').replace('T',' ').slice(0,16)+'</td><td>'+x.pair+'</td><td>'+pretty(x.entry_model)+'</td><td>'+String(ctx.regime||'—')+' / '+String(ctx.session||'—')+' / '+String(ctx.volume_bucket||'—')+'</td><td>'+pretty(x.management_model||'baseline')+'</td><td>'+pretty(x.exit_reason)+'</td><td>'+Number(x.max_favorable_r||0).toFixed(2)+'R</td><td>'+Number(x.max_adverse_r||0).toFixed(2)+'R</td><td>'+Number(x.giveback_r||0).toFixed(2)+'R</td><td class="'+(r>=0?'pos':'neg')+'">'+r.toFixed(2)+'R</td><td class="'+(p>=0?'pos':'neg')+'">'+money(p)+'</td></tr>'}).join('')+'</table></div>'}
function dailyTable(s,rows){let a=(rows||[]).filter(x=>x.paper_id===s.paper_id);if(!a.length)return'';return'<div class="results"><table><tr><th>Dag</th><th>P/L</th><th>Trades</th><th>W/L</th><th>Eindbalans</th></tr>'+a.map(x=>'<tr><td>'+x.trade_date+'</td><td class="'+(Number(x.realized_pnl)>=0?'pos':'neg')+'">'+money(x.realized_pnl)+'</td><td>'+x.trade_count+'</td><td>'+x.wins+'/'+x.losses+'</td><td>'+money(x.end_balance)+'</td></tr>').join('')+'</table></div>'}
function setPaper(id){activePaper=id;localStorage.setItem('paperTab',id);document.querySelectorAll('.paper-tab').forEach(x=>x.classList.toggle('active',x.dataset.id===id));document.querySelectorAll('.paper-wrap').forEach(x=>x.style.display=x.dataset.id===id?'block':'none')}
function renderPaper(strats,daily,pos,trades){if(!strats.length){$('paper-tabs').innerHTML='';$('paper-content').innerHTML='<div class="empty">Nog geen paperstrategie actief of in review.</div>';return}if(!strats.some(x=>x.paper_id===activePaper))activePaper=strats[0].paper_id;$('paper-tabs').innerHTML=strats.map(s=>'<button class="paper-tab '+(s.paper_id===activePaper?'active':'')+'" data-id="'+s.paper_id+'" onclick="setPaper(\''+s.paper_id+'\')">'+codeName(s)+'</button>').join('');$('paper-content').innerHTML=strats.map(s=>'<div class="paper-wrap" data-id="'+s.paper_id+'" style="display:'+(s.paper_id===activePaper?'block':'none')+'"><div class="paper-pane">'+paperHeader(s,trades)+'<h3 style="margin-top:15px">Open trades</h3>'+openTable(s,pos)+'<h3 style="margin-top:15px">Gesloten trades</h3>'+closedTable(s,trades)+'<h3 style="margin-top:15px">Per dag</h3>'+dailyTable(s,daily)+'</div></div>').join('')}
async function loadPaper(){let d=await req('/api/paper/results'),st=d.state||{},strats=(d.strategies||[]).filter(x=>['active','retiring','review_pause','ruined'].includes(String(x.status||''))),ids=new Set(strats.map(x=>x.paper_id)),daily=(d.daily||[]).filter(x=>ids.has(x.paper_id)),pos=(d.positions||[]).filter(x=>ids.has(x.paper_id)),trades=(d.trades||[]).filter(x=>ids.has(x.paper_id));$('paper-status').className='pill '+(st.running?'run':st.stage==='error'?'bad':'');$('paper-status').textContent=st.running?'ACTIEF':pretty(st.stage);$('paper-message').textContent=(st.message||'')+' · Verwacht '+Number(st.expected_portfolio_trades_per_day||0).toFixed(1)+' trades/dag';$('n-paper').textContent=strats.length;setStage('stage-paper',!!st.running,strats.length>0);$('criterion-frequency').textContent='≥'+Number(st.strategy_min_trades_per_day||3).toFixed(0)+' trades/dag per strategie; voorkeur '+Number(st.strategy_preferred_trades_per_day||5).toFixed(0)+'–'+Number(st.strategy_target_trades_per_day||10).toFixed(0)+'.';renderPaper(strats,daily,pos,trades)}
async function refresh(){await loadResearch();await loadPaper()}
async function connect(){try{$('authbox').style.display='none';$('main').style.display='block';let h=await fetch('/health').then(r=>r.json());$('connection').className='pill '+(h.ctrader_ready?'good':'bad');$('connection').textContent='cTrader '+(h.ctrader_ready?'READY':'NIET READY')+' · LIVE UIT';await refresh()}catch(e){$('authbox').style.display='block';$('main').style.display='none';$('connection').className='pill bad';$('connection').textContent='Geen toegang'}}
async function authorize(){try{let d=await req('/api/ctrader/oauth-url');location.href=d.url}catch(e){alert(e.message)}}
if(tok)connect();else $('authbox').style.display='block';setInterval(()=>{if(tok)refresh().catch(()=>{})},30000);
</script>
</body></html>'''
