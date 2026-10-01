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

DASHBOARD=r'''<!doctype html><html lang="nl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ForexTrader</title><style>
*{box-sizing:border-box}body{font-family:system-ui,-apple-system,sans-serif;background:#0e1116;color:#edf2f7;margin:0;padding:24px;max-width:1000px;margin:auto}h1{margin:0}.muted{color:#96a4b5}.top,.row{display:flex;gap:9px;flex-wrap:wrap;align-items:center}.top{justify-content:space-between;margin-bottom:18px}.card{background:#171c24;border:1px solid #2a3442;border-radius:16px;padding:18px;margin:14px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(135px,1fr));gap:9px;margin-top:12px}.metric{background:#111720;padding:11px;border-radius:11px}.metric b{display:block;font-size:1.1rem;margin-top:4px}.pill{padding:5px 9px;border-radius:999px;background:#242d39;font-size:.82rem}.good{background:#17351f;color:#9ee8ad}.run{background:#17314a;color:#a8d8ff}.bad{background:#3a1c22;color:#ffadb5}button,input{font:inherit;border-radius:10px;border:1px solid #3a4658;padding:10px 12px;background:#0f141b;color:#fff}button{cursor:pointer}.progress{height:9px;background:#0f141b;border-radius:99px;overflow:hidden;margin:12px 0}.progress div{height:100%;background:#e9eef5;width:0}.results{overflow:auto;margin-top:12px;max-height:420px;border:1px solid #263241;border-radius:10px}.results table{min-width:620px}table{width:100%;border-collapse:collapse;font-size:.84rem}th,td{text-align:left;padding:8px;border-bottom:1px solid #2a3442}th{color:#96a4b5;position:sticky;top:0;background:#171c24;z-index:2}.overview{position:static}.config{font-size:.9rem;line-height:1.5}.pipeline-head{display:flex;justify-content:space-between;align-items:end;gap:10px;margin-top:18px}.pipeline-head b{font-size:.92rem}.pipeline-head span{font-size:.76rem;color:#96a4b5}.phases{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:10px;margin-top:9px}.phase{position:relative;background:linear-gradient(180deg,#131a24,#10161f);border:1px solid #2b3746;border-radius:14px;padding:12px;min-height:128px;overflow:hidden}.phase:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:#4b5a6b}.phase.promoted:before{background:#62d985}.phase.deep_search:before,.phase.precision_deep_search:before{background:#72b8ff}.phase.incubator:before{background:#d9b962}.phase.discovery:before,.phase.precision_discovery:before{background:#9b8cff}.phase.rejected{opacity:.72}.phase.rejected:before{background:#d66b76}.phase-top{display:flex;justify-content:space-between;gap:8px;align-items:flex-start}.phase h4{margin:0;font-size:.82rem;text-transform:uppercase;letter-spacing:.04em}.phase .count{font-size:1.55rem;font-weight:800;line-height:1}.phase .sub{color:#96a4b5;font-size:.72rem;margin-top:4px}.phase .names{display:flex;flex-wrap:wrap;gap:5px;margin-top:10px;max-height:76px;overflow:auto}.phase .tag{font-size:.72rem;line-height:1.2;padding:5px 7px;border-radius:999px;background:#202a36;color:#cbd6e3;border:1px solid #2f3a48}.phase-empty{grid-column:1/-1;background:#111720;border:1px dashed #2e3a49;border-radius:12px;padding:14px;color:#96a4b5}.rules{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:9px;margin-top:10px}.rule{background:#111720;border:1px solid #263241;border-radius:12px;padding:11px}.rule b{display:block;margin-top:3px}details{margin-top:12px;border-top:1px solid #263241;padding-top:10px}summary{cursor:pointer;color:#b7c4d3;font-weight:650}.pnlpos{color:#9ee8ad}.pnlneg{color:#ffadb5}.paper-tabs{display:flex;gap:8px;overflow-x:auto;padding:3px 1px 8px;margin-top:14px;scrollbar-width:thin}.paper-tab{white-space:nowrap;background:#111720;border:1px solid #2f3a48;color:#aebccc;padding:9px 12px;border-radius:11px}.paper-tab.active{background:#243246;border-color:#5c7798;color:#fff;box-shadow:inset 0 0 0 1px #5c7798}.paper-pane{background:#10161f;border:1px solid #263241;border-radius:14px;padding:15px;margin-top:4px}.paper-pane h3{margin:18px 0 7px;font-size:.98rem}.paper-name{font-size:1.18rem;font-weight:800}.paper-family{margin-top:3px}.paper-params{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}.paper-param{font-size:.75rem;padding:5px 8px;border-radius:999px;background:#202a36;border:1px solid #2f3a48;color:#cbd6e3}.paper-subsection{margin-top:16px}.paper-empty{padding:12px;color:#96a4b5;background:#111720;border:1px dashed #2e3a49;border-radius:10px}@media(max-width:600px){body{padding:13px}.card{padding:13px;margin:10px 0}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}h1{font-size:1.5rem}.metric{padding:9px}button{padding:9px 10px}.results{max-height:320px}.phases{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body><div class="top"><div><h1>ForexTrader</h1><div class="muted">100% forex research via cTrader · geen stocks · geen live trading</div></div><span class="pill good">RESEARCH ONLY</span></div>
<div class="card"><div class="row"><input id="token" type="password" placeholder="Dashboard token"><button onclick="connect()">Connect</button><button onclick="authorize()">cTrader koppelen</button></div><div id="connection" class="muted" style="margin-top:8px">Voer je token in.</div></div>
<div class="card overview"><div class="row" style="justify-content:space-between"><div class="row"><b>Status</b><span id="rmini" class="pill">Research —</span><span id="amini" class="pill">Agent —</span><span id="fmini" class="pill">Forex —</span><span id="pmini" class="pill">Precision —</span><span id="papermini" class="pill">Paper —</span></div><span class="pill">DASHBOARD V10</span></div></div>
<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">Funnel overzicht</h2><span class="pill good">LIVE PER LAB</span></div><p class="muted config">Hier zie je direct hoeveel kandidaten in elke fase zitten en welke strategieën dat zijn.</p><h3>Research Lab</h3><div id="overview-research-phases" class="phases"><div class="phase-empty">Laden…</div></div><h3>Forex Lab</h3><div id="overview-forex-phases" class="phases"><div class="phase-empty">Laden…</div></div><h3>Precision Lab</h3><div id="overview-precision-phases" class="phases"><div class="phase-empty">Laden…</div></div></div>
<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">Actieve regels</h2><span class="pill good">HARD FILTERS</span></div><div class="rules"><div class="rule"><span class="muted">Timeframes</span><b>1m + 5m</b></div><div class="rule"><span class="muted">Richting</span><b>Long + short</b></div><div class="rule"><span class="muted">Per strategie</span><b>≥ 3 trades/dag</b></div><div class="rule"><span class="muted">Voorkeur</span><b>5–10 trades/dag</b></div><div class="rule"><span class="muted">Gecombineerd portfolio</span><b>doel ≥ 10 trades/dag</b></div><div class="rule"><span class="muted">Entry-sessies</span><b>London + New York</b></div><div class="rule"><span class="muted">Volume</span><b>≥ 0,70× normaal</b></div><div class="rule"><span class="muted">Startkapitaal</span><b>€50 paper</b></div><div class="rule"><span class="muted">Live geld</span><b>UIT</b></div><div class="rule"><span class="muted">Broker feasibility</span><b>NOG TE VALIDEREN</b></div></div><p class="muted config">Nieuwe entries alleen tijdens liquide London/New York-uren, DST-aware. Stops, targets en positiebeheer blijven ook buiten die uren actief.</p></div>
<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">Research Lab</h2><span class="pill">FOREX DISCOVERY + 17-STAGE VALIDATION</span></div><p class="config muted"><b>Forex-only via cTrader/Fusion · 1m/5m · harde ondergrens 3 trades/dag per strategie.</b> Geen stocks. Research zoekt brede FX-signaalhypotheses; kandidaten die Promoted zijn én de exacte 17/17-validatie afronden gaan automatisch naar forward Paper Trading met hun Research-native signaal/exitlogica.</p><div class="row"><button onclick="researchAct('start')">Start</button><button onclick="researchAct('stop')">Stop</button><button onclick="loadResearch()">Refresh</button></div><div id="research-status" class="row" style="margin-top:12px"></div><div id="research-message" class="muted"></div><div id="research-metrics" class="grid"></div><div class="pipeline-head"><b>Funnel per fase</b><span>hoeveel + welke strategieën</span></div><div id="research-phases" class="phases"></div><details><summary>Alle Research-resultaten</summary><div id="research-results" class="results"></div></details></div>
<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">Research Agent</h2><span id="agent-badge" class="pill">AGENT —</span></div><p class="config muted">De agent analyseert de forex-resultaten, kiest welke strategiefamilies en 1m/5m-timeframes extra aandacht krijgen en stuurt de volgende Research-batches. Hij kan geen orders plaatsen en live trading niet inschakelen.</p><div id="agent-message" class="muted"></div><div id="agent-metrics" class="grid"></div><div class="pipeline-head"><b>Huidige focus</b><span>wordt automatisch bijgewerkt</span></div><div id="agent-focus" class="row" style="margin-top:9px"></div><div class="pipeline-head"><b>Hypotheses</b><span>waar de agent nu op let</span></div><div id="agent-hypotheses" class="results"></div></div>
<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">Forex Lab</h2><span class="pill">BREDE FUNNEL</span></div><p class="config muted"><b>HARDE EIS: gemiddeld minimaal 3 trades per handelsdag per strategie.</b> 5–10 trades/dag is alleen een voorkeur/scorebonus. De ≥10 trades/dag is uitsluitend het gecombineerde portfolio-doel, niet een harde eis per strategie. Breakout · trend pullback · range reversal · exit-management varianten. Trade management wordt pas bij kansrijke Incubator-kandidaten getest: baseline, triggers op 1R/1,5R/2R en netto locks van +0,05R/+0,25R/+0,50R; een nieuwe beschermende stop wordt conservatief pas vanaf de volgende candle actief.</p><div class="row"><button onclick="act('forex-lab','start')">Start</button><button onclick="act('forex-lab','stop')">Stop</button><button onclick="load('forex-lab')">Refresh</button></div><div id="forex-lab-status" class="row" style="margin-top:12px"></div><div id="forex-lab-message" class="muted"></div><div class="progress"><div id="forex-lab-progress"></div></div><div id="forex-lab-metrics" class="grid"></div><div id="forex-lab-promoted"></div><div class="pipeline-head"><b>Funnel per fase</b><span>hoeveel + welke strategieën</span></div><div id="forex-lab-phases" class="phases"></div><details><summary>Alle Forex-testresultaten</summary><div id="forex-lab-results" class="results"></div></details></div>
<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">Paper Trading</h2><span class="pill good">€50 START · SIMULATED</span></div><p class="config muted">Elke paperstrategie heeft hieronder een eigen tabblad, vaste unieke referentienaam en geïsoleerde €50-paperledger. Daily statistieken, open posities en gesloten trades blijven volledig per strategie gescheiden. Geen brokerorders.</p><div class="row"><button onclick="paperAct('start')">Start</button><button onclick="paperAct('stop')">Stop</button><button onclick="loadPaper()">Refresh</button></div><div id="paper-status" class="row" style="margin-top:12px"></div><div id="paper-message" class="muted"></div><div id="paper-tabs" class="paper-tabs"></div><div id="paper-tab-content"></div></div>
<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">Precision Lab</h2><span class="pill">PIP FUNNEL</span></div><p class="config muted"><b>HARDE EIS: gemiddeld minimaal 3 trades per handelsdag per strategie.</b> Frequentievoorkeur 5–10/dag; gecombineerd portfolio-doel ≥10/dag. €50 researchmodel · €2/€3 risico · stops 2–5 pips · targets 4–10 pips · bid/ask tick execution. “Qualified” betekent geslaagd voor de huidige chronologische tick-validatie; Precision heeft geen nep-Deep-Search-fase.</p><div class="row"><button onclick="act('precision-lab','start')">Start</button><button onclick="act('precision-lab','stop')">Stop</button><button onclick="load('precision-lab')">Refresh</button></div><div id="precision-lab-status" class="row" style="margin-top:12px"></div><div id="precision-lab-message" class="muted"></div><div class="progress"><div id="precision-lab-progress"></div></div><div id="precision-lab-metrics" class="grid"></div><div class="pipeline-head"><b>Funnel per fase</b><span>hoeveel + welke strategieën</span></div><div id="precision-lab-phases" class="phases"></div><details><summary>Alle Precision-testresultaten</summary><div id="precision-lab-results" class="results"></div></details></div>
<script>const $=x=>document.getElementById(x);let tok=localStorage.getItem('ftToken')||'';$('token').value=tok;function H(){tok=$('token').value.trim();localStorage.setItem('ftToken',tok);return{'Authorization':'Bearer '+tok}}async function req(u,o={}){o.headers={...(o.headers||{}),...H()};let r=await fetch(u,o),j=await r.json().catch(()=>({}));if(!r.ok)throw Error(j.detail||'HTTP '+r.status);return j}function M(a,b){return`<div class="metric"><span class="muted">${a}</span><b>${b??'—'}</b></div>`}function D(rs){let ps=(rs||[]).filter(r=>(r.status||r.funnel_stage)==='promoted');if(!ps.length)return'';return ps.map((r,i)=>{let m=r.oos||{},st=r.stress_oos||{},p=r.params||{};return `<div class="card" style="margin:12px 0 0;padding:14px;background:#10161f"><div class="row" style="justify-content:space-between"><b>Promoted #${i+1}: ${r.strategy||'—'}</b><span class="pill good">PROMOTED</span></div><div class="grid">${M('Timeframe',(r.timeframe_min??p.timeframe_min??'—')+'m')}${M('Risk','€'+(p.risk_eur??r.risk_model?.fixed_risk_eur??'—'))}${M('Target',(p.target_r??'—')+'R')}${M('Stop ATR',p.stop_atr??'—')}${M('OOS trades',m.trades??'—')}${M('Trades/dag',m.avg_trades_per_day??'—')}${M('Winrate',(m.win_rate_pct??'—')+'%')}${M('Profit factor',m.profit_factor??'—')}${M('Expectancy',(m.expectancy_r??'—')+'R')}${M('Max DD',(m.max_drawdown_pct??'—')+'%')}${M('Payoff',m.payoff_ratio??'—')}${M('Stress PF',st.profit_factor??'—')}${M('Stress exp',(st.expectancy_r??'—')+'R')}</div><div class="muted" style="margin-top:8px">Score: ${r.funnel_score??'—'} · Positieve pairs: ${r.positive_pairs??'—'}/${r.pair_count??'—'} · Exit: ${exitLabel(p)} · Params: ${Object.entries(p).filter(([k])=>k!=='_phase').map(([k,v])=>k+'='+v).join(' · ')}</div></div>`}).join('')}function stageLabel(k){let x={promoted:'Promoted',precision_qualified:'Qualified',precision_deep_search:'Legacy qualified',precision_incubator:'Incubator',deep_search:'Deep search',incubator:'Incubator',discovery:'Discovery',precision_discovery:'Discovery',continuing:'Continuing',rejected:'Rejected',overig:'Overig'};return x[k]||String(k||'Overig').replaceAll('_',' ')}function PS(summary,rs){if(!summary?.length)return P(rs);let by={};for(let x of summary){let k=x.funnel_stage||'rejected',fam=x.family||'—',n=Number(x.candidates||0);by[k]??={total:0,fam:{}};by[k].total+=n;by[k].fam[fam]=(by[k].fam[fam]||0)+n}let order=['promoted','precision_qualified','precision_incubator','deep_search','incubator','discovery','rejected'];return order.filter(k=>by[k]?.total).map(k=>{let x=by[k],tags=Object.entries(x.fam).sort((a,b)=>b[1]-a[1]).map(([n,v])=>'<span class="tag">'+n+(v>1?' ×'+v:'')+'</span>').join('');return '<div class="phase '+k+'"><div class="phase-top"><div><h4>'+stageLabel(k)+'</h4><div class="sub">'+Object.keys(x.fam).length+' strategietype'+(Object.keys(x.fam).length===1?'':'n')+'</div></div><div class="count">'+x.total+'</div></div><div class="names">'+tags+'</div></div>'}).join('')}function P(rs){if(!rs?.length)return'<div class="phase-empty">Nog geen resultaten om over de fases te verdelen.</div>';let g={};for(let r of rs){let k=r.status||r.funnel_stage||r.phase||'overig';(g[k]??=[]).push(r)}let order=['promoted','precision_deep_search','deep_search','incubator','discovery','precision_discovery','continuing','rejected','overig'];let keys=[...new Set([...order,...Object.keys(g)])].filter(k=>g[k]?.length);return keys.map(k=>{let rows=g[k],freq={};for(let r of rows){let n=r.strategy||r.candidate?.strategy||'—';freq[n]=(freq[n]||0)+1}let tags=Object.entries(freq).sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0])).map(([n,v])=>'<span class="tag">'+n+(v>1?' ×'+v:'')+'</span>').join('');return '<div class="phase '+k+'"><div class="phase-top"><div><h4>'+stageLabel(k)+'</h4><div class="sub">'+Object.keys(freq).length+' strategietype'+(Object.keys(freq).length===1?'':'n')+'</div></div><div class="count">'+rows.length+'</div></div><div class="names">'+tags+'</div></div>'}).join('')}function T(rs){if(!rs?.length)return'<p class="muted">Nog geen opgeslagen resultaten.</p>';let h='<table><tr><th>Strategie</th><th>Stage</th><th>Trades</th><th>Trades/dag</th><th>PF</th><th>Expectancy</th></tr>';for(let r of rs.slice(0,100)){let m=r.oos||r.metrics||{};h+=`<tr><td>${r.strategy||r.candidate?.strategy||'—'}</td><td>${r.status||r.funnel_stage||r.phase||'—'}</td><td>${m.trades??r.trade_count??'—'}</td><td>${Number(m.avg_trades_per_day||0).toFixed(1)}</td><td>${Number(m.profit_factor||0).toFixed(2)}</td><td>${Number(m.expectancy_r||0).toFixed(2)}R</td></tr>`}return h+'</table>'}function agentHypotheses(rows){if(!rows?.length)return'<p class="muted" style="padding:10px">Nog geen hypotheses; de agent verzamelt eerst voldoende forex-resultaten.</p>';let h='<table><tr><th>Familie</th><th>TF</th><th>Score</th><th>PF</th><th>Hypothese</th></tr>';for(let x of rows.slice(0,8)){h+='<tr><td>'+(x.family||'—')+'</td><td>'+(x.timeframe_min||'—')+'m</td><td>'+Number(x.funnel_score||0).toFixed(1)+'</td><td>'+Number(x.profit_factor||0).toFixed(2)+'</td><td>'+(x.thesis||'—')+'</td></tr>'}return h+'</table>'}function renderAgent(ag){let on=!!ag.running,cl=on?'run':ag.stage==='error'?'bad':'';$('agent-badge').className='pill '+cl;$('agent-badge').textContent=on?'AGENT RUNNING':String(ag.stage||'AGENT OFF').toUpperCase();$('amini').className='pill '+cl;$('amini').textContent='Agent '+(on?'ON':String(ag.stage||'OFF').toUpperCase());$('agent-message').textContent=ag.message||'';let dec=ag.last_decision||{};$('agent-metrics').innerHTML=M('Status',ag.stage||'—')+M('Cycli',ag.cycles??0)+M('Laatste cyclus',String(ag.last_cycle_at||'—').replace('T',' ').slice(0,16))+M('Mode',dec.mode||dec.action||'—')+M('Focus TF',(ag.focus_timeframes||[]).map(x=>x+'m').join(' · ')||'—')+M('Hypotheses',(ag.hypotheses||[]).length);$('agent-focus').innerHTML=(ag.focus_families||[]).length?(ag.focus_families||[]).map(x=>'<span class="tag">'+x+'</span>').join(''):'<span class="muted">Nog geen focus gekozen.</span>';$('agent-hypotheses').innerHTML=agentHypotheses(ag.hypotheses||[])}async function loadResearch(){try{let d=await req('/api/research/results'),s=d.state||{},co=d.coordinator||{},ag=d.agent||{},val=d.validation||{},run=!!s.running,cl=run?'run':s.stage==='error'?'bad':'';$('research-status').innerHTML=`<span class="pill ${cl}">${run?'RUNNING':String(s.stage||'IDLE').toUpperCase()}</span><span class="pill">${String(co.mode||'COORDINATOR').toUpperCase()}</span>`;$('research-message').textContent=s.message||co.message||'';$('research-metrics').innerHTML=M('Getest',s.tested_total??0)+M('Promoted',s.promoted_total??0)+M('Kandidaat',s.current_candidate||'—')+M('TF',(s.current_params?.timeframe_min??'—')+'m')+M('Validaties',co.validations_completed??0)+M('17-staps lab',val.running?(val.current_lab||'RUNNING'):(val.completed_at?'COMPLETED':'IDLE'));renderAgent(ag);let rows=d.results||[];$('research-phases').innerHTML=PS(d.funnel_summary,rows);$('overview-research-phases').innerHTML=PS(d.funnel_summary,rows);let h='<table><tr><th>Strategie</th><th>TF</th><th>Stage</th><th>Trades/dag</th><th>PF</th><th>Expectancy</th></tr>';for(let r of rows.slice(0,50)){let m=r.oos||{},p=r.params||{};h+=`<tr><td>${r.strategy||'—'}</td><td>${p.timeframe_min||'—'}m</td><td>${r.funnel_stage||'—'}</td><td>${Number(m.avg_trades_per_day||0).toFixed(1)}</td><td>${Number(m.profit_factor||0).toFixed(2)}</td><td>${Number(m.expectancy_bps||0).toFixed(2)} bps</td></tr>`}$('research-results').innerHTML=h+'</table>';$('rmini').className='pill '+cl;$('rmini').textContent='Research '+(run?'RUNNING':String(s.stage||'IDLE').toUpperCase())}catch(e){$('research-message').textContent=e.message}}async function researchAct(a){try{await req('/api/research/'+a,{method:'POST'});await loadResearch()}catch(e){$('research-message').textContent=e.message}}async function load(n){try{let d=await req('/api/'+n+'/results'),s=d.state||{},run=s.running,cl=run?'run':s.stage==='error'?'bad':'';$(n+'-status').innerHTML=`<span class="pill ${cl}">${run?'RUNNING':(s.stage||'IDLE').toUpperCase()}</span>`;$(n+'-message').textContent=s.message||'';$(n+'-progress').style.width=(s.total?Math.round((s.progress||0)*100/s.total):0)+'%';$(n+'-metrics').innerHTML=M('Getest',s.tested_total)+M('Voortgang',(s.progress||0)+' / '+(s.total||0))+M('Strategie',s.current_candidate||'—')+M('Pair',s.current_pair||'—')+M('Opgeslagen',s.results_loaded??d.results?.length??0)+M('Laatste',s.last_completed_candidate||'—');if(n==='forex-lab')$('forex-lab-promoted').innerHTML=D(d.results);$(n+'-phases').innerHTML=PS(d.funnel_summary,d.results);if(n==='forex-lab')$('overview-forex-phases').innerHTML=PS(d.funnel_summary,d.results);if(n==='precision-lab')$('overview-precision-phases').innerHTML=PS(d.funnel_summary,d.results);$(n+'-results').innerHTML=T(d.results);let id=n==='forex-lab'?'fmini':'pmini';$(id).className='pill '+cl;$(id).textContent=(n==='forex-lab'?'Forex ':'Precision ')+(run?'RUNNING':(s.stage||'IDLE').toUpperCase())}catch(e){$(n+'-message').textContent=e.message}}function money(x){return '€'+Number(x||0).toFixed(2)}function exitLabel(p){p=p||{};let tr=p.management_trigger_r,lr=p.management_lock_net_r;if(tr!=null&&lr!=null)return Number(tr).toFixed(Number(tr)%1?1:0)+'R → netto +'+Number(lr).toFixed(2)+'R';let m=String(p.exit_mode||'baseline');return m==='protect_2r_025r'?'2R → netto +0.25R':m==='breakeven_2r'?'2R → kosten-BE +0.05R':m==='lock_2r_05r'?'2R → netto +0.50R':'Baseline'}let activePaperTab=localStorage.getItem('paperTab')||'';
function prettyStrategyName(v){return String(v||'—').split('_').map(x=>x?x[0].toUpperCase()+x.slice(1):x).join(' ')}
function strategyVariant(r){let p=r.params||{},s=String(r.strategy||'');if(s==='trend_pullback')return 'F'+(p.fast??'—')+'/S'+(p.slow??'—')+' · z'+(p.pullback_z??'—');if(s==='range_reversal'||s==='mean_reversion')return 'W'+(p.window??'—')+' · z'+(p.z_entry??'—');if(s==='momentum')return 'F'+(p.fast??'—')+'/S'+(p.slow??'—')+' · '+(p.entry_bps??'—')+'bps';if(s==='breakout')return 'W'+(p.window??'—')+' · '+(p.buffer_bps??'—')+'bps';if(s==='volatility_breakout')return 'W'+(p.window??'—')+' · vol×'+(p.vol_mult??'—');if(s==='extreme_reversal')return 'W'+(p.window??'—')+' · shock '+(p.shock_z??'—')+'σ';if(s==='vwap_reversion')return 'W'+(p.window??'—')+' · z'+(p.z_entry??'—');if(s==='vwap_momentum')return 'W'+(p.window??'—')+' · '+(p.buffer_bps??'—')+'bps';if(s==='asymmetric_breakout')return 'W'+(p.window??'—')+' · stop '+(p.stop_atr??'—')+'ATR · '+(p.target_r??'—')+'R';return ''}
function paperCodeName(r){let x=String(r.paper_id||'paper-000000').replace('paper-',''),names=['Atlas','Nova','Orion','Vega','Titan','Apex','Pulse','Falcon','Nimbus','Vector','Quartz','Comet','Raven','Helix','Drift','Echo','Zenith','Vertex','Sierra','Cosmos'];let h=0;for(let i=0;i<x.length;i++)h=(h*31+x.charCodeAt(i))>>>0;return names[h%names.length]+'-'+x.slice(-3).toUpperCase()}
function stratLabel(r){return paperCodeName(r)}
function paperParamLabel(k){let m={fast:'Fast',slow:'Slow',window:'Window',pullback_z:'Pullback z',z_entry:'Entry z',z_exit:'Exit z',entry_bps:'Entry',buffer_bps:'Buffer',vol_mult:'Vol mult',shock_z:'Shock z',max_hold:'Max hold',timeframe_min:'Timeframe',min_volume_ratio:'Min volume',volume_window:'Volume window',stop_atr:'Stop ATR',target_r:'Target R',risk_eur:'Risk €',entry_sessions:'Sessions',direction_mode:'Direction'};return m[k]||k.replaceAll('_',' ')}
function fmtParam(k,v){if(k==='timeframe_min')return v+'m';if(k==='entry_bps'||k==='buffer_bps')return v+' bps';if(k==='risk_eur')return '€'+Number(v).toFixed(2);return String(v)}
function paperParamsHtml(s){let p=s.params||{},skip=new Set(['market','data_source','_paper_source','_paper_exit_model','_policy_version','_phase','start_capital_eur']);skip.add('risk_eur');let rows=Object.entries(p).filter(([k,v])=>!skip.has(k)&&!k.startsWith('_')&&v!==null&&v!==undefined);if(!rows.length)return'';return '<div class="paper-params">'+rows.map(([k,v])=>'<span class="paper-param">'+paperParamLabel(k)+': <b>'+fmtParam(k,v)+'</b></span>').join('')+'</div>'}
function liveStreaks(rows){let maxW=0,maxL=0,w=0,l=0;for(let r of rows||[]){let p=Number(r.pnl||0);if(p>0){w++;l=0;maxW=Math.max(maxW,w)}else if(p<0){l++;w=0;maxL=Math.max(maxL,l)}else{w=0;l=0}}return{maxW,maxL}}
function paperOverview(s,trades){let p=s.params||{},tr=Number(s.trades||0),wins=Number(s.wins||0),losses=Number(s.losses||0),wr=tr?100*wins/tr:0,src=p._paper_source==='research'?'Research 17/17':'Forex Promoted',closed=(trades||[]).filter(r=>r.paper_id===s.paper_id).sort((a,b)=>String(a.exit_time||'').localeCompare(String(b.exit_time||''))),ls=liveStreaks(closed),btL=p._research_max_loss_streak,btW=p._research_max_win_streak;return '<div class="row" style="justify-content:space-between;align-items:flex-start"><div><div class="paper-name">'+paperCodeName(s)+'</div><div class="paper-family muted">'+prettyStrategyName(s.strategy)+' · '+(s.timeframe_min??p.timeframe_min??'—')+'m · '+strategyVariant(s)+'</div></div><div class="row"><span class="pill">'+src+'</span><span class="pill '+(s.status==='active'?'run':(['ruined','review_pause'].includes(s.status)?'bad':''))+'">'+String(s.status||'').replaceAll('_',' ').toUpperCase()+'</span></div></div><div class="grid">'+M('Start',money(s.start_balance))+M('Balans',money(s.balance))+M('P/L',money(s.net_pnl))+M('Trades',tr)+M('Winrate',wr.toFixed(1)+'%')+M('Wins / Losses',wins+' / '+losses)+M('Open posities',s.open_positions??0)+M('Risk / nieuwe trade',money(s.paper_risk_eur??p.risk_eur))+M('Paper max win streak',ls.maxW)+M('Paper max loss streak',ls.maxL)+M('Huidige loss streak',s.paper_current_loss_streak??0)+M('Pause grens',s.paper_loss_streak_limit??'—')+M('Backtest max win streak',btW??'legacy')+M('Backtest max loss streak',btL??'legacy')+M('Exit model',p._paper_exit_model==='research_native'?'Research native':'RR management')+'</div>'+paperParamsHtml(s)+'<div class="muted" style="margin-top:10px;font-size:.78rem">Referentie: <b>'+paperCodeName(s)+'</b> · Paper ID '+String(s.paper_id||'—')+' · gestart '+String(s.started_at||'').replace('T',' ').slice(0,16)+'</div>'}
function dailyFor(s,rows){let rs=(rows||[]).filter(r=>r.paper_id===s.paper_id);if(!rs.length)return'<div class="paper-empty">Nog geen dagdata.</div>';return '<div class="results"><table><tr><th>Datum</th><th>Start</th><th>P/L</th><th>Eind</th><th>Trades</th><th>W/L</th><th>Open</th></tr>'+rs.map(r=>'<tr><td>'+r.trade_date+'</td><td>'+money(r.start_balance)+'</td><td class="'+(Number(r.realized_pnl)>=0?'pnlpos':'pnlneg')+'">'+money(r.realized_pnl)+'</td><td>'+money(r.end_balance)+'</td><td>'+r.trade_count+'</td><td>'+r.wins+'/'+r.losses+'</td><td>'+r.open_positions+'</td></tr>').join('')+'</table></div>'}
function positionsFor(s,rows){let rs=(rows||[]).filter(r=>r.paper_id===s.paper_id);if(!rs.length)return'<div class="paper-empty">Geen open posities.</div>';let h='<div class="results"><table><tr><th>Pair</th><th>Side</th><th>Entry</th><th>Huidig</th><th>Open R</th><th>Open P/L</th><th>Stop</th><th>Target</th></tr>';for(let r of rs){let ur=Number(r.unrealized_r||0),up=Number(r.unrealized_pnl||0),cl=up>=0?'pnlpos':'pnlneg',native=(r.params||{})._paper_source==='research'&&(r.params||{})._paper_exit_model==='research_native';h+='<tr><td>'+r.pair+'</td><td>'+(Number(r.direction)>0?'LONG':'SHORT')+'</td><td>'+Number(r.entry_price).toFixed(5)+'</td><td>'+Number(r.current_price||r.entry_price).toFixed(5)+'</td><td class="'+cl+'">'+ur.toFixed(2)+'R</td><td class="'+cl+'">'+money(up)+'</td><td>'+(native?'hard ≤1R':Number(r.stop_price).toFixed(5))+'</td><td>'+(native?'native':Number(r.target_price).toFixed(5))+'</td></tr>'}return h+'</table></div>'}
function tradesFor(s,rows){let rs=(rows||[]).filter(r=>r.paper_id===s.paper_id);if(!rs.length)return'<div class="paper-empty">Nog geen gesloten trades.</div>';let h='<div class="results"><table><tr><th>Exit</th><th>Pair</th><th>Side</th><th>Reden</th><th>R</th><th>P/L</th><th>Balans</th></tr>';for(let r of rs){let rr=Number(r.r_multiple),pnl=Number(r.pnl),cl=pnl>=0?'pnlpos':'pnlneg';h+='<tr><td>'+String(r.exit_time).replace('T',' ').slice(0,16)+'</td><td>'+r.pair+'</td><td>'+String(r.side||'').toUpperCase()+'</td><td>'+r.exit_reason+'</td><td class="'+cl+'">'+rr.toFixed(2)+'R</td><td class="'+cl+'">'+money(pnl)+'</td><td>'+money(r.balance_after)+'</td></tr>'}return h+'</table></div>'}
function renderPaperPane(s,daily,positions,trades){return '<div class="paper-pane">'+paperOverview(s,trades)+'<div class="paper-subsection"><h3>Daily statistieken</h3>'+dailyFor(s,daily)+'</div><div class="paper-subsection"><h3>Open trades</h3>'+positionsFor(s,positions)+'</div><div class="paper-subsection"><h3>Gesloten trades</h3>'+tradesFor(s,trades)+'</div></div>'}
function setPaperTab(id){activePaperTab=id;localStorage.setItem('paperTab',id);document.querySelectorAll('.paper-tab').forEach(b=>b.classList.toggle('active',b.dataset.id===id));document.querySelectorAll('.paper-pane-wrap').forEach(p=>p.style.display=p.dataset.id===id?'block':'none')}
function renderPaperTabs(strategies,daily,positions,trades){if(!strategies?.length){$('paper-tabs').innerHTML='';$('paper-tab-content').innerHTML='<div class="paper-empty" style="margin-top:12px">Nog geen actieve paperstrategieën.</div>';return}if(!strategies.some(s=>s.paper_id===activePaperTab))activePaperTab=strategies[0].paper_id;$('paper-tabs').innerHTML=strategies.map(s=>'<button class="paper-tab '+(s.paper_id===activePaperTab?'active':'')+'" data-id="'+s.paper_id+'" onclick="setPaperTab(\''+s.paper_id+'\')"><b>'+paperCodeName(s)+'</b> · '+prettyStrategyName(s.strategy)+'</button>').join('');$('paper-tab-content').innerHTML=strategies.map(s=>'<div class="paper-pane-wrap" data-id="'+s.paper_id+'" style="display:'+(s.paper_id===activePaperTab?'block':'none')+'">'+renderPaperPane(s,daily,positions,trades)+'</div>').join('')}
async function loadPaper(){try{let d=await req('/api/paper/results'),st=d.state||{},run=st.running,cl=run?'run':st.stage==='error'?'bad':'',eligible=(d.strategies||[]).filter(r=>['active','retiring','review_pause','ruined'].includes(String(r.status||''))).map(r=>({...r,paper_risk_eur:Number(st.paper_risk_eur||1.5)})),ids=new Set(eligible.map(r=>r.paper_id)),daily=(d.daily||[]).filter(r=>ids.has(r.paper_id)),positions=(d.positions||[]).filter(r=>ids.has(r.paper_id)),trades=(d.trades||[]).filter(r=>ids.has(r.paper_id));$('paper-status').innerHTML='<span class="pill '+cl+'">'+(run?'RUNNING':String(st.stage||'IDLE').toUpperCase())+'</span><span class="pill '+(st.portfolio_frequency_ready?'good':'')+'">Portfolio '+Number(st.expected_portfolio_trades_per_day||0).toFixed(1)+'/'+Number(st.portfolio_target_trades_per_day||10).toFixed(0)+' trades/dag · '+eligible.length+' strategieën</span>';$('paper-message').textContent=st.message||'';renderPaperTabs(eligible,daily,positions,trades);$('papermini').className='pill '+cl;$('papermini').textContent='Paper '+(run?'RUNNING':String(st.stage||'IDLE').toUpperCase())}catch(e){$('paper-message').textContent=e.message}}
async function paperAct(a){try{await req('/api/paper/'+a,{method:'POST'});await loadPaper()}catch(e){$('paper-message').textContent=e.message}}async function connect(){try{let h=await fetch('/health').then(r=>r.json());$('connection').textContent=`Online · live trading UIT · cTrader ${h.ctrader_ready?'ready':'niet ready'}`;await Promise.all([loadResearch(),load('forex-lab'),load('precision-lab'),loadPaper()])}catch(e){$('connection').textContent=e.message}}async function act(n,a){try{await req('/api/'+n+'/'+a,{method:'POST'});await load(n)}catch(e){$(n+'-message').textContent=e.message}}async function authorize(){try{let d=await req('/api/ctrader/oauth-url');location.href=d.url}catch(e){$('connection').textContent=e.message}}if(tok)connect();setInterval(()=>{if(tok){loadResearch();load('forex-lab');load('precision-lab');loadPaper()}},15000)</script></body></html>'''
