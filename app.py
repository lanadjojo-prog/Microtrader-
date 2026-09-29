from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from config import settings
from ctrader_client import CTraderClient, CTraderError
from forex_lab import ForexStrategyLab
from precision_lab import PrecisionStrategyLab

logging.basicConfig(level=logging.INFO)
ctrader=CTraderClient(settings); forex_lab=ForexStrategyLab(settings,ctrader); precision_lab=PrecisionStrategyLab(settings,ctrader)

async def autostart():
    if settings.forex_lab_auto_start: await forex_lab.start()
    if settings.precision_lab_auto_start: await precision_lab.start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await autostart(); yield
    await precision_lab.stop(); await forex_lab.stop(); await ctrader.close()

app=FastAPI(title="ForexTrader Research",lifespan=lifespan)
def auth(a):
    if not settings.dashboard_token or a!=f"Bearer {settings.dashboard_token}": raise HTTPException(401,"Unauthorized")

@app.get('/health')
async def health(): return {'ok':True,'mode':'forex-research-only','live_trading_enabled':False,'forex_lab_running':forex_lab.state.running,'forex_stage':forex_lab.state.stage,'precision_lab_running':precision_lab.state.running,'precision_stage':precision_lab.state.stage,'ctrader_ready':ctrader.api_ready}
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
        await ctrader.exchange_code(code); state=await ctrader.connect_and_authenticate(); await autostart()
        return HTMLResponse(f"<h3>Fusion Markets demo connected</h3><p>{state.get('environment','-')} · account {state.get('account_id','-')}</p><p>Research labs zijn gestart waar mogelijk. Live orders staan uit.</p>")
    except Exception as e:return HTMLResponse(f'<h3>cTrader authorization failed</h3><p>{e}</p>',503)
@app.get('/',response_class=HTMLResponse)
async def home():return HTMLResponse(DASHBOARD)

DASHBOARD=r'''<!doctype html><html lang="nl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>ForexTrader</title><style>
*{box-sizing:border-box}body{font-family:system-ui,-apple-system,sans-serif;background:#0e1116;color:#edf2f7;margin:0;padding:24px;max-width:1000px;margin:auto}h1{margin:0}.muted{color:#96a4b5}.top,.row{display:flex;gap:9px;flex-wrap:wrap;align-items:center}.top{justify-content:space-between;margin-bottom:18px}.card{background:#171c24;border:1px solid #2a3442;border-radius:16px;padding:18px;margin:14px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(135px,1fr));gap:9px;margin-top:12px}.metric{background:#111720;padding:11px;border-radius:11px}.metric b{display:block;font-size:1.1rem;margin-top:4px}.pill{padding:5px 9px;border-radius:999px;background:#242d39;font-size:.82rem}.good{background:#17351f;color:#9ee8ad}.run{background:#17314a;color:#a8d8ff}.bad{background:#3a1c22;color:#ffadb5}button,input{font:inherit;border-radius:10px;border:1px solid #3a4658;padding:10px 12px;background:#0f141b;color:#fff}button{cursor:pointer}.progress{height:9px;background:#0f141b;border-radius:99px;overflow:hidden;margin:12px 0}.progress div{height:100%;background:#e9eef5;width:0}.results{overflow-x:auto;margin-top:12px}table{width:100%;border-collapse:collapse;font-size:.84rem}th,td{text-align:left;padding:8px;border-bottom:1px solid #2a3442}th{color:#96a4b5}.overview{position:static}.config{font-size:.9rem;line-height:1.5}@media(max-width:600px){body{padding:13px}.card{padding:13px;margin:10px 0}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}h1{font-size:1.5rem}.metric{padding:9px}button{padding:9px 10px}}
</style></head><body><div class="top"><div><h1>ForexTrader</h1><div class="muted">Autonome research · geen live trading</div></div><span class="pill good">RESEARCH ONLY</span></div>
<div class="card"><div class="row"><input id="token" type="password" placeholder="Dashboard token"><button onclick="connect()">Connect</button><button onclick="authorize()">cTrader koppelen</button></div><div id="connection" class="muted" style="margin-top:8px">Voer je token in.</div></div>
<div class="card overview"><div class="row"><b>Status</b><span id="fmini" class="pill">Forex —</span><span id="pmini" class="pill">Precision —</span></div></div>
<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">Forex Lab</h2><span class="pill">BREDE FUNNEL</span></div><p class="config muted">Breakout · trend pullback · range reversal · liquidity sweep · VWAP · momentum continuation. Robuustheid over pairs, periodes en regimes.</p><div class="row"><button onclick="act('forex-lab','start')">Start</button><button onclick="act('forex-lab','stop')">Stop</button><button onclick="load('forex-lab')">Refresh</button></div><div id="forex-lab-status" class="row" style="margin-top:12px"></div><div id="forex-lab-message" class="muted"></div><div class="progress"><div id="forex-lab-progress"></div></div><div id="forex-lab-metrics" class="grid"></div><div id="forex-lab-results" class="results"></div></div>
<div class="card"><div class="row" style="justify-content:space-between"><h2 style="margin:0">Precision Lab</h2><span class="pill">PIP FUNNEL</span></div><p class="config muted">€50 start · €2/€3 risico · stops 2–5 pips · targets 4–10 pips · bid/ask tick execution · kosten- en survivaltests.</p><div class="row"><button onclick="act('precision-lab','start')">Start</button><button onclick="act('precision-lab','stop')">Stop</button><button onclick="load('precision-lab')">Refresh</button></div><div id="precision-lab-status" class="row" style="margin-top:12px"></div><div id="precision-lab-message" class="muted"></div><div class="progress"><div id="precision-lab-progress"></div></div><div id="precision-lab-metrics" class="grid"></div><div id="precision-lab-results" class="results"></div></div>
<script>const $=x=>document.getElementById(x);let tok=localStorage.getItem('ftToken')||'';$('token').value=tok;function H(){tok=$('token').value.trim();localStorage.setItem('ftToken',tok);return{'Authorization':'Bearer '+tok}}async function req(u,o={}){o.headers={...(o.headers||{}),...H()};let r=await fetch(u,o),j=await r.json().catch(()=>({}));if(!r.ok)throw Error(j.detail||'HTTP '+r.status);return j}function M(a,b){return`<div class="metric"><span class="muted">${a}</span><b>${b??'—'}</b></div>`}function T(rs){if(!rs?.length)return'<p class="muted">Nog geen opgeslagen resultaten.</p>';let h='<table><tr><th>Strategie</th><th>Stage</th><th>Trades</th><th>PF</th><th>Expectancy</th></tr>';for(let r of rs.slice(0,12)){let m=r.oos||r.metrics||{};h+=`<tr><td>${r.strategy||r.candidate?.strategy||'—'}</td><td>${r.funnel_stage||'—'}</td><td>${m.trades??r.trade_count??'—'}</td><td>${Number(m.profit_factor||0).toFixed(2)}</td><td>${Number(m.expectancy_r||0).toFixed(2)}R</td></tr>`}return h+'</table>'}async function load(n){try{let d=await req('/api/'+n+'/results'),s=d.state||{},run=s.running,cl=run?'run':s.stage==='error'?'bad':'';$(n+'-status').innerHTML=`<span class="pill ${cl}">${run?'RUNNING':(s.stage||'IDLE').toUpperCase()}</span>`;$(n+'-message').textContent=s.message||'';$(n+'-progress').style.width=(s.total?Math.round((s.progress||0)*100/s.total):0)+'%';$(n+'-metrics').innerHTML=M('Getest',s.tested_total)+M('Voortgang',(s.progress||0)+' / '+(s.total||0))+M('Strategie',s.current_candidate||'—')+M('Pair',s.current_pair||'—')+M('Opgeslagen',s.results_loaded??d.results?.length??0)+M('Laatste',s.last_completed_candidate||'—');$(n+'-results').innerHTML=T(d.results);let id=n==='forex-lab'?'fmini':'pmini';$(id).className='pill '+cl;$(id).textContent=(n==='forex-lab'?'Forex ':'Precision ')+(run?'RUNNING':(s.stage||'IDLE').toUpperCase())}catch(e){$(n+'-message').textContent=e.message}}async function connect(){try{let h=await fetch('/health').then(r=>r.json());$('connection').textContent=`Online · live trading UIT · cTrader ${h.ctrader_ready?'ready':'niet ready'}`;await Promise.all([load('forex-lab'),load('precision-lab')])}catch(e){$('connection').textContent=e.message}}async function act(n,a){try{await req('/api/'+n+'/'+a,{method:'POST'});await load(n)}catch(e){$(n+'-message').textContent=e.message}}async function authorize(){try{let d=await req('/api/ctrader/oauth-url');location.href=d.url}catch(e){$('connection').textContent=e.message}}if(tok)connect();setInterval(()=>{if(tok){load('forex-lab');load('precision-lab')}},15000)</script></body></html>'''
