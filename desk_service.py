from __future__ import annotations
import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import HTMLResponse
from pathlib import Path
from ctrader_client import CTraderClient
from desk_core import (Costs, candidates, dt, roll_day, sessions, trade_day, can_enter,
                       size_plan, open_position, reconcile_bars, reconcile_quote, VERSION)
from desk_agents import AgentDesk
from desk_store import DeskStore

log=logging.getLogger('microtrader.desk')


class DeskFeed(CTraderClient):
    """One serialized, read-only socket for this tester."""
    def __init__(self,settings):
        super().__init__(settings)
        self.cache={}

    async def symbol(self,pair):
        if pair not in self.cache:
            self.cache[pair]=await self.resolve_symbol(pair)
        return self.cache[pair]

    async def quote(self,pair):
        symbol=await self.symbol(pair)
        sid=int(symbol['symbolId'])
        await self._send(2127,{'ctidTraderAccountId':self.account_id,'symbolId':[sid],
                               'subscribeToSpotTimestamp':True})
        parts={}
        try:
            async with asyncio.timeout(12):
                while True:
                    msg=json.loads(await self._ws.recv())
                    typ=int(msg.get('payloadType',0)); p=msg.get('payload') or {}
                    if typ in (50,2142):
                        raise RuntimeError('cTrader quote error')
                    if typ!=2131 or int(p.get('symbolId',0))!=sid:
                        continue
                    stamp=int(p.get('timestamp',0))/1000
                    for k in ('bid','ask'):
                        if k in p:
                            parts[k]=(float(p[k])/100000,stamp)
                    if 'bid' in parts and 'ask' in parts:
                        now=datetime.now(timezone.utc).timestamp()
                        if any(not 0<=now-x[1]<=15 for x in parts.values()):
                            raise RuntimeError('Verouderde cTrader-koers')
                        return dict(bid=parts['bid'][0],ask=parts['ask'][0],
                                    timestamp=datetime.fromtimestamp(min(x[1] for x in parts.values()),timezone.utc).isoformat())
        finally:
            await self._send(2129,{'ctidTraderAccountId':self.account_id,'symbolId':[sid]})
            await self._recv_until(2130,timeout=5)


class DeskService:
    def __init__(self,settings,token_source):
        self.settings=settings
        self.database_url=os.getenv('DESK_DATABASE_URL','').strip() or settings.database_url
        self.token_source=token_source
        self.feed=DeskFeed(settings)
        self.store=DeskStore(self.database_url)
        self.agents=AgentDesk()
        self.costs=Costs(commission_usd_per_unit=float(os.getenv('DESK_COMMISSION_USD_PER_LOT','4.5'))/100000,
                         slippage_pips=float(os.getenv('DESK_SLIPPAGE_PIPS','0.2')),
                         max_spread_pips=float(os.getenv('DESK_MAX_SPREAD_PIPS','1.5')),
                         leverage=float(os.getenv('DESK_MODEL_LEVERAGE','30')))
        self.task=None
        self.last_error=''
        self.last_cycle=None
        self.phase='idle'
        self.bars={}
        self.bars_at={}

    def blockers(self):
        b=[]
        if not self.database_url:
            b.append('DATABASE_URL ontbreekt')
        if not self.agents.ready:
            b.append('OPENAI_API_KEY ontbreekt voor de vijf agents')
        if not self.token_source.api_ready:
            b.append('cTrader demoautorisatie ontbreekt')
        if self.settings.ctrader_environment!='demo' or not self.settings.ctrader_demo_only:
            b.append('Alleen cTrader demo toegestaan')
        return b

    async def start(self):
        if self.task and not self.task.done():
            return
        if not self.database_url:
            self.phase='blocked';return
        await self.store.init()
        if os.getenv('DESK_AUTO_START','false').lower()=='true' and not self.blockers():
            await self.store.enable(True)
        self.task=asyncio.create_task(self.loop(),name='five-agent-desk')

    async def stop(self):
        if self.task:
            self.task.cancel()
            try:await self.task
            except asyncio.CancelledError:pass
        self.task=None
        await self.feed.close()

    async def status(self):
        state=await self.store.read() if self.database_url else None
        return dict(state=state,phase=self.phase,last_error=self.last_error,last_cycle=self.last_cycle,
                    blockers=self.blockers(),model=self.agents.model,version=VERSION,
                    running=bool(self.task and not self.task.done()),mode='forward-paper-only',
                    sessions=sessions(datetime.now(timezone.utc)),costs=self.costs.__dict__,
                    limitations=['Geen brokerorders: eigen virtueel testboek',
                                 'M1-candles met gemodelleerde ask/spread; dubbel geraakte candle = stop eerst',
                                 'Geen externe nieuwsfeed; agents mogen geen actueel nieuws verzinnen',
                                 'Geen gegarandeerde maximale verliesfill bij gaps',
                                 'Commissie/leverage zijn configureerbare aannames, geen geverifieerd accounttarief'])

    async def loop(self):
        while True:
            try:
                await self.cycle()
                self.last_error=''
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.phase='error'
                # Never expose provider exception strings, which can contain secrets.
                self.last_error=f'{type(e).__name__}: cyclus niet voltooid; geen nieuwe trade uitgevoerd'
                log.warning('Agent desk cycle failed (%s)',type(e).__name__)
                await self.feed.disconnect()
            self.last_cycle=datetime.now(timezone.utc).isoformat()
            await asyncio.sleep(5)

    async def rows(self,pair,now):
        minute=now.replace(second=0,microsecond=0)
        if self.bars_at.get(pair)!=minute:
            self.bars[pair]=await self.feed.historical_bars(pair,timeframe_min=1,max_bars=800,lookback_days=4)
            self.bars_at[pair]=minute
            await asyncio.sleep(.25)
        return self.bars[pair]

    async def cycle(self):
        owner,state=await self.store.acquire()
        if not owner:
            return
        events=[]
        try:
            async with asyncio.timeout(150):
                now=datetime.now(timezone.utc)
                roll_day(state,now)
                if not state['enabled'] and not state['position']:
                    self.phase='paused'
                    return
                # Existing paper positions keep being managed while entries are paused or AI is down.
                if not self.token_source.api_ready:
                    self.phase='blocked';state['message']='cTrader demoautorisatie nodig';return
                if self.settings.ctrader_environment!='demo' or not self.settings.ctrader_demo_only:
                    raise RuntimeError('demo required')
                if not sessions(now) and not state['position']:
                    self.phase='outside_session';state['message']='Wacht op London / New York';return
                if self.feed.active_access_token != self.token_source.active_access_token:
                    await self.feed.disconnect()
                    self.feed.set_runtime_tokens(self.token_source.active_access_token,self.token_source.active_refresh_token)
                if not self.feed.account_authenticated:
                    await self.feed.connect_and_authenticate()
                conversion=await self.feed.quote('EUR/USD')
                eurusd=(conversion['bid']+conversion['ask'])/2
                p=state['position']
                if p:
                    self.phase='managing'
                    quote=conversion if p['pair']=='EUR/USD' else await self.feed.quote(p['pair'])
                    rows=await self.rows(p['pair'],now)
                    spread=max(quote['ask']-quote['bid'],p.get('spread_at_entry',0))
                    closed=reconcile_bars(state,rows,spread,now,eurusd)
                    if not closed:
                        closed=reconcile_quote(state,quote,now,eurusd,
                                               force=not sessions(now) or p['day']!=trade_day(now))
                    if closed:
                        events.append(('trade_closed',closed))
                        state['message']=f"Trade gesloten: {closed['r']:.2f}R ({closed['reason']})"
                        roll_day(state,now)
                    else:
                        state['position']['mark']=quote
                    return
                if not can_enter(state,now):
                    self.phase='day_done';state['message']='Daglimiet bereikt: winsttrade of twee trades/verliezen';return
                if not self.agents.ready:
                    self.phase='blocked';state['message']='OPENAI_API_KEY nodig voor de vijf agents';return
                if state['debate_count']>=int(os.getenv('DESK_MAX_DEBATES_PER_DAY','12')):
                    self.phase='budget_limit';state['message']='Daglimiet AI-beoordelingen bereikt';return
                self.phase='scanning'
                for pair in ('EUR/USD','GBP/USD'):
                    rows=await self.rows(pair,now)
                    found=candidates(pair,rows,now)
                    for candidate in found:
                        if candidate['id'] in state['debated']:
                            continue
                        # One fully deliberated candidate per cycle, no parallel entries.
                        state['debated'].append(candidate['id']);state['debate_count']+=1
                        quote=conversion if pair=='EUR/USD' else await self.feed.quote(pair)
                        symbol=await self.feed.symbol(pair)
                        try:
                            plan=size_plan(candidate,quote,state['balance'],eurusd,symbol,self.costs)
                        except ValueError as e:
                            state['message']=str(e);events.append(('candidate_rejected',dict(candidate_id=candidate['id'],reason=str(e))))
                            return
                        snapshot=dict(candidate=candidate,plan=plan,quote=quote,as_of=now.isoformat(),
                                      sessions=sessions(now),balance=state['balance'],
                                      data_source='cTrader demo',news='not supplied',
                                      daily=dict(entries=state['entries'],losses=state['losses']))
                        self.phase='debating'
                        decision=await self.agents.decide(snapshot)
                        after=datetime.now(timezone.utc)
                        decision['at']=after.isoformat()
                        state['decisions']=(state['decisions']+[decision])[-30:]
                        events.append(('decision',decision))
                        if not decision['approved']:
                            state['message']='Geen unanimiteit: setup overgeslagen';return
                        if not can_enter(state,after) or after>dt(candidate['expires_at']):
                            state['message']='Setup verlopen tijdens overleg';return
                        fresh=await self.feed.quote(pair)
                        fx=fresh if pair=='EUR/USD' else await self.feed.quote('EUR/USD')
                        fxrate=(fx['bid']+fx['ask'])/2
                        try:
                            newplan=size_plan(candidate,fresh,state['balance'],fxrate,symbol,self.costs)
                        except ValueError as e:
                            state['message']=str(e);return
                        # All five agreed to these levels. A changed fill requires a new debate.
                        if abs(newplan['entry']-plan['entry']) > .1*candidate['atr']:
                            state['message']='Koers te ver veranderd na consensus';return
                        plan['spread_at_entry']=fresh['ask']-fresh['bid']
                        # Final exact plan/size is recomputed; stop and >=2R policy remain frozen.
                        newplan['spread_at_entry']=fresh['ask']-fresh['bid']
                        newplan['decision_id']=candidate['id']
                        open_position(state,newplan,datetime.now(timezone.utc),candidate['id'])
                        events.append(('trade_opened',state['position']))
                        state['message']='5/5 akkoord: virtuele trade geopend';self.phase='managing'
                        return
                state['message']='Geen nieuwe geldige M1/M5 setup; geen trade geforceerd'
        finally:
            # Atomic state + audit events. If canceled, lease expires; no external orders exist.
            try:
                await self.store.save(owner,state,events)
            except BaseException:
                await self.store.release(owner)
                raise


def desk_router(service,auth):
    router=APIRouter()

    @router.get('/desk',response_class=HTMLResponse)
    async def page():
        return HTMLResponse(Path(__file__).with_name('desk.html').read_text(),headers={'Cache-Control':'no-store'})

    @router.get('/api/desk/status')
    async def status(authorization:str|None=Header(None)):
        auth(authorization)
        try:return await service.status()
        except Exception:raise HTTPException(503,'Testboek niet beschikbaar; controleer database')

    @router.post('/api/desk/start')
    async def start(authorization:str|None=Header(None)):
        auth(authorization)
        if service.blockers():raise HTTPException(503,service.blockers())
        await service.start()
        await service.store.enable(True)
        return await service.status()

    @router.post('/api/desk/pause')
    async def pause(authorization:str|None=Header(None)):
        auth(authorization)
        await service.store.enable(False)
        return await service.status()

    @router.get('/api/desk/export')
    async def export(authorization:str|None=Header(None)):
        auth(authorization)
        return dict(state=await service.store.read(),journal=await service.store.export())

    return router
