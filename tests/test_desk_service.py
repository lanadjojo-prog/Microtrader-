import copy
import unittest
from datetime import datetime,timezone,timedelta
from types import SimpleNamespace
from unittest.mock import patch,AsyncMock
import httpx
from fastapi import FastAPI,HTTPException
from desk_service import DeskService,desk_router
from desk_core import initial_state,dt,ROLES

NOW=dt('2026-11-02T10:00:00Z')


class Clock(datetime):
    @classmethod
    def now(cls,tz=None):return NOW


class Store:
    def __init__(self):self.state=initial_state(1000);self.state['enabled']=True;self.events=[]
    async def acquire(self):return 'lease',copy.deepcopy(self.state)
    async def save(self,owner,state,events):self.state=state;self.events.extend(events)
    async def release(self,owner):pass
    async def read(self):return self.state


class Feed:
    active_access_token='token';account_authenticated=True
    async def quote(self,pair):return dict(bid=1.1,ask=1.1001,timestamp=NOW.isoformat())
    async def symbol(self,pair):return dict(minVolume=100000,stepVolume=100000,maxVolume=1000000000)
    async def historical_bars(self,*a,**kw):return []
    async def close(self):pass


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        settings=SimpleNamespace(database_url='unused',ctrader_environment='demo',ctrader_demo_only=True)
        self.service=DeskService(settings,SimpleNamespace(api_ready=True,active_access_token='token'))
        await self.service.feed.close()
        self.service.feed=Feed();self.service.store=Store()

    async def test_cycle_opens_only_after_consensus_and_never_duplicates(self):
        cand=dict(id='x',pair='EUR/USD',timeframe=1,setup='breakout20',direction=1,
                  signal_close=1.1,stop=1.099,atr=.002,expires_at=(NOW+timedelta(seconds=90)).isoformat())
        self.service.agents.key='test-not-used'
        self.service.agents.decide=AsyncMock(return_value=dict(approved=True,rounds=[],candidate_id='x'))
        with patch('desk_service.datetime',Clock),patch('desk_service.candidates',return_value=[cand]):
            await self.service.cycle()
            self.assertEqual(self.service.store.state['entries'],1)
            self.assertIsNotNone(self.service.store.state['position'])
            await self.service.cycle()
            self.assertEqual(self.service.store.state['entries'],1)
        self.assertEqual([e[0] for e in self.service.store.events],['decision','trade_opened'])

    async def test_no_key_does_not_make_fake_trade(self):
        self.service.agents.key=''
        with patch('desk_service.datetime',Clock):await self.service.cycle()
        self.assertIsNone(self.service.store.state['position'])
        self.assertEqual(self.service.phase,'blocked')

    async def test_pause_will_still_manage_existing_position(self):
        from desk_core import roll_day,size_plan,open_position
        state=self.service.store.state;roll_day(state,NOW)
        c=dict(id='x',pair='EUR/USD',timeframe=1,setup='x',direction=1,signal_close=1.1,stop=1.099,atr=.002)
        p=size_plan(c,await self.service.feed.quote('EUR/USD'),1000,1.1,await self.service.feed.symbol('EUR/USD'))
        open_position(state,p,NOW,'x');state['enabled']=False
        with patch('desk_service.datetime',Clock):await self.service.cycle()
        self.assertIsNotNone(state['position'])
        self.assertEqual(self.service.phase,'managing')

    async def test_all_control_and_data_routes_require_auth(self):
        def auth(token):
            if token!='Bearer test-token':raise HTTPException(401,'Unauthorized')
        app=FastAPI();app.include_router(desk_router(self.service,auth))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as c:
            for method,path in [('GET','status'),('POST','start'),('POST','pause'),('GET','export')]:
                r=await c.request(method,'/api/desk/'+path)
                self.assertEqual(r.status_code,401)
            r=await c.get('/desk');self.assertEqual(r.status_code,200)
            r=await c.get('/api/desk/status',headers={'Authorization':'Bearer test-token'})
            self.assertEqual(r.status_code,200)
            self.assertEqual(r.json()['mode'],'forward-paper-only')
