import asyncio
import copy
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from desk_core import (sessions,trade_day,initial_state,roll_day,can_enter,Costs,size_plan,
                       open_position,reconcile_quote,reconcile_bars,dt,candidates,aggregate5,ROLES)
from desk_agents import AgentDesk

NOW=dt('2026-11-02T10:00:00Z')
SYMBOL={'minVolume':100000,'stepVolume':100000,'maxVolume':1000000000}


def candidate(side=1):
    return dict(id='test-candidate',pair='EUR/USD',timeframe=1,setup='test',direction=side,
                signal_at=NOW.isoformat(),signal_close=1.1,atr=.002,
                stop=1.099 if side==1 else 1.101,expires_at=(NOW+timedelta(seconds=90)).isoformat())


def ready(balance=1000):
    s=initial_state(balance);s['enabled']=True;roll_day(s,NOW);return s


def plan(side=1,balance=1000):
    return size_plan(candidate(side),dict(bid=1.1,ask=1.1001),balance,1.1,SYMBOL)


class DeskCoreTests(unittest.TestCase):
    def test_sessions_follow_local_dst_in_mismatched_weeks(self):
        self.assertIn('London',sessions('2026-07-06T07:00:00Z'))
        self.assertNotIn('London',sessions('2026-11-02T07:59:00Z'))
        self.assertIn('London',sessions('2026-11-02T08:00:00Z'))
        self.assertIn('New York',sessions('2026-10-26T12:00:00Z'))
        self.assertNotIn('New York',sessions('2026-11-02T12:59:00Z'))
        self.assertIn('New York',sessions('2026-11-02T13:00:00Z'))
        self.assertFalse(sessions('2026-11-01T14:00:00Z'))

    def test_net_target_is_two_r_long_and_short(self):
        for side in (1,-1):
            s=ready();p=plan(side);open_position(s,p,NOW,p['candidate_id'])
            t=reconcile_quote(s,dict(bid=p['target'],ask=p['target']),NOW+timedelta(minutes=2),1.1)
            self.assertAlmostEqual(t['r'],2.,places=8)
            self.assertTrue(s['winning_trade'])
            self.assertFalse(can_enter(s,NOW+timedelta(hours=5)))

    def test_two_losses_stop_both_sessions_and_survive_restart(self):
        s=ready()
        for i in range(2):
            p=plan(balance=s['balance']);open_position(s,p,NOW+timedelta(minutes=i*2),p['candidate_id'])
            t=reconcile_quote(s,dict(bid=p['stop'],ask=p['stop']),NOW+timedelta(minutes=i*2+1),1.1)
            self.assertAlmostEqual(t['r'],-1,places=8)
            s=json.loads(json.dumps(s))
        self.assertEqual(s['losses'],2)
        self.assertFalse(can_enter(s,NOW+timedelta(hours=5)))
        self.assertLessEqual(s['entries'],2)
        self.assertGreaterEqual(s['balance'],810-1e-8)
        roll_day(s,NOW+timedelta(days=1))
        self.assertTrue(can_enter(s,NOW+timedelta(days=1)))

    def test_loss_then_win_stops_on_winning_trade_not_daily_net_two_r(self):
        s=ready();p=plan();open_position(s,p,NOW,p['candidate_id'])
        reconcile_quote(s,dict(bid=p['stop'],ask=p['stop']),NOW+timedelta(minutes=1),1.1)
        self.assertTrue(can_enter(s,NOW+timedelta(minutes=2)))
        p=plan(balance=s['balance']);open_position(s,p,NOW+timedelta(minutes=2),p['candidate_id'])
        reconcile_quote(s,dict(bid=p['target'],ask=p['target']),NOW+timedelta(minutes=3),1.1)
        self.assertFalse(can_enter(s,NOW+timedelta(hours=4)))

    def test_one_open_position_and_risk_cap(self):
        s=ready();p=plan();self.assertLessEqual(p['risk_eur'],100)
        open_position(s,p,NOW,p['candidate_id'])
        with self.assertRaises(ValueError):open_position(s,p,NOW,p['candidate_id'])
        s=ready();p['risk_eur']=101
        with self.assertRaises(ValueError):open_position(s,p,NOW,p['candidate_id'])

    def test_broker_minimum_and_margin_are_respected(self):
        p=plan(balance=50)
        self.assertEqual(p['units']%1000,0)
        self.assertLessEqual(p['risk_eur'],5)
        self.assertLessEqual(p['units']*p['entry']/1.1/30,50*.95)
        with self.assertRaises(ValueError):plan(balance=1)

    def test_wide_spread_or_invalid_price_blocks(self):
        for q in [dict(bid=1.1,ask=1.1005),dict(bid=float('nan'),ask=1.1),dict(bid=1.2,ask=1.1)]:
            with self.assertRaises(ValueError):size_plan(candidate(),q,1000,1.1,SYMBOL)

    def test_gap_can_exceed_one_r_and_is_not_hidden(self):
        s=ready();p=plan();open_position(s,p,NOW,p['candidate_id'])
        t=reconcile_quote(s,dict(bid=p['stop']-.005,ask=p['stop']-.0049),NOW+timedelta(minutes=1),1.1)
        self.assertLess(t['r'],-1.)

    def test_ambiguous_candle_resolves_stop_first_and_is_idempotent(self):
        s=ready();p=plan();open_position(s,p,NOW,p['candidate_id'])
        b=dict(t=(NOW+timedelta(minutes=1)).isoformat(),o=p['entry'],h=p['target']+.001,l=p['stop']-.001,c=p['entry'])
        t=reconcile_bars(s,[b],.0001,NOW+timedelta(minutes=2),1.1)
        self.assertEqual(t['reason'],'stop_ambiguous');self.assertLess(t['r'],0)
        self.assertIsNone(reconcile_bars(s,[b],.0001,NOW+timedelta(minutes=2),1.1))
        self.assertEqual(len(s['trades']),1)

    def test_entry_candle_cannot_award_pre_entry_target(self):
        s=ready();p=plan();open_position(s,p,NOW+timedelta(seconds=15),p['candidate_id'])
        b=dict(t=NOW.isoformat(),o=p['entry'],h=p['target']+.001,l=p['entry'],c=p['entry'])
        self.assertIsNone(reconcile_bars(s,[b],.0001,NOW+timedelta(minutes=1),1.1))
        b['l']=p['stop']-.001;s['position']['last_bar']=None
        t=reconcile_bars(s,[b],.0001,NOW+timedelta(minutes=1),1.1)
        self.assertEqual(t['reason'],'stop_entry_ambiguous')

    def test_day_rollover_does_not_erase_open_trade_limits(self):
        s=ready();p=plan();open_position(s,p,NOW,p['candidate_id'])
        roll_day(s,NOW+timedelta(days=1));self.assertEqual(s['entries'],1)
        self.assertEqual(s['day'],trade_day(NOW))

    def test_no_future_dependency_and_no_incomplete_m5(self):
        rows=[]
        for i in range(400):
            t=NOW-timedelta(minutes=400-i)
            price=1.08+i*.00002
            rows.append(dict(t=t.isoformat(),o=price,h=price+.000015,l=price-.000015,c=price+.00001))
        # Last candle is a breakout within the ATR distance bounds.
        rows[-1]['c']+=.00002;rows[-1]['h']=rows[-1]['c']+.000005
        before=candidates('EUR/USD',rows,NOW)
        future=dict(t=NOW.isoformat(),o=2,h=3,l=1,c=2)
        self.assertEqual(before,candidates('EUR/USD',rows+[future],NOW))
        self.assertTrue(before,'fixture should produce an actionable setup')
        self.assertEqual(aggregate5(rows[-4:]),[])

    def test_rr_cannot_be_configured_below_two(self):
        for kw in [dict(target_r=1.9),dict(risk_fraction=.11),dict(slippage_pips=-1)]:
            with self.assertRaises(ValueError):Costs(**kw)


class AgentConsensusTests(unittest.IsolatedAsyncioTestCase):
    async def test_five_independent_votes_then_shared_round(self):
        desk=AgentDesk();seen=[]
        async def vote(role,snapshot,previous=None):
            seen.append((role,previous))
            return dict(role=role,candidate_id='x',approve=True,reason='test',concerns=[])
        desk.vote=vote
        decision=await desk.decide({'candidate':{'id':'x'}})
        self.assertTrue(decision['approved']);self.assertEqual(len(seen),10)
        self.assertEqual({r for r,p in seen[:5]},set(ROLES))
        self.assertTrue(all(p is None for r,p in seen[:5]))
        self.assertTrue(all(len(p)==5 for r,p in seen[5:]))

    async def test_single_veto_or_provider_failure_blocks_consensus(self):
        for failure in (False,True):
            desk=AgentDesk()
            async def vote(role,snapshot,previous=None):
                if role=='Risk' and failure:raise TimeoutError()
                return dict(role=role,candidate_id='x',approve=role!='Risk',reason='test',concerns=[])
            desk.vote=vote
            decision=await desk.decide({'candidate':{'id':'x'}})
            self.assertFalse(decision['approved'])

    async def test_missing_api_key_never_falls_back_to_fake_votes(self):
        desk=AgentDesk();desk.key=''
        decision=await desk.decide({'candidate':{'id':'x'}})
        self.assertFalse(decision['approved'])
        self.assertTrue(all('agent_error' in v['concerns'] for v in decision['rounds'][-1]))
