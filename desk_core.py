"""Frozen v1 rules for the five-agent forward tester. No broker order methods."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone, time
from zoneinfo import ZoneInfo

VERSION = 'desk-m1-m5-v1'
ROLES = ('Scout', 'Charts', 'Risk', 'Sceptic', 'Chief')
UTC = timezone.utc


def dt(value):
    d = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('Timezone required')
    return d.astimezone(UTC)


def sessions(now):
    now = dt(now)
    active = []
    for label, tz in [('London', 'Europe/London'), ('New York', 'America/New_York')]:
        local = now.astimezone(ZoneInfo(tz))
        if local.weekday() < 5 and time(8) <= local.time().replace(tzinfo=None) < time(17):
            active.append(label)
    return active


def trade_day(now):
    return dt(now).astimezone(ZoneInfo('Europe/London')).date().isoformat()


def closed_bars(rows, now, minutes=1):
    cutoff = dt(now)
    out, seen = [], set()
    for r in sorted(rows, key=lambda x: dt(x['t'])):
        t = dt(r['t'])
        vals = [float(r[k]) for k in ('o', 'h', 'l', 'c')]
        if not all(math.isfinite(v) and v > 0 for v in vals):
            raise ValueError('Invalid OHLC')
        o, h, l, c = vals
        if h < max(o, c, l) or l > min(o, c, h):
            raise ValueError('Invalid OHLC bounds')
        if t + timedelta(minutes=minutes) <= cutoff and t not in seen:
            out.append(dict(r, t=t.isoformat(), o=o, h=h, l=l, c=c))
            seen.add(t)
    return out


def aggregate5(rows):
    groups = {}
    for b in rows:
        t = dt(b['t'])
        key = t.replace(minute=t.minute//5*5, second=0, microsecond=0)
        groups.setdefault(key, []).append(b)
    out = []
    for t, bs in sorted(groups.items()):
        if len(bs) != 5 or any(dt(b['t']) != t+timedelta(minutes=i) for i, b in enumerate(bs)):
            continue
        out.append(dict(t=t.isoformat(), o=bs[0]['o'], h=max(b['h'] for b in bs),
                        l=min(b['l'] for b in bs), c=bs[-1]['c']))
    return out


def ema(values, n):
    out = [float(values[0])]
    for v in values[1:]:
        out.append(out[-1] + 2/(n+1)*(float(v)-out[-1]))
    return out


def atr(rows, n=14):
    return sum(max(b['h']-b['l'], abs(b['h']-a['c']), abs(b['l']-a['c']))
               for a, b in zip(rows[-n-1:-1], rows[-n:]))/n


def contiguous(rows, minutes):
    return all(dt(b['t'])-dt(a['t']) == timedelta(minutes=minutes) for a,b in zip(rows,rows[1:]))


def candidates(pair, raw, now):
    """Only last closed bars; M5 trend plus M1 or M5 breakout/pullback."""
    m1 = closed_bars(raw, now)
    m5 = aggregate5(m1)
    if len(m1)<65 or len(m5)<65:
        return []
    if not contiguous(m1[-65:],1) or not contiguous(m5[-65:],5):
        return []
    if dt(now)-dt(m1[-1]['t']) > timedelta(seconds=95):
        return []
    trend = ema([b['c'] for b in m5],50)
    direction = 1 if m5[-1]['c']>trend[-1]>trend[-4] else -1 if m5[-1]['c']<trend[-1]<trend[-4] else 0
    if not direction:
        return []
    out=[]
    for tf, bars in [(1,m1),(5,m5)]:
        b, prev=bars[-1],bars[-21:-1]
        # Avoid repeated M5 evaluation on each of its following M1 bars.
        if tf == 5 and dt(m1[-1]['t'])+timedelta(minutes=1) != dt(b['t'])+timedelta(minutes=5):
            continue
        a=atr(bars)
        if a<=0:
            continue
        e=ema([x['c'] for x in bars],20)[-1]
        breakout=(b['c']>max(x['h'] for x in prev)) if direction==1 else (b['c']<min(x['l'] for x in prev))
        pullback=(b['l']<=e<b['c'] and b['c']>b['o']) if direction==1 else (b['h']>=e>b['c'] and b['c']<b['o'])
        for name, yes in [('breakout20',breakout),('ema20_pullback',pullback)]:
            if not yes:
                continue
            stop=(min(x['l'] for x in bars[-3:])-0.1*a) if direction==1 else (max(x['h'] for x in bars[-3:])+0.1*a)
            distance=direction*(b['c']-stop)
            if not 0.7*a <= distance <= 3*a:
                continue
            key=f'{VERSION}|{pair}|{tf}|{name}|{b["t"]}|{direction}'
            out.append(dict(id=hashlib.sha256(key.encode()).hexdigest()[:24], pair=pair,
                            timeframe=tf, setup=name, direction=direction, signal_at=b['t'],
                            signal_close=b['c'], expires_at=(dt(b['t'])+timedelta(minutes=tf,seconds=90)).isoformat(),
                            stop=stop, atr=a, trend='up' if direction==1 else 'down',
                            evidence={'ema20':e,'ema50_m5':trend[-1],'last_bars':bars[-12:]}))
    return out


def initial_state(balance=50.):
    return dict(version=VERSION, balance=balance, enabled=False, day='', entries=0,
                losses=0, winning_trade=False, day_start=balance, position=None,
                last_seen={}, debated=[], debate_count=0, decisions=[], trades=[], message='Nog niet gestart')


def roll_day(state, now):
    day=trade_day(now)
    if day != state['day']:
        if state['position']:
            return  # Old positions must be settled before resetting risk counters.
        state.update(day=day, entries=0, losses=0, winning_trade=False, day_start=state['balance'],
                     debate_count=0, debated=[])


def can_enter(state, now):
    return (state['enabled'] and bool(sessions(now)) and state['day']==trade_day(now)
            and not state['position'] and state['entries']<2 and state['losses']<2
            and not state['winning_trade'] and state['balance']>0)


@dataclass(frozen=True)
class Costs:
    # Explicit modelling assumptions, not a claim about this account's tariff.
    commission_usd_per_unit: float = 4.5/100000
    slippage_pips: float = 0.2
    max_spread_pips: float = 1.5
    leverage: float = 30.
    target_r: float = 2.
    risk_fraction: float = .10

    def __post_init__(self):
        values=asdict(self)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError('Finite costs required')
        if not 0<self.risk_fraction<=.10 or self.target_r<2 or self.leverage<=0:
            raise ValueError('Invalid risk/RR/leverage')
        if min(self.commission_usd_per_unit,self.slippage_pips,self.max_spread_pips)<0:
            raise ValueError('Negative cost')


def size_plan(candidate, quote, balance, eurusd, symbol, costs=Costs()):
    if candidate['pair'] not in ('EUR/USD','GBP/USD'):
        raise ValueError('v1 supports USD quote currency only')
    bid,ask=float(quote['bid']),float(quote['ask'])
    if not all(math.isfinite(v) and v>0 for v in (bid,ask,balance,eurusd)) or ask<bid:
        raise ValueError('Invalid quote/balance/conversion')
    if (ask-bid)/.0001>costs.max_spread_pips:
        raise ValueError('Spread too wide')
    side=candidate['direction']; slip=costs.slippage_pips*.0001
    entry=(ask+slip) if side==1 else (bid-slip)
    distance=side*(entry-candidate['stop'])
    if distance<=0 or abs(entry-candidate['signal_close'])>0.5*candidate['atr']:
        raise ValueError('Setup moved/invalidated during debate')
    fee=costs.commission_usd_per_unit
    risk_per_unit=distance+slip+fee
    risk_budget=balance*costs.risk_fraction
    max_units_risk=risk_budget*eurusd/risk_per_unit
    max_units_margin=balance*.95*costs.leverage*eurusd/entry
    minimum=float(symbol['minVolume'])/100
    step=float(symbol['stepVolume'])/100
    maximum=float(symbol['maxVolume'])/100
    if not (minimum>0 and step>0 and maximum>=minimum):
        raise ValueError('Invalid broker volume limits')
    cap=min(max_units_risk,max_units_margin,maximum)
    units=minimum+math.floor((cap-minimum)/step)*step
    if units<minimum:
        raise ValueError('Balance/risk too small for broker minimum volume')
    risk=units*risk_per_unit/eurusd
    # Net TP is >=2R AFTER the modelled entry/exit slippage and commission.
    target=entry+side*(costs.target_r*risk_per_unit+fee+slip)
    return dict(candidate_id=candidate['id'], pair=candidate['pair'], direction=side,
                timeframe=candidate['timeframe'], setup=candidate['setup'], entry=entry,
                stop=candidate['stop'], target=target, units=units, risk_eur=risk,
                risk_fraction=risk/balance, requested_risk_fraction=costs.risk_fraction,
                risk_per_unit=risk_per_unit, eurusd_at_entry=eurusd,
                costs=asdict(costs), target_r=costs.target_r)


def open_position(state, plan, now, candidate_id):
    if not can_enter(state,now) or candidate_id!=plan['candidate_id']:
        raise ValueError('Daily risk gate or candidate mismatch')
    if plan['risk_eur']>state['balance']*.10+1e-8 or plan['target_r']<2:
        raise ValueError('Risk/RR gate')
    state['position']=dict(plan, opened_at=dt(now).isoformat(), day=state['day'], last_bar=None)
    state['entries']+=1


def settle(state, exit_price, now, reason, eurusd):
    p=state['position']; side=p['direction']
    fee=p['costs']['commission_usd_per_unit']
    pnl=(side*(exit_price-p['entry'])-fee)*p['units']/eurusd
    r=pnl/p['risk_eur']
    trade=dict(p,exit_price=exit_price,closed_at=dt(now).isoformat(),reason=reason,
               pnl_eur=pnl,r=r,balance_before=state['balance'],balance_after=state['balance']+pnl)
    state['balance']+=pnl
    state['trades']=(state['trades']+[trade])[-200:]
    state['losses']+=int(pnl < -1e-8)
    # A hit of the >=2R net-designed TP ends the day, even if FX conversion moved.
    state['winning_trade'] |= reason=='target' or r>=2-1e-8
    state['position']=None
    return trade


def reconcile_bars(state, rows, spread, now, eurusd):
    """Conservative M1 OHLC paper fills. Stop first if both levels touched."""
    p=state['position']
    if not p:
        return None
    side=p['direction']; slip=p['costs']['slippage_pips']*.0001
    for b in closed_bars(rows,now):
        t=dt(b['t'])
        entry_minute=dt(p['opened_at']).replace(second=0,microsecond=0)
        if t<entry_minute or (p['last_bar'] and t<=dt(p['last_bar'])):
            continue
        shift=0 if side==1 else spread
        op,hi,lo=(b[k]+shift for k in ('o','h','l'))
        stop_hit=lo<=p['stop'] if side==1 else hi>=p['stop']
        target_hit=hi>=p['target'] if side==1 else lo<=p['target']
        # The entry candle includes pre-entry prices. Never award a historical
        # target on that candle. Conservatively book an adverse touch and flag it.
        if t==entry_minute and t<dt(p['opened_at']):
            if stop_hit:
                px=p['stop']-side*slip
                return settle(state,px,t+timedelta(minutes=1),'stop_entry_ambiguous',eurusd)
            p['last_bar']=b['t']
            continue
        if stop_hit:
            px=min(op,p['stop'])-slip if side==1 else max(op,p['stop'])+slip
            return settle(state,px,t+timedelta(minutes=1),'stop_ambiguous' if target_hit else 'stop',eurusd)
        if target_hit:
            return settle(state,p['target']-side*slip,t+timedelta(minutes=1),'target',eurusd)
        p['last_bar']=b['t']
    return None


def reconcile_quote(state,quote,now,eurusd,force=False):
    p=state['position']
    if not p:
        return None
    side=p['direction']; px=quote['bid'] if side==1 else quote['ask']
    stop=px<=p['stop'] if side==1 else px>=p['stop']
    target=px>=p['target'] if side==1 else px<=p['target']
    if stop or target or force:
        # TP uses its limit; stop/session close may gap beyond the planned 1R loss.
        fill=(p['target'] if target and not stop else px)-side*p['costs']['slippage_pips']*.0001
        return settle(state,fill,now,'stop' if stop else 'target' if target else 'session_close',eurusd)
    return None
