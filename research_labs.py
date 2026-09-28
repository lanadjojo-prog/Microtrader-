from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from statistics import mean, median
from typing import Dict, List, Optional

from alpaca_client import AlpacaClient
from config import Settings
from research_store import ResearchStore
from strategy_lab import Candidate, candidate_signature, metrics, simulate

log = logging.getLogger("microtrader.research_labs")


@dataclass
class ResearchState:
    running: bool = False
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    current_lab: str = ""
    completed_labs: int = 0
    total_labs: int = 0
    last_error: Optional[str] = None


LABS = [
    "market", "session", "regime", "high_frequency",
    "walk_forward", "parameter_stability", "cost_stress",
    "monte_carlo", "position_sizing", "compounding",
    "leverage", "risk_of_ruin", "recovery",
    "portfolio", "capital_allocation", "aggressive_growth", "master"
]


class ResearchLabs:
    def __init__(self, settings: Settings, client: AlpacaClient):
        self.settings=settings
        self.client=client
        self.store=ResearchStore(settings.database_url)
        self.state=ResearchState(total_labs=len(LABS))
        self._task: Optional[asyncio.Task]=None
        self._results: Dict[str,dict]={}

    def public_state(self):
        p=asdict(self.state)
        p["labs"]={k:v for k,v in self._results.items()}
        return p

    async def start(self):
        if self.state.running:
            return
        self.state=ResearchState(running=True,started_at=datetime.now(timezone.utc).isoformat(),total_labs=len(LABS))
        self._task=asyncio.create_task(self._run(),name="microtrader-research-labs")

    async def stop(self):
        if self._task and not self._task.done():
            self._task.cancel()
            try: await self._task
            except asyncio.CancelledError: pass
        self.state.running=False

    async def _run(self):
        try:
            await self.store.init()
            end=datetime.now(timezone.utc)
            start=end-timedelta(days=max(120,self.settings.lab_lookback_days))
            bars_by_symbol={}
            for symbol in self.settings.lab_symbols:
                bars=await self.client.historical_bars(symbol,start,end,self.settings.lab_timeframe,self.settings.lab_max_bars_per_symbol)
                if len(bars)>=300:
                    bars_by_symbol[symbol]=bars

            from strategy_store import StrategyStore
            ss=StrategyStore(self.settings.database_url)
            rows=await ss.load_results(limit=25)
            candidates=[]
            for row in rows:
                try:
                    candidates.append(Candidate(str(row["strategy"]),dict(row["params"])))
                except Exception:
                    continue
            if not candidates:
                raise RuntimeError("No Strategy Lab candidates available yet")

            # favor stronger candidates but still test multiple distinct configs
            seen=set(); uniq=[]
            for c in candidates:
                sig=candidate_signature(c)
                if sig not in seen:
                    seen.add(sig); uniq.append(c)
            candidates=uniq[:8]

            for name in LABS[:-1]:
                self.state.current_lab=name
                fn=getattr(self,f"_lab_{name}")
                result=await asyncio.to_thread(fn,candidates,bars_by_symbol)
                self._results[name]=result
                await self.store.save(name,"completed",result)
                self.state.completed_labs+=1
                await asyncio.sleep(0)

            self.state.current_lab="master"
            master=self._lab_master(candidates,bars_by_symbol)
            self._results["master"]=master
            await self.store.save("master","completed",master)
            self.state.completed_labs+=1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.state.last_error=str(exc)
            log.exception("Research labs failed")
        finally:
            self.state.running=False
            self.state.completed_at=datetime.now(timezone.utc).isoformat()

    def _candidate_trades(self,c,bars_by_symbol,cost=None):
        cost=self.settings.lab_cost_bps if cost is None else cost
        out=[]
        for symbol,bars in bars_by_symbol.items():
            split=max(2,int(len(bars)*0.70))
            out.extend(simulate(c,symbol,bars[split:],cost))
        return sorted(out,key=lambda x:str(x.get("exit_time") or ""))

    def _score_candidates(self,candidates,bars_by_symbol):
        rows=[]
        for c in candidates:
            t=self._candidate_trades(c,bars_by_symbol)
            m=metrics(t)
            rows.append({"strategy":c.strategy,"params":c.params,"metrics":m,"trades":t})
        rows.sort(key=lambda r:(r["metrics"]["profit_factor"],r["metrics"]["expectancy_bps"]),reverse=True)
        return rows

    def _lab_market(self,candidates,bars):
        best=self._score_candidates(candidates,bars)[0]
        per={}
        c=Candidate(best["strategy"],best["params"])
        for s,b in bars.items():
            split=int(len(b)*.7)
            per[s]=metrics(simulate(c,s,b[split:],self.settings.lab_cost_bps))
        return {"candidate":best["strategy"],"params":best["params"],"per_symbol":per}

    def _lab_session(self,candidates,bars):
        best=self._score_candidates(candidates,bars)[0]
        buckets={"open":[],"midday":[],"close":[],"other":[]}
        for t in best["trades"]:
            try:
                hour=datetime.fromisoformat(str(t["entry_time"]).replace("Z","+00:00")).hour
            except Exception:
                hour=-1
            if 13<=hour<15: k="open"
            elif 15<=hour<18: k="midday"
            elif 18<=hour<21: k="close"
            else: k="other"
            buckets[k].append(t)
        return {"candidate":best["strategy"],"sessions":{k:metrics(v) for k,v in buckets.items()}}

    def _lab_regime(self,candidates,bars):
        best=self._score_candidates(candidates,bars)[0]
        c=Candidate(best["strategy"],best["params"])
        out={}
        for s,b in bars.items():
            n=len(b); chunks=[b[:n//3],b[n//3:2*n//3],b[2*n//3:]]
            out[s]=[metrics(simulate(c,s,x,self.settings.lab_cost_bps)) for x in chunks if len(x)>50]
        return {"candidate":best["strategy"],"chronological_regimes":out}

    def _lab_high_frequency(self,candidates,bars):
        rows=self._score_candidates(candidates,bars)
        out=[]
        for r in rows:
            t=r["trades"]
            days=max(1,len(set(str(x.get("entry_time",""))[:10] for x in t)))
            out.append({"strategy":r["strategy"],"params":r["params"],"trades_per_day":round(len(t)/days,2),
                        "expectancy_bps":r["metrics"]["expectancy_bps"],"profit_factor":r["metrics"]["profit_factor"]})
        return {"ranking":out}

    def _lab_walk_forward(self,candidates,bars):
        out=[]
        for c in candidates:
            windows=[]
            for wi in range(4):
                tr=[]
                for s,b in bars.items():
                    n=len(b); a=int(n*wi/8); z=int(n*(wi+4)/8)
                    chunk=b[a:z]
                    if len(chunk)<100: continue
                    split=int(len(chunk)*.7)
                    tr.extend(simulate(c,s,chunk[split:],self.settings.lab_cost_bps))
                windows.append(metrics(tr))
            positive=sum(1 for m in windows if m["expectancy_bps"]>0 and m["profit_factor"]>=1)
            out.append({"strategy":c.strategy,"params":c.params,"positive_windows":positive,"windows":windows})
        return {"candidates":out}

    def _neighbors(self,c):
        p=dict(c.params); out=[]
        for k,v in list(p.items()):
            if isinstance(v,(int,float)) and k not in {"max_hold"}:
                for mult in (.9,1.1):
                    q=dict(p); nv=v*mult
                    q[k]=int(round(nv)) if isinstance(v,int) else round(nv,4)
                    if q[k]!=v: out.append(Candidate(c.strategy,q))
        return out[:8]

    def _lab_parameter_stability(self,candidates,bars):
        rows=[]
        for c in candidates[:5]:
            ms=[]
            for n in self._neighbors(c):
                try: ms.append(metrics(self._candidate_trades(n,bars)))
                except Exception: pass
            pos=sum(1 for m in ms if m["expectancy_bps"]>0)
            rows.append({"strategy":c.strategy,"params":c.params,"neighbors_tested":len(ms),"positive_neighbors":pos,
                         "stability_ratio":round(pos/max(1,len(ms)),3)})
        return {"candidates":rows}

    def _lab_cost_stress(self,candidates,bars):
        best=self._score_candidates(candidates,bars)[0]
        c=Candidate(best["strategy"],best["params"])
        return {"strategy":c.strategy,"params":c.params,
                "cost_scenarios":{str(x):metrics(self._candidate_trades(c,bars,self.settings.lab_cost_bps*x))
                                  for x in (1.0,1.5,2.0,3.0)}}

    def _equity_stats(self,returns,fraction=1.0,leverage=1.0,start=50.0):
        eq=start; peak=start; maxdd=0.0
        for r in returns:
            eq*=max(0.0,1+r*fraction*leverage)
            peak=max(peak,eq)
            maxdd=max(maxdd,0 if peak<=0 else (peak-eq)/peak)
            if eq<=0: break
        return {"ending":round(eq,2),"multiple":round(eq/start,3) if start else 0,"max_drawdown_pct":round(maxdd*100,2)}

    def _lab_monte_carlo(self,candidates,bars):
        best=self._score_candidates(candidates,bars)[0]
        returns=[float(t["net_return"]) for t in best["trades"]]
        rng=random.Random(42); endings=[]; dds=[]
        for _ in range(500):
            x=list(returns); rng.shuffle(x); st=self._equity_stats(x)
            endings.append(st["ending"]); dds.append(st["max_drawdown_pct"])
        return {"candidate":best["strategy"],"runs":500,"median_ending":round(median(endings),2),
                "p10_ending":round(sorted(endings)[max(0,int(.1*len(endings))-1)],2),
                "median_max_drawdown_pct":round(median(dds),2)}

    def _lab_position_sizing(self,candidates,bars):
        best=self._score_candidates(candidates,bars)[0]; rr=[float(t["net_return"]) for t in best["trades"]]
        return {"candidate":best["strategy"],"sizes":{str(f):self._equity_stats(rr,fraction=f) for f in (.01,.02,.05,.1,.2,.5,1.0)}}

    def _lab_compounding(self,candidates,bars):
        best=self._score_candidates(candidates,bars)[0]; rr=[float(t["net_return"]) for t in best["trades"]]
        paths={}
        for f in (.1,.25,.5,1.0):
            eq=50.0; reached={}
            for i,r in enumerate(rr,1):
                eq*=max(0.0,1+r*f)
                for target in (100,250,500,1000,5000):
                    if target not in reached and eq>=target: reached[target]=i
            paths[str(f)]={"ending":round(eq,2),"milestone_trade":reached}
        return {"candidate":best["strategy"],"start":50,"paths":paths}

    def _lab_leverage(self,candidates,bars):
        best=self._score_candidates(candidates,bars)[0]; rr=[float(t["net_return"]) for t in best["trades"]]
        return {"candidate":best["strategy"],"leverage":{str(x):self._equity_stats(rr,1.0,x) for x in (1,1.5,2,3,5)}}

    def _lab_risk_of_ruin(self,candidates,bars):
        best=self._score_candidates(candidates,bars)[0]; base=[float(t["net_return"]) for t in best["trades"]]
        rng=random.Random(7); out={}
        for f in (.1,.25,.5,1.0):
            ruins=0; deep=0
            for _ in range(500):
                x=list(base); rng.shuffle(x); eq=50; peak=50
                for r in x:
                    eq*=max(0.0,1+r*f)
                    peak=max(peak,eq)
                    if eq<=5: ruins+=1; break
                    if eq<=25: deep+=1; break
            out[str(f)]={"ruin_le_5_pct":round(ruins/5,1),"drawdown_to_25_pct":round(deep/5,1)}
        return {"candidate":best["strategy"],"simulations_per_size":500,"risk":out}

    def _lab_recovery(self,candidates,bars):
        best=self._score_candidates(candidates,bars)[0]; rr=[float(t["net_return"]) for t in best["trades"]]
        eq=peak=1.0; peak_i=0; worst=0; max_recovery=0
        for i,r in enumerate(rr):
            eq*=max(.000001,1+r)
            if eq>=peak:
                max_recovery=max(max_recovery,i-peak_i); peak=eq; peak_i=i
            else:
                worst=max(worst,(peak-eq)/peak)
        return {"candidate":best["strategy"],"max_drawdown_pct":round(worst*100,2),"max_recovery_trades":max_recovery}

    def _lab_portfolio(self,candidates,bars):
        ranked=self._score_candidates(candidates,bars)[:4]
        combined=[]
        for r in ranked:
            for t in r["trades"]:
                x=dict(t); x["net_return"]=float(x["net_return"])/len(ranked); combined.append(x)
        return {"components":[{"strategy":r["strategy"],"params":r["params"]} for r in ranked],"equal_weight_metrics":metrics(combined)}

    def _lab_capital_allocation(self,candidates,bars):
        ranked=self._score_candidates(candidates,bars)[:3]
        if not ranked: return {}
        pfs=[max(.01,r["metrics"]["profit_factor"]) for r in ranked]; total=sum(pfs)
        return {"allocation":[{"strategy":r["strategy"],"params":r["params"],"weight_pct":round(100*pf/total,1)}
                              for r,pf in zip(ranked,pfs)]}

    def _lab_aggressive_growth(self,candidates,bars):
        ranked=self._score_candidates(candidates,bars)[:3]
        scenarios=[]
        for r in ranked:
            rr=[float(t["net_return"]) for t in r["trades"]]
            for f in (.25,.5,.75,1.0):
                for lev in (1,1.5,2,3):
                    st=self._equity_stats(rr,f,lev)
                    scenarios.append({"strategy":r["strategy"],"params":r["params"],"fraction":f,"leverage":lev,**st})
        scenarios.sort(key=lambda x:x["ending"],reverse=True)
        return {"start":50,"top_scenarios":scenarios[:20],"note":"Historical path only; not a forecast."}

    def _lab_master(self,candidates,bars):
        ranked=self._score_candidates(candidates,bars)
        return {
            "objective":"fast capital growth with explicit robustness and ruin controls",
            "candidate_count":len(ranked),
            "best_by_profit_factor":{"strategy":ranked[0]["strategy"],"params":ranked[0]["params"],"metrics":ranked[0]["metrics"]} if ranked else None,
            "labs_completed":LABS[:-1],
            "warning":"Research output only. Promotion to live trading remains disabled."
        }
