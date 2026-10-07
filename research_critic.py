"""Independent, deterministic research review. No trading or approval authority."""
from collections import Counter
from adaptive_router import ADAPTIVE_CONTEXT_VERSION


def review_research(results, recent, *, route_count=0):
    # Final holdout is verdict-only; never use it to design the next strategy.
    evidence = [r for r in results if (r.get('params') or {}).get('_phase') in {'discovery', 'incubator'}]
    observations = [r for r in recent if (r.get('params') or {}).get('_phase') in {'discovery', 'incubator'}]
    reasons = Counter(reason for r in observations for reason in r.get('rejection_reasons', []))
    zero = sum(int((r.get('oos') or {}).get('trades') or 0) == 0 for r in observations)
    evidence_rows = 0
    contexts = {}
    for row in evidence:
        diag = row.get('adaptive_diagnostics') or {}
        if diag.get('context_version') != ADAPTIVE_CONTEXT_VERSION:
            continue
        breakdown = diag.get('context_entry_breakdown') or {}
        evidence_rows += bool(breakdown)
        # Preserve exact timeframe / entry / context linkage. Repeated tests are
        # alternatives, not independent samples; do not add their trade counts.
        for context, met in breakdown.items():
            n = int(met.get('trades') or 0)
            exp = float(met.get('expectancy_bps') or 0)
            stress = (diag.get('stress_context_entry_breakdown') or {}).get(context) or {}
            if n < 30 or exp <= 0 or float(met.get('profit_factor') or 0) < 1.05:
                continue
            if int(stress.get('trades') or 0) < 15 or float(stress.get('expectancy_bps') or 0) <= 0:
                continue
            tf = int((row.get('params') or {}).get('timeframe_min') or 0)
            key = (tf, context, row.get('strategy'))
            proposal = {
                'family': row.get('strategy'), 'timeframe_min': tf,
                'context': context, 'trades': n, 'expectancy_bps': exp,
                'stress_expectancy_bps': float(stress['expectancy_bps']),
                'status': 'hypothesis_only',
                'thesis': f"Test {row.get('strategy')} op {tf}m alleen binnen {context}; buiten deze context geen instap.",
                'rationale': f"{n} onderzoeks-trades: netto {exp:.3f} bps/trade; kostenstress {float(stress['expectancy_bps']):.3f} bps/trade.",
                'falsification': 'Verwerp bij niet-positieve netto expectancy in nieuwe, chronologische validatie of kostenstress; bevries de regels vooraf.',
            }
            old = contexts.get(key)
            if old is None or proposal['stress_expectancy_bps'] > old['stress_expectancy_bps']:
                contexts[key] = proposal
    hypotheses = sorted(contexts.values(), key=lambda x: (x['stress_expectancy_bps'], x['trades']), reverse=True)[:5]
    stalled = len(observations) >= 20 and zero / len(observations) >= .8
    if stalled:
        status = 'no_signal'
        message = f'{zero}/{len(observations)} recente tests zonder OOS-trades. Zoek breder binnen begrensde instapdrempels; er is nog geen bewijs voor exploitatie.'
    elif not evidence_rows:
        status = 'missing_context_evidence'
        message = 'Nog geen bruikbare contextmetingen. Verzamel eerst onderzoeks-trades voordat handelsregels worden samengesteld.'
    elif not route_count:
        status = 'no_robust_routes'
        message = f'{evidence_rows} contextmetingen beschikbaar, maar nog geen route door de selectie. Bekijk steekproefgrootte, netto resultaat en kostenstress.'
    else:
        status = 'awaiting_validation'
        message = f'{route_count} routes beschikbaar voor onderzoek; paper vereist nog validatie van de complete bevroren strategie.'
    return {
        'version': 'research-critic-v1', 'method': 'deterministic_evidence_review',
        'status': status, 'message': message,
        'recent_sample_size': len(observations), 'recent_zero_trade_results': zero,
        'context_evidence_rows': evidence_rows, 'route_count': route_count,
        'rejection_reasons': dict(reasons.most_common(8)), 'hypotheses': hypotheses,
        'steering': 'bounded_exploration' if stalled or not evidence_rows else 'evidence_focus',
        'final_holdout_excluded': True, 'can_approve_paper': False,
        'can_place_orders': False, 'can_relax_validation': False,
        'evidence_limit': 'Geselecteerde onderzoeksresultaten zijn geen onafhankelijk bewijs van winstgevendheid. Forward paper-resultaten zijn nog nodig.',
    }
