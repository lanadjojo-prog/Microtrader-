import unittest
from adaptive_router import ADAPTIVE_CONTEXT_VERSION
from research_critic import review_research
from strategy_lab import choose_batch, discovery_candidates, candidate_signature, prune_research_results


def evidence(phase='discovery'):
    return {'strategy': 'breakout', 'params': {'_phase': phase, 'timeframe_min': 1},
            'funnel_stage': 'rejected', 'funnel_score': 10,
            'oos': {'trades': 45, 'expectancy_bps': -.1},
            'adaptive_diagnostics': {'context_version': ADAPTIVE_CONTEXT_VERSION,
                'context_entry_breakdown': {'trend|London||breakout': {'trades': 40, 'expectancy_bps': 1, 'profit_factor': 1.3}},
                'stress_context_entry_breakdown': {'trend|London||breakout': {'trades': 40, 'expectancy_bps': .3}}}}


class CriticTests(unittest.TestCase):
    def test_zero_trade_stall_uses_recent_not_selected_winners(self):
        recent = [{'params': {'_phase': 'discovery'}, 'oos': {'trades': 0}} for _ in range(20)]
        report = review_research([evidence()], recent)
        self.assertEqual(report['status'], 'no_signal')
        self.assertEqual(report['steering'], 'bounded_exploration')
        self.assertFalse(report['can_approve_paper'])

    def test_holdout_is_not_recycled_into_hypotheses(self):
        self.assertEqual(review_research([evidence('deep_search')], [])['hypotheses'], [])

    def test_missing_stress_cannot_support_hypothesis(self):
        row = evidence()
        row['adaptive_diagnostics']['stress_context_entry_breakdown'] = {}
        self.assertEqual(review_research([row], [])['hypotheses'], [])

    def test_repeated_tests_are_not_independent_observations(self):
        report = review_research([evidence(), evidence()], [])
        self.assertEqual(len(report['hypotheses']), 1)
        self.assertEqual(report['hypotheses'][0]['trades'], 40)
        self.assertEqual(report['status'], 'no_robust_routes')
        self.assertEqual(report['context_evidence_rows'], 2)

    def test_evidence_survives_better_scoring_zero_trade_rows(self):
        empty = {'strategy': 'breakout', 'funnel_stage': 'rejected', 'funnel_score': 25, 'oos': {'trades': 0}}
        self.assertEqual(prune_research_results([empty, evidence()], limit=1, per_family_stage=1)[0]['oos']['trades'], 45)

    def test_late_generations_keep_feasible_windows_and_thresholds(self):
        seen = {candidate_signature(c) for c in discovery_candidates()}
        for generation in (3423, 3424, 10000, 1000000):
            batch = choose_batch([], seen, generation, 20)
            self.assertEqual(len(batch), 20)
            for candidate in batch:
                p = candidate.params
                for key in ('window', 'slow', 'max_hold', 'vwap_window', 'regime_atr_long'):
                    if key in p:
                        self.assertLessEqual(p[key], 120)
                if 'z_entry' in p:
                    self.assertLess(p['z_entry'], 3)
                if 'entry_bps' in p:
                    self.assertLess(p['entry_bps'], 20)
            seen.update(candidate_signature(c) for c in batch)
        self.assertEqual(choose_batch([], set(), 3423, 20), choose_batch([], set(), 3423, 20))

if __name__ == '__main__':
    unittest.main()
