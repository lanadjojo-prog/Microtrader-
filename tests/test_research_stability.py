import unittest
from datetime import datetime, timedelta, timezone

from strategy_lab import (
    Candidate,
    aggregate_bars,
    candidate_signature,
    choose_batch,
    discovery_candidates,
    evaluate_candidate,
    simulate,
)


def make_bar(ts, o, h, l, c, v=100):
    return {
        "t": ts.isoformat(),
        "o": float(o),
        "h": float(h),
        "l": float(l),
        "c": float(c),
        "v": float(v),
    }


class ResearchStabilityTests(unittest.TestCase):
    def test_five_minute_aggregation_is_clock_aligned(self):
        start = datetime(2026, 9, 29, 8, 2, tzinfo=timezone.utc)
        bars = [
            make_bar(start + timedelta(minutes=i), 1+i, 2+i, 0.5+i, 1.5+i)
            for i in range(7)
        ]
        out = aggregate_bars(bars, 5)
        self.assertEqual(len(out), 2)
        self.assertTrue(out[0]["t"].endswith("08:00:00+00:00"))
        self.assertTrue(out[1]["t"].endswith("08:05:00+00:00"))
        self.assertEqual(out[0]["o"], 1.0)
        self.assertEqual(out[1]["o"], 4.0)

    def test_breakout_level_excludes_trigger_bar(self):
        start = datetime(2026, 9, 29, 8, 0, tzinfo=timezone.utc)
        bars = [
            make_bar(start + timedelta(minutes=0), 1.000, 1.005, 0.995, 1.000),
            make_bar(start + timedelta(minutes=1), 1.000, 1.006, 0.996, 1.001),
            make_bar(start + timedelta(minutes=2), 1.001, 1.007, 0.997, 1.002),
            make_bar(start + timedelta(minutes=3), 1.002, 1.008, 0.998, 1.003),
            # Trigger candle closes above the previous three highs.
            make_bar(start + timedelta(minutes=4), 1.003, 1.020, 1.002, 1.015),
            make_bar(start + timedelta(minutes=5), 1.016, 1.019, 1.010, 1.017),
            make_bar(start + timedelta(minutes=6), 1.017, 1.020, 1.012, 1.018),
        ]
        candidate = Candidate(
            "breakout",
            {
                "window": 3,
                "buffer_bps": 0.0,
                "max_hold": 1,
                "min_volume_ratio": 0.0,
                "volume_window": 10,
            },
        )
        trades = simulate(candidate, "EUR/USD", bars, 0.0)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["side"], "long")

    def test_deep_search_freezes_incubator_parameters(self):
        seen = {candidate_signature(c) for c in discovery_candidates()}
        params = {
            "timeframe_min": 5,
            "_phase": "incubator",
            "_policy_version": "test",
            "market": "forex",
            "data_source": "ctrader",
            "direction_mode": "long_short",
            "entry_sessions": "london_new_york",
            "min_volume_ratio": 0.70,
            "volume_window": 50,
            "window": 31,
            "z_entry": 1.7,
            "z_exit": 0.2,
            "max_hold": 21,
        }
        row = {
            "strategy": "mean_reversion",
            "family": "mean_reversion",
            "params": params,
            "funnel_stage": "deep_search",
            "funnel_score": 70,
            "oos": {"expectancy_bps": 1.0},
        }
        batch = choose_batch([row], seen, generation=5, batch_size=10)
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0].strategy, "mean_reversion")
        self.assertEqual(batch[0].params["window"], 31)
        self.assertEqual(batch[0].params["z_entry"], 1.7)
        self.assertEqual(batch[0].params["_phase"], "deep_search")

    def test_deep_search_is_terminal(self):
        start = datetime(2026, 8, 3, 7, 0, tzinfo=timezone.utc)
        bars = []
        price = 1.10
        for i in range(600):
            price += 0.00002 if (i // 15) % 2 == 0 else -0.00002
            bars.append(
                make_bar(
                    start + timedelta(minutes=i),
                    price,
                    price + 0.0002,
                    price - 0.0002,
                    price,
                )
            )
        candidate = Candidate(
            "mean_reversion",
            {
                "_phase": "deep_search",
                "timeframe_min": 1,
                "window": 20,
                "z_entry": 1.5,
                "z_exit": 0.25,
                "max_hold": 20,
                "entry_sessions": "london_new_york",
                "min_volume_ratio": 0.0,
                "volume_window": 10,
            },
        )
        result = evaluate_candidate(
            candidate,
            {"EUR/USD": bars},
            0.5,
            2.0,
            10,
            min_profit_factor=1.05,
            max_drawdown_pct=20,
            min_positive_symbol_ratio=0,
            min_trades_per_day=0.1,
        )
        self.assertIn(result["funnel_stage"], {"promoted", "rejected"})
        self.assertEqual(result["validation_policy"]["phase"], "deep_search")
        self.assertTrue(result["validation_policy"]["deep_search_parameters_frozen"])


if __name__ == "__main__":
    unittest.main()
