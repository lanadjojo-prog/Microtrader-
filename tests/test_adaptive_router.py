import unittest

from adaptive_router import (
    adaptive_signal,
    build_adaptive_policy,
    market_context,
)


def _bars(count=140):
    rows = []
    price = 1.1000
    for i in range(count):
        price += 0.00005
        hour = 8 + (i // 60)
        minute = i % 60
        rows.append({
            "t": f"2026-10-01T{hour:02d}:{minute:02d}:00+00:00",
            "o": price - 0.00002,
            "h": price + 0.00008,
            "l": price - 0.00008,
            "c": price,
            "v": 1000 + i,
        })
    return rows


class AdaptiveRouterTests(unittest.TestCase):
    def test_market_context_is_causal(self):
        bars = _bars()
        before = market_context(bars, 90, {
            "regime_atr_short": 14,
            "regime_atr_long": 50,
            "fast": 8,
            "slow": 30,
            "vwap_window": 60,
        })
        bars[90]["h"] = 9.0
        bars[90]["l"] = 0.1
        bars[90]["c"] = 8.0
        after = market_context(bars, 90, {
            "regime_atr_short": 14,
            "regime_atr_long": 50,
            "fast": 8,
            "slow": 30,
            "vwap_window": 60,
        })
        self.assertEqual(before, after)
        self.assertTrue(before["valid"])
        self.assertIn(before["regime"], {"expansion", "trend", "range"})

    def test_policy_uses_positive_context_specialists_only(self):
        results = [{
            "strategy": "trend_pullback",
            "params": {"fast": 8, "slow": 30, "pullback_z": 1.0},
            "funnel_score": 62.0,
            "adaptive_diagnostics": {
                "context_entry_breakdown": {
                    "trend|London||trend_pullback": {
                        "trades": 12,
                        "expectancy_bps": 4.5,
                        "profit_factor": 1.4,
                        "max_loss_streak": 3,
                    },
                    "range|London||trend_pullback": {
                        "trades": 12,
                        "expectancy_bps": -2.0,
                        "profit_factor": 0.8,
                        "max_loss_streak": 5,
                    },
                },
                "stress_context_entry_breakdown": {
                    "trend|London||trend_pullback": {
                        "trades": 12,
                        "expectancy_bps": 1.2,
                        "profit_factor": 1.15,
                    },
                    "range|London||trend_pullback": {
                        "trades": 12,
                        "expectancy_bps": -4.0,
                        "profit_factor": 0.7,
                    },
                },
            },
        }]
        policy = build_adaptive_policy(results, min_context_trades=4)
        self.assertIn("trend|London", policy["routes"])
        self.assertNotIn("range|London", policy["routes"])
        self.assertEqual(
            policy["routes"]["trend|London"][0]["entry_model"],
            "trend_pullback",
        )

    def test_adaptive_signal_uses_policy_for_current_context(self):
        bars = _bars()
        base = {
            "regime_atr_short": 14,
            "regime_atr_long": 50,
            "regime_vol_ratio": 1.20,
            "regime_trend_atr": 0.20,
            "fast": 8,
            "slow": 30,
            "vwap_window": 60,
        }
        ctx = market_context(bars, 100, base)
        self.assertTrue(ctx["valid"])
        params = {
            **base,
            "adaptive_policy": {
                "routes": {
                    ctx["key"]: [{
                        "entry_model": "momentum",
                        "entry_params": {"fast": 3, "slow": 12, "entry_bps": 0.01},
                        "evidence_score": 12.3,
                        "source_trades": 20,
                    }]
                },
                "exit_profiles": {
                    ctx["key"]: {
                        "name": "test_exit",
                        "stop_atr": 1.0,
                        "target_r": 2.0,
                        "max_hold": 20,
                    }
                },
            },
        }
        direction, model, context, exit_profile, meta = adaptive_signal(
            params, bars, 100
        )
        self.assertEqual(direction, 1)
        self.assertEqual(model, "momentum")
        self.assertEqual(context["key"], ctx["key"])
        self.assertEqual(exit_profile["name"], "test_exit")
        self.assertEqual(meta["source_trades"], 20)


if __name__ == "__main__":
    unittest.main()
