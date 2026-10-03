import unittest

from adaptive_router import (
    ADAPTIVE_CONTEXT_VERSION,
    adaptive_signal,
    build_adaptive_policy,
    execution_policy,
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
            "params": {
                "_phase": "discovery",
                "fast": 8,
                "slow": 30,
                "pullback_z": 1.0,
            },
            "funnel_score": 62.0,
            "adaptive_diagnostics": {
                "context_version": ADAPTIVE_CONTEXT_VERSION,
                "context_entry_breakdown": {
                    "trend|London||trend_pullback": {
                        "trades": 40,
                        "expectancy_bps": 4.5,
                        "profit_factor": 1.4,
                        "max_loss_streak": 3,
                    },
                    "range|London||trend_pullback": {
                        "trades": 40,
                        "expectancy_bps": -2.0,
                        "profit_factor": 0.8,
                        "max_loss_streak": 5,
                    },
                },
                "stress_context_entry_breakdown": {
                    "trend|London||trend_pullback": {
                        "trades": 20,
                        "expectancy_bps": 1.2,
                        "profit_factor": 1.15,
                    }
                },
                "context_entry_management_breakdown": {
                    "trend|London||trend_pullback||baseline": {
                        "trades": 40,
                        "expectancy_bps": 4.5,
                        "profit_factor": 1.4,
                    },
                    "trend|London||trend_pullback||protect_1_5r_0_2r": {
                        "trades": 40,
                        "expectancy_bps": 5.0,
                        "profit_factor": 1.45,
                        "max_loss_streak": 3,
                    },
                },
                "stress_context_entry_management_breakdown": {
                    "trend|London||trend_pullback||baseline": {
                        "trades": 20,
                        "expectancy_bps": 1.0,
                        "profit_factor": 1.10,
                    },
                    "trend|London||trend_pullback||protect_1_5r_0_2r": {
                        "trades": 20,
                        "expectancy_bps": 1.2,
                        "profit_factor": 1.12,
                    },
                },
                "management_profiles": {
                    "baseline": {"name": "baseline"},
                    "protect_1_5r_0_2r": {
                        "name": "protect_1_5r_0_2r",
                        "management_trigger_r": 1.5,
                        "management_lock_net_r": 0.2,
                    },
                },
                "context_entry_exit_breakdown": {
                    "trend|London||trend_pullback||balanced_2_5r": {
                        "trades": 40,
                        "expectancy_bps": 4.2,
                        "profit_factor": 1.35,
                        "max_loss_streak": 3,
                    }
                },
                "stress_context_entry_exit_breakdown": {
                    "trend|London||trend_pullback||balanced_2_5r": {
                        "trades": 20,
                        "expectancy_bps": 1.0,
                        "profit_factor": 1.08,
                    }
                },
                "exit_profiles": {
                    "balanced_2_5r": {
                        "name": "balanced_2_5r",
                        "stop_atr": 1.0,
                        "target_r": 2.5,
                        "max_hold": 32,
                    }
                },
            },
        }]
        policy = build_adaptive_policy(results, min_context_trades=30)
        self.assertIn("trend|London", policy["routes"])
        self.assertNotIn("range|London", policy["routes"])
        route = policy["routes"]["trend|London"][0]
        self.assertEqual(route["entry_model"], "trend_pullback")
        self.assertEqual(
            route["management_profile"]["name"],
            "protect_1_5r_0_2r",
        )
        self.assertEqual(route["exit_profile"]["name"], "balanced_2_5r")

    def test_policy_never_learns_from_final_holdout_rows(self):
        results = [{
            "strategy": "trend_pullback",
            "params": {
                "_phase": "deep_search",
                "fast": 8,
                "slow": 30,
                "pullback_z": 1.0,
            },
            "funnel_score": 99.0,
            "adaptive_diagnostics": {
                "context_entry_breakdown": {
                    "trend|London||trend_pullback": {
                        "trades": 100,
                        "expectancy_bps": 20.0,
                        "profit_factor": 3.0,
                        "max_loss_streak": 1,
                    }
                },
                "stress_context_entry_breakdown": {},
            },
        }]
        policy = build_adaptive_policy(results, min_context_trades=4)
        self.assertEqual(policy["routes"], {})
        self.assertTrue(policy["final_holdout_excluded"])

    def test_execution_policy_ignores_non_actionable_evidence_scores(self):
        base = {
            "version": "adaptive-context-v1",
            "routes": {
                "trend|London": [{
                    "entry_model": "momentum",
                    "entry_params": {"fast": 3, "slow": 12, "entry_bps": 4.0},
                    "source_strategy": "momentum",
                    "evidence_score": 10.0,
                    "source_trades": 20,
                    "source_expectancy_bps": 3.0,
                }]
            },
            "exit_profiles": {
                "trend|London": {
                    "name": "trend_atr_2_5r",
                    "stop_atr": 1.0,
                    "target_r": 2.5,
                    "max_hold": 32,
                }
            },
        }
        changed_evidence = {
            **base,
            "routes": {
                "trend|London": [{
                    **base["routes"]["trend|London"][0],
                    "evidence_score": 99.0,
                    "source_trades": 200,
                    "source_expectancy_bps": 9.0,
                }]
            },
        }
        self.assertEqual(execution_policy(base), execution_policy(changed_evidence))

    def test_policy_can_distill_one_time_condition(self):
        base_diag = {
            "context_version": ADAPTIVE_CONTEXT_VERSION,
            "context_entry_breakdown": {
                "trend|London||momentum": {
                    "trades": 80,
                    "expectancy_bps": 1.2,
                    "profit_factor": 1.10,
                    "max_loss_streak": 4,
                }
            },
            "stress_context_entry_breakdown": {
                "trend|London||momentum": {
                    "trades": 40,
                    "expectancy_bps": 0.4,
                    "profit_factor": 1.03,
                }
            },
            "context_entry_time_breakdown": {
                "trend|London||momentum||london_open": {
                    "trades": 40,
                    "expectancy_bps": 4.0,
                    "profit_factor": 1.35,
                    "max_loss_streak": 2,
                }
            },
            "stress_context_entry_time_breakdown": {
                "trend|London||momentum||london_open": {
                    "trades": 20,
                    "expectancy_bps": 1.0,
                    "profit_factor": 1.10,
                }
            },
            "context_entry_exit_breakdown": {
                "trend|London||momentum||balanced_2_5r": {
                    "trades": 80,
                    "expectancy_bps": 2.0,
                    "profit_factor": 1.25,
                    "max_loss_streak": 3,
                }
            },
            "stress_context_entry_exit_breakdown": {
                "trend|London||momentum||balanced_2_5r": {
                    "trades": 40,
                    "expectancy_bps": 0.5,
                    "profit_factor": 1.05,
                }
            },
            "exit_profiles": {
                "balanced_2_5r": {
                    "name": "balanced_2_5r",
                    "stop_atr": 1.0,
                    "target_r": 2.5,
                    "max_hold": 32,
                }
            },
        }
        entry_row = {
            "strategy": "momentum",
            "params": {
                "_phase": "discovery",
                "fast": 4,
                "slow": 16,
                "entry_bps": 8.0,
            },
            "funnel_score": 55.0,
            "adaptive_diagnostics": base_diag,
        }
        policy = build_adaptive_policy(
            [entry_row],
            min_context_trades=30,
            max_entries_per_context=2,
        )
        route = policy["routes"]["trend|London"][0]
        self.assertEqual(route["entry_model"], "momentum")
        self.assertEqual(
            route["conditions"],
            {"time_bucket": "london_open"},
        )
        self.assertEqual(route["exit_profile"]["name"], "balanced_2_5r")
        self.assertLessEqual(len(route["conditions"]), 1)

    def test_old_context_version_cannot_feed_router(self):
        row = {
            "strategy": "momentum",
            "params": {"_phase": "discovery"},
            "funnel_score": 99.0,
            "adaptive_diagnostics": {
                "context_version": "adaptive-context-v1",
                "context_entry_breakdown": {
                    "trend|London||momentum": {
                        "trades": 1000,
                        "expectancy_bps": 50.0,
                        "profit_factor": 5.0,
                    }
                },
            },
        }
        policy = build_adaptive_policy([row], min_context_trades=30)
        self.assertEqual(policy["routes"], {})

    def test_adaptive_signal_has_no_execution_fallback_without_evidence(self):
        bars = _bars()
        params = {
            "regime_atr_short": 14,
            "regime_atr_long": 50,
            "regime_vol_ratio": 1.20,
            "regime_trend_atr": 0.20,
            "fast": 8,
            "slow": 30,
            "vwap_window": 60,
            "_adaptive_research_mode": False,
            "adaptive_policy": {"routes": {}, "exit_profiles": {}},
        }
        direction, model, context, exit_profile, meta = adaptive_signal(
            params, bars, 100
        )
        self.assertEqual(direction, 0)
        self.assertEqual(model, "none")
        self.assertEqual(exit_profile, {})
        self.assertEqual(meta["no_trade_reason"], "no_proven_route")

    def test_conditioned_route_only_trades_when_condition_matches(self):
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
        params = {
            **base,
            "_adaptive_research_mode": False,
            "adaptive_policy": {
                "routes": {
                    ctx["key"]: [{
                        "entry_model": "momentum",
                        "entry_params": {"fast": 3, "slow": 12, "entry_bps": 0.01},
                        "conditions": {"volume_bucket": "impossible_bucket"},
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
        direction, model, _, _, _ = adaptive_signal(params, bars, 100)
        self.assertEqual(direction, 0)
        self.assertEqual(model, "none")

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
                        "exit_profile": {
                            "name": "test_exit",
                            "stop_atr": 1.0,
                            "target_r": 2.0,
                            "max_hold": 20,
                        },
                    }]
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
