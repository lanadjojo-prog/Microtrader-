import unittest

from forex_backtest import _exit_management, _run_asymmetric_exit


class TradeManagementTests(unittest.TestCase):
    def test_breakeven_is_positive_net_after_modeled_costs(self):
        trigger_r, lock_r = _exit_management(
            {"exit_mode": "breakeven_2r", "breakeven_buffer_r": 0.05},
            entry=1.0,
            risk_distance=0.01,
            cost_bps=0.50,
        )
        # Round-trip modeled cost = 1 bp = 0.01R at a 1% stop.
        # Gross stop must therefore lock 0.06R to retain +0.05R net.
        self.assertAlmostEqual(trigger_r, 2.0, places=8)
        self.assertAlmostEqual(lock_r, 0.06, places=8)

    def test_custom_trigger_grid_is_supported(self):
        trigger_r, lock_r = _exit_management(
            {
                "exit_mode": "safe_be_1_5r",
                "management_trigger_r": 1.5,
                "management_lock_net_r": 0.05,
            },
            entry=1.0,
            risk_distance=0.01,
            cost_bps=0.50,
        )
        self.assertAlmostEqual(trigger_r, 1.5, places=8)
        self.assertAlmostEqual(lock_r, 0.06, places=8)

    def test_profit_locks_are_net_of_costs(self):
        _, lock_025 = _exit_management(
            {"exit_mode": "protect_2r_025r"},
            entry=1.0,
            risk_distance=0.01,
            cost_bps=0.50,
        )
        _, lock_050 = _exit_management(
            {"exit_mode": "lock_2r_05r"},
            entry=1.0,
            risk_distance=0.01,
            cost_bps=0.50,
        )
        self.assertAlmostEqual(lock_025, 0.26, places=8)
        self.assertAlmostEqual(lock_050, 0.51, places=8)

    def test_new_protective_stop_activates_next_bar_only(self):
        bars = [
            {"t": "2026-09-30T08:00:00+00:00", "o": 1.0, "h": 1.021, "l": 0.999, "c": 1.015, "v": 100},
            {"t": "2026-09-30T08:01:00+00:00", "o": 1.015, "h": 1.016, "l": 1.000, "c": 1.002, "v": 100},
            {"t": "2026-09-30T08:02:00+00:00", "o": 1.002, "h": 1.003, "l": 1.001, "c": 1.002, "v": 100},
        ]
        idx, px = _run_asymmetric_exit(
            bars,
            0,
            direction=1,
            entry=1.0,
            risk_distance=0.01,
            target_r=5.0,
            max_hold=2,
            params={"exit_mode": "breakeven_2r", "breakeven_buffer_r": 0.05},
            cost_bps=0.50,
        )
        self.assertEqual(idx, 1)
        self.assertAlmostEqual(px, 1.0006, places=8)


if __name__ == "__main__":
    unittest.main()
