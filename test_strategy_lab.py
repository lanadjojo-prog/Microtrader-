import unittest

from strategy_lab import Candidate, candidate_grid, metrics, simulate


class StrategyLabTests(unittest.TestCase):
    def test_candidate_grid_contains_two_strategy_families(self):
        names = {c.strategy for c in candidate_grid()}
        self.assertEqual(names, {"momentum", "mean_reversion"})
        self.assertEqual(len(candidate_grid()), 10)

    def test_metrics_positive_trade_stream(self):
        result = metrics([
            {"net_return": 0.01, "exit_time": "2026-01-01T10:00:00Z"},
            {"net_return": -0.002, "exit_time": "2026-01-01T10:05:00Z"},
            {"net_return": 0.006, "exit_time": "2026-01-01T10:10:00Z"},
        ])
        self.assertEqual(result["trades"], 3)
        self.assertGreater(result["expectancy_bps"], 0)
        self.assertGreater(result["profit_factor"], 1)

    def test_costs_reduce_simulated_momentum_expectancy(self):
        bars = []
        price = 100.0
        for i in range(120):
            price *= 1.0008
            bars.append({
                "t": f"2026-01-01T{i//60:02d}:{i%60:02d}:00Z",
                "o": price * 0.9999,
                "h": price * 1.0005,
                "l": price * 0.9995,
                "c": price,
                "v": 1000,
            })
        candidate = Candidate("momentum", {"fast": 3, "slow": 10, "entry_bps": 1.0, "max_hold": 10})
        low_cost = metrics(simulate(candidate, "TEST", bars, 0.0))
        high_cost = metrics(simulate(candidate, "TEST", bars, 20.0))
        self.assertGreater(low_cost["trades"], 0)
        self.assertGreater(low_cost["expectancy_bps"], high_cost["expectancy_bps"])


if __name__ == "__main__":
    unittest.main()
