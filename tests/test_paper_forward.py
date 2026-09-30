import unittest
from datetime import datetime, timedelta, timezone

from paper_trader import _causal_close_entry, _research_signal, _research_exit_reason


def bar(ts, o, h, l, c, v=100):
    return {
        "t": ts.isoformat(),
        "o": float(o),
        "h": float(h),
        "l": float(l),
        "c": float(c),
        "v": float(v),
    }


class PaperForwardTests(unittest.TestCase):
    def test_entry_uses_observable_close_not_historical_open(self):
        start = datetime(2026, 9, 29, 8, 0, tzinfo=timezone.utc)
        closes = [1.10, 1.11, 1.10, 1.09, 1.08, 1.00]
        bars = []
        for i, close in enumerate(closes):
            open_px = close + (0.05 if i == len(closes) - 1 else 0.001)
            bars.append(
                bar(
                    start + timedelta(minutes=i),
                    open_px,
                    max(open_px, close) + 0.002,
                    min(open_px, close) - 0.002,
                    close,
                )
            )

        params = {
            "window": 3,
            "z_entry": 0.5,
            "stop_atr": 1.0,
            "min_volume_ratio": 0.0,
            "volume_window": 3,
        }
        decision = _causal_close_entry(
            "range_reversal",
            params,
            bars,
            len(bars) - 1,
            1,
            default_min_volume_ratio=0.0,
            default_volume_window=3,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision["direction"], 1)
        self.assertEqual(decision["entry_price"], bars[-1]["c"])
        self.assertNotEqual(decision["entry_price"], bars[-1]["o"])
        expected_time = start + timedelta(minutes=len(bars))
        self.assertEqual(
            datetime.fromisoformat(decision["entry_time"]),
            expected_time,
        )
        self.assertGreater(decision["risk_distance"], 0)

    def test_research_mean_reversion_signal_and_exit_are_supported(self):
        start = datetime(2026, 9, 29, 8, 0, tzinfo=timezone.utc)
        closes = [1.10, 1.10, 1.10, 1.10, 1.00]
        bars = [
            bar(
                start + timedelta(minutes=i),
                close + 0.001,
                close + 0.003,
                close - 0.003,
                close,
                100,
            )
            for i, close in enumerate(closes)
        ]
        params = {"window": 4, "z_entry": 1.0, "z_exit": 0.1}
        self.assertEqual(
            _research_signal("mean_reversion", params, bars, len(bars)),
            1,
        )
        recovery = bars + [
            bar(start + timedelta(minutes=5), 1.00, 1.11, 0.99, 1.10, 100)
        ]
        reason = _research_exit_reason(
            "mean_reversion", params, recovery, len(recovery)-1, 1
        )
        self.assertEqual(reason, "research_mean_exit")

    def test_entry_decision_has_no_future_bar_dependency(self):
        start = datetime(2026, 9, 29, 8, 0, tzinfo=timezone.utc)
        bars = [
            bar(start + timedelta(minutes=i), 1.10, 1.12, 0.98, close)
            for i, close in enumerate([1.10, 1.11, 1.10, 1.09, 1.08, 1.00])
        ]
        params = {
            "window": 3,
            "z_entry": 0.5,
            "stop_atr": 1.0,
            "min_volume_ratio": 0.0,
            "volume_window": 3,
        }
        a = _causal_close_entry(
            "range_reversal", params, bars, len(bars) - 1, 1,
            default_min_volume_ratio=0.0, default_volume_window=3,
        )
        future = bar(start + timedelta(minutes=6), 2.0, 3.0, 0.5, 2.5)
        b = _causal_close_entry(
            "range_reversal", params, bars + [future], len(bars) - 1, 1,
            default_min_volume_ratio=0.0, default_volume_window=3,
        )
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
