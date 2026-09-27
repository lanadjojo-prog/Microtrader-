from strategy import moving_average_momentum


def bars(values):
    return [{"c": str(v), "t": f"2026-01-01T00:{i:02d}:00Z"} for i, v in enumerate(values)]


def test_buy_signal():
    s = moving_average_momentum(bars([100]*8 + [101,102,103,104]), 4, 12, 8, 1, False)
    assert s.action == "BUY"


def test_sell_signal_when_momentum_fades():
    s = moving_average_momentum(bars([100,101,102,103] + [100]*8), 4, 12, 8, 1, True)
    assert s.action == "SELL"


def test_hold_if_insufficient_bars():
    s = moving_average_momentum(bars([100,101]), 4, 12, 8, 1, False)
    assert s.action == "HOLD"
