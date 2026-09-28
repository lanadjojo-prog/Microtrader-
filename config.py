import os
from dataclasses import dataclass
from typing import List


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    api_key: str = os.getenv("ALPACA_API_KEY", "")
    api_secret: str = os.getenv("ALPACA_API_SECRET", "")
    paper: bool = _bool("ALPACA_PAPER", True)
    live_trading_enabled: bool = _bool("LIVE_TRADING_ENABLED", False)
    auto_start: bool = _bool("AUTO_START", False)
    dashboard_token: str = os.getenv("DASHBOARD_TOKEN", "")
    database_url: str = os.getenv("DATABASE_URL", "")

    symbols_raw: str = os.getenv(
        "SYMBOLS",
        "SPY,QQQ,AAPL,MSFT,NVDA,AMD,AMZN,META,GOOGL,TSLA"
    )
    poll_seconds: int = _int("POLL_SECONDS", 20)
    bars_lookback: int = _int("BARS_LOOKBACK", 30)
    fast_window: int = _int("FAST_WINDOW", 4)
    slow_window: int = _int("SLOW_WINDOW", 12)
    entry_edge_bps: float = _float("ENTRY_EDGE_BPS", 8.0)
    exit_edge_bps: float = _float("EXIT_EDGE_BPS", 1.0)

    trade_notional: float = _float("TRADE_NOTIONAL", 3.0)
    max_open_positions: int = _int("MAX_OPEN_POSITIONS", 5)
    max_trades_per_day: int = _int("MAX_TRADES_PER_DAY", 100)
    max_daily_loss: float = _float("MAX_DAILY_LOSS", 5.0)
    take_profit_pct: float = _float("TAKE_PROFIT_PCT", 0.004)
    stop_loss_pct: float = _float("STOP_LOSS_PCT", 0.003)
    cooldown_seconds: int = _int("COOLDOWN_SECONDS", 60)
    kill_close_positions: bool = _bool("KILL_CLOSE_POSITIONS", True)

    data_feed: str = os.getenv("ALPACA_DATA_FEED", "iex")

    lab_symbols_raw: str = os.getenv(
        "LAB_SYMBOLS",
        "SPY,QQQ,AAPL,MSFT,NVDA,AMD,AMZN,META,GOOGL,TSLA"
    )
    lab_lookback_days: int = _int("LAB_LOOKBACK_DAYS", 45)
    lab_timeframe: str = os.getenv("LAB_TIMEFRAME", "1Min")
    lab_max_bars_per_symbol: int = _int("LAB_MAX_BARS_PER_SYMBOL", 15000)
    lab_cost_bps: float = _float("LAB_COST_BPS", 2.5)
    lab_stress_cost_multiplier: float = _float("LAB_STRESS_COST_MULTIPLIER", 2.0)
    lab_min_oos_trades: int = _int("LAB_MIN_OOS_TRADES", 80)
    lab_continuous: bool = _bool("LAB_CONTINUOUS", True)
    lab_auto_start: bool = _bool("LAB_AUTO_START", True)
    lab_target_promoted: int = _int("LAB_TARGET_PROMOTED", 3)
    lab_batch_size: int = _int("LAB_BATCH_SIZE", 20)
    lab_min_profit_factor: float = _float("LAB_MIN_PROFIT_FACTOR", 1.50)
    lab_max_drawdown_pct: float = _float("LAB_MAX_DRAWDOWN_PCT", 6.0)
    lab_min_positive_symbol_ratio: float = _float("LAB_MIN_POSITIVE_SYMBOL_RATIO", 0.60)
    research_auto_start: bool = _bool("RESEARCH_AUTO_START", True)
    research_interval_seconds: int = _int("RESEARCH_INTERVAL_SECONDS", 21600)
    research_agent_auto_start: bool = _bool("RESEARCH_AGENT_AUTO_START", True)
    research_agent_interval_seconds: int = _int("RESEARCH_AGENT_INTERVAL_SECONDS", 60)
    research_agent_focus_families: int = _int("RESEARCH_AGENT_FOCUS_FAMILIES", 4)

    # Crypto / microstructure lab (public Bitvavo data, simulation-only)
    crypto_lab_auto_start: bool = _bool("CRYPTO_LAB_AUTO_START", False)
    crypto_lab_symbols_raw: str = os.getenv("CRYPTO_LAB_SYMBOLS", "BTC-EUR,ETH-EUR,SOL-EUR,BTC-USDC,ETH-USDC,SOL-USDC")
    crypto_lab_poll_seconds: int = _int("CRYPTO_LAB_POLL_SECONDS", 3)
    crypto_lab_window: int = _int("CRYPTO_LAB_WINDOW", 40)
    crypto_lab_maker_fee_bps: float = _float("CRYPTO_LAB_MAKER_FEE_BPS", 15.0)
    crypto_lab_z_entry: float = _float("CRYPTO_LAB_Z_ENTRY", 1.5)
    crypto_lab_imbalance_threshold: float = _float("CRYPTO_LAB_IMBALANCE_THRESHOLD", 0.35)
    crypto_lab_notional_eur: float = _float("CRYPTO_LAB_NOTIONAL_EUR", 5.0)
    crypto_lab_pending_cycles: int = _int("CRYPTO_LAB_PENDING_CYCLES", 5)
    crypto_lab_max_hold_cycles: int = _int("CRYPTO_LAB_MAX_HOLD_CYCLES", 80)
    crypto_lab_target_edge_bps: float = _float("CRYPTO_LAB_TARGET_EDGE_BPS", 10.0)
    crypto_lab_book_depth: int = _int("CRYPTO_LAB_BOOK_DEPTH", 20)
    crypto_lab_flow_window_seconds: int = _int("CRYPTO_LAB_FLOW_WINDOW_SECONDS", 30)
    crypto_lab_flow_threshold: float = _float("CRYPTO_LAB_FLOW_THRESHOLD", 0.35)
    crypto_lab_book_threshold: float = _float("CRYPTO_LAB_BOOK_THRESHOLD", 0.20)

    @property
    def symbols(self) -> List[str]:
        return [s.strip().upper() for s in self.symbols_raw.split(",") if s.strip()]

    @property
    def lab_symbols(self) -> List[str]:
        return [s.strip().upper() for s in self.lab_symbols_raw.split(",") if s.strip()]

    @property
    def crypto_lab_symbols(self) -> List[str]:
        return [s.strip().upper() for s in self.crypto_lab_symbols_raw.split(",") if s.strip()]

    @property
    def trading_base_url(self) -> str:
        return "https://paper-api.alpaca.markets" if self.paper else "https://api.alpaca.markets"

    @property
    def data_base_url(self) -> str:
        return "https://data.alpaca.markets"

    @property
    def can_trade(self) -> bool:
        if self.paper:
            return True
        return self.live_trading_enabled


settings = Settings()
