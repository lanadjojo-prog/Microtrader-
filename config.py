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

    # Forex-first research. External data is deliberately separate from broker execution.
    forex_first: bool = _bool("FOREX_FIRST", False)
    forex_lab_auto_start: bool = _bool("FOREX_LAB_AUTO_START", False)
    forex_pairs_raw: str = os.getenv("FOREX_PAIRS", "EUR/USD,GBP/USD,USD/JPY")
    forex_start_capital: float = _float("FOREX_START_CAPITAL_EUR", 50.0)
    forex_risk_eur: float = _float("FOREX_RISK_EUR", 0.75)
    forex_cost_bps: float = _float("FOREX_COST_BPS", 0.50)
    forex_stress_cost_multiplier: float = _float("FOREX_STRESS_COST_MULTIPLIER", 2.0)
    forex_lookback_days: int = _int("FOREX_LOOKBACK_DAYS", 120)
    forex_max_bars_per_pair: int = _int("FOREX_MAX_BARS_PER_PAIR", 12000)
    forex_lab_batch_size: int = _int("FOREX_LAB_BATCH_SIZE", 12)
    forex_min_oos_trades: int = _int("FOREX_MIN_OOS_TRADES", 60)
    forex_min_profit_factor: float = _float("FOREX_MIN_PROFIT_FACTOR", 1.25)
    forex_min_payoff_ratio: float = _float("FOREX_MIN_PAYOFF_RATIO", 1.80)
    forex_max_retail_leverage: float = _float("FOREX_MAX_RETAIL_LEVERAGE", 30.0)
    forex_min_lot: float = _float("FOREX_MIN_LOT", 0.01)

    # Read-only market-data layer for testing. external/auto uses Dukascopy and
    # optionally overlays recent Twelve Data bars when a key is configured.
    forex_data_provider: str = os.getenv("FOREX_DATA_PROVIDER", "external").strip().lower()
    twelve_data_api_key: str = os.getenv("TWELVE_DATA_API_KEY", "").strip()
    twelve_data_base_url: str = os.getenv(
        "TWELVE_DATA_BASE_URL", "https://api.twelvedata.com"
    ).strip()
    dukascopy_base_urls_raw: str = os.getenv(
        "DUKASCOPY_BASE_URLS",
        "https://datafeed.dukascopy.com/datafeed,https://www.dukascopy.com/datafeed",
    )

    # cTrader / Fusion demo validation layer.
    ctrader_environment: str = os.getenv("CTRADER_ENVIRONMENT", "demo").strip().lower()
    ctrader_demo_only: bool = _bool("CTRADER_DEMO_ONLY", True)
    ctrader_oauth_scope: str = os.getenv("CTRADER_OAUTH_SCOPE", "accounts").strip().lower()
    ctrader_client_id: str = os.getenv("CTRADER_CLIENT_ID", "")
    ctrader_client_secret: str = os.getenv("CTRADER_CLIENT_SECRET", "")
    ctrader_redirect_uri: str = os.getenv(
        "CTRADER_REDIRECT_URI",
        "https://microtrader-6thu.onrender.com/ctrader/callback",
    )
    ctrader_access_token: str = os.getenv("CTRADER_ACCESS_TOKEN", "")
    ctrader_refresh_token: str = os.getenv("CTRADER_REFRESH_TOKEN", "")
    ctrader_account_id: str = os.getenv("CTRADER_ACCOUNT_ID", "")

    # Precision Lab: separate tight-stop tester using bid/ask tick execution.
    precision_lab_auto_start: bool = _bool("PRECISION_LAB_AUTO_START", False)
    precision_data_provider: str = os.getenv("PRECISION_DATA_PROVIDER", "external").strip().lower()
    precision_pairs_raw: str = os.getenv("PRECISION_PAIRS", "EUR/USD")
    precision_lookback_days: int = _int("PRECISION_LOOKBACK_DAYS", 7)
    precision_max_bars_per_pair: int = _int("PRECISION_MAX_BARS_PER_PAIR", 12000)
    precision_max_ticks_per_side: int = _int("PRECISION_MAX_TICKS_PER_SIDE", 250000)
    precision_batch_size: int = _int("PRECISION_BATCH_SIZE", 12)
    precision_start_capital: float = _float("PRECISION_START_CAPITAL_EUR", 50.0)
    precision_risk_eur: float = _float("PRECISION_RISK_EUR", 0.75)
    precision_commission_pips: float = _float("PRECISION_COMMISSION_PIPS", 0.50)
    precision_stress_multiplier: float = _float("PRECISION_STRESS_MULTIPLIER", 2.0)
    precision_min_oos_trades: int = _int("PRECISION_MIN_OOS_TRADES", 40)
    precision_min_profit_factor: float = _float("PRECISION_MIN_PROFIT_FACTOR", 1.20)

    # Forward paper trading for promoted Forex strategies.
    paper_trading_auto_start: bool = _bool("PAPER_TRADING_AUTO_START", True)
    paper_start_balance: float = _float("PAPER_START_BALANCE_EUR", 50.0)
    paper_poll_seconds: int = _int("PAPER_POLL_SECONDS", 60)

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
    crypto_lab_process_interval_ms: int = _int("CRYPTO_LAB_PROCESS_INTERVAL_MS", 2000)

    @property
    def symbols(self) -> List[str]:
        return [s.strip().upper() for s in self.symbols_raw.split(",") if s.strip()]

    @property
    def lab_symbols(self) -> List[str]:
        return [s.strip().upper() for s in self.lab_symbols_raw.split(",") if s.strip()]

    @property
    def forex_pairs(self) -> List[str]:
        return [s.strip().upper() for s in self.forex_pairs_raw.split(",") if s.strip()]

    @property
    def dukascopy_base_urls(self) -> List[str]:
        return [s.strip() for s in self.dukascopy_base_urls_raw.split(",") if s.strip()]

    @property
    def ctrader_configured(self) -> bool:
        return bool(
            self.ctrader_client_id
            and self.ctrader_client_secret
            and self.ctrader_access_token
        )

    @property
    def precision_pairs(self) -> List[str]:
        return [s.strip().upper() for s in self.precision_pairs_raw.split(",") if s.strip()]

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
