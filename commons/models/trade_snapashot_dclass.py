
from dataclasses import dataclass

from commons.enums.signal_enum import Signal


@dataclass
class TradeSnapshot:
    symbol: str
    entry_price: float
    size: float
    sl: float
    tp: float
    signal: Signal
    candle_type: str  # ex: 'bullish engulfing', 'doji', etc.
    rsi: float
    stochastic: float
    adx: float
    macd: float
    cci: float
    weights_trend: float
    weights_momentum: float
    weights_oscillators: float 
    weights_price_action: float
    weights_price_levels: float
    weights_divergence: float
    weights_channel_position: float
    penalty_exhaustion: float
    penalty_factor: float
    penalty_manipulation: float
    penalty_confirmation_candle: float
    volume_ratio: float
    atr_ratio: float
    timestamp: int  # ou datetime