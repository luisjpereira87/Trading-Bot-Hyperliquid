
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
    trend: float
    momentum: float
    divergence: float
    oscillators: float 
    price_action: float
    price_levels: float
    volume_ratio: float
    atr_ratio: float
    timestamp: int  # ou datetime