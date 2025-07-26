from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np

from commons.enums.mode_enum import ModeEnum
from commons.enums.signal_enum import Signal


@dataclass
class StrategyParams:
    #symbol, timeframe, mode='conservative', multiplier=0.9, adx_threshold=15, rsi_buy_threshold=40, rsi_sell_threshold=60
    #ohlcv: dict
    #symbol: str
    #timeframe: str
    mode: ModeEnum = ModeEnum.CONSERVATIVE
    multiplier: float = 0.0
    adx_threshold: float = 0.0
    rsi_buy_threshold: float = 0.0
    rsi_sell_threshold: float = 0.0
    adx_trend_threshold : float = 0.0
    adx_lateral_threshold: float = 0.0
    rsi_oversold_threshold : float = 0.0
    buy_threshold : float = 0.0
    volatility_breakout_ratio: float = 0.0
    sl_multiplier_aggressive : float = 0.0
    tp_multiplier_aggressive : float = 0.0
    sl_multiplier_conservative : float = 0.0
    tp_multiplier_conservative : float = 0.0
    volume_threshold_ratio : float = 0.0
    atr_threshold_ratio : float = 0.0

    block_lateral_market: bool = True

    """
    weights_trend: float = 0.0
    weights_rsi : float = 0.0
    weights_stochastic: float = 0.0
    weights_price_action: float = 0.0
    weights_proximity_to_bands: float = 0.0
    weights_exhaustion: float = 0.0
    weights_penalty_factor: float = 0.0
    weights_macd: float = 0.0
    weights_cci: float = 0.0
    weights_confirmation_candle_penalty: float = 0.0
    weights_divergence: float = 0.0
    """
    weights_trend: float = 0.0
    weights_momentum: float = 0.0
    weights_oscillators: float = 0.0
    weights_price_action: float = 0.0
    weights_price_levels: float = 0.0

    def __post_init__(self):
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)

            # Converte None para 0.0 nos campos float
            if isinstance(value, type(None)):
                setattr(self, field_name, 0.0)

            # Converte np.generic para float nativo
            elif isinstance(value, np.generic):
                setattr(self, field_name, value.item())