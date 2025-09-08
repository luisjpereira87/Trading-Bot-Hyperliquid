import logging

import numpy as np

from commons.enums.signal_enum import Signal
from commons.models.signal_result_dclass import SignalResult
from commons.models.strategy_base_dclass import StrategyBase
from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.ai_supertrend.ai_super_trend_utils import AISuperTrendUtils
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from trading_bot.exchange_client import ExchangeClient


class AISuperTrendStrategy(StrategyBase):
    def __init__(self, exchange: ExchangeClient):
        super().__init__()
    
        self.exchange = exchange
        self.ohlcv: OhlcvWrapper
        self.ohlcv_higher: OhlcvWrapper
        self.symbol = None


    def required_init(self, ohlcv: OhlcvWrapper, ohlcv_higher: OhlcvWrapper, symbol: str, price_ref: float):
        self.ohlcv = ohlcv
        self.ohlcv_higher = ohlcv_higher
        self.symbol = symbol
        self.price_ref = price_ref
    
    def set_params(self, params: StrategyParams):
        pass
  
    def set_candles(self, ohlcv):
        self.ohlcv = ohlcv

    def set_higher_timeframe_candles(self, ohlcv_higher: OhlcvWrapper):
        self.ohlcv_higher = ohlcv_higher

    async def get_signal(self) -> SignalResult:

        last_closed_candle = self.ohlcv.get_last_closed_candle()
        aISuperTrendUtils = AISuperTrendUtils(self.ohlcv)
        supertrend, trend, upperband, lowerband, supertrend_smooth = aISuperTrendUtils.get_supertrend()
        ema_cross_signal = aISuperTrendUtils.get_ema_cross_signal()

        signal = ema_cross_signal[-2]
        close = last_closed_candle.close

        indicators = IndicatorsUtils(self.ohlcv)
        ema21 = indicators.ema(21)
        ema50 = indicators.ema(50)
        atr = indicators.atr(14)  # para dar algum espaço ao SL
        upper, mid, lower = indicators.bollinger_bands()

        if signal == Signal.BUY:
            sl = lower[-1]
            tp = upper[-1] + (upper[-1] - lower[-1]) * 0.5

        elif signal == Signal.SELL:
            #sl = upperband[-2]
            sl = upper[-1]
            tp = lower[-1] - (upper[-1] - lower[-1]) * 0.5

        else:
            return SignalResult(Signal.HOLD, None, None, None, 0)
        
        return SignalResult(signal, sl, tp, None, 0, 0, 0, 0,  None)
    
