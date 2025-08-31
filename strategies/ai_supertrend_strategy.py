import logging

import numpy as np

from commons.enums.signal_enum import Signal
from commons.models.signal_result_dclass import SignalResult
from commons.models.strategy_base_dclass import StrategyBase
from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.ai_supertrend.ai_super_trend_utils import AISuperTrendUtils
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from commons.utils.strategies.momentum_utils import MomentumUtils
from commons.utils.strategies.risk_utils import RiskUtils
from commons.utils.strategies.support_resistance_utils import \
    SupportResistanceUtils
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
        supertrend, trend, upperband, lowerband, supertrend_smooth, trend_signal, trend_signal_filtered = AISuperTrendUtils(self.ohlcv).get_supertrend()
        atr = IndicatorsUtils(self.ohlcv).atr(14)


        signal = trend_signal[-2]
        close = last_closed_candle.close
        high = last_closed_candle.high
        low = last_closed_candle.low
        atr_val = atr[-2]
        ub = upperband[-2]
        lb = lowerband[-2]

        #print("AQUIIIII", trend_signal_filtered)

        if signal == Signal.BUY:
            sl = lb
            #tp = close
            tp = close + (upperband[-2] - lowerband[-2])

        elif signal == Signal.SELL:
            sl = ub
            #tp = close
            tp = close - (upperband[-2] - lowerband[-2]) 

        else:
            return SignalResult(Signal.HOLD, None, None, None, 0)
        
        return SignalResult(signal, sl, tp, None, 0, 0, 0, 0,  None)
    
