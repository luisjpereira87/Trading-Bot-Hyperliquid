import logging

from commons.enums.signal_enum import Signal
from commons.models.signal_result_dclass import SignalResult
from commons.models.strategy_base_dclass import StrategyBase
from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.ai_supertrend.ai_super_trend_utils import AISuperTrendUtils
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
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
        supertrend, trend, upperband, lowerband, supertrend_smooth, trend_signal = AISuperTrendUtils(self.ohlcv).get_supertrend()
        #signal = trend_signal[-1]
        atr = IndicatorsUtils(self.ohlcv).atr(14)

        signal = trend_signal[-1]
        close = last_closed_candle.close
        high = last_closed_candle.high
        low = last_closed_candle.low
        atr_val = atr[-1]
        ub = upperband[-1]
        lb = lowerband[-1]

        
        tolerance = 0.005  # 0.5% de margem
        resistance_levels, support_levels = SupportResistanceUtils.detect_multiple_support_resistance(self.ohlcv)
        #closes = self.ohlcv.closes
        #highs = self.ohlcv.highs
        #lows = self.ohlcv.lows
        tolerance = atr_val / close  # converte ATR em percentagem relativa ao preço
        sl = tp = 0.0

        if signal == Signal.BUY:
            
            nearest_res = min([r for r in resistance_levels if r > close], default=None)
            if nearest_res:
                sl = low - 0.5 * atr_val   # SL com margem de 0.5x ATR abaixo do candle
                #sl = self.ohlcv.lows[-1] - (self.ohlcv.highs[-1] - self.ohlcv.lows[-1]) * 0.25
                nearest_res *= (1 - tolerance)
                tp = min(nearest_res, ub, high)
            else:
                sl = lb
                tp = ub
                #tp = self.price_ref + (upperband[-1] - lowerband[-1])

            sl, tp = RiskUtils.adjust_sl_tp(signal, sl, tp, self.price_ref, atr_val, support_levels, resistance_levels)
            if tp is None:
                logging.info(f"TP demasiado próximo do preço de entrada ({signal.value}): {tp} vs {self.price_ref}")
                return SignalResult(Signal.HOLD, None, None, None, 0)

        elif signal == Signal.SELL:

            nearest_sup = max([s for s in support_levels if s < close], default=None)
            if nearest_sup:
                sl = high + 0.5 * atr_val  # SL com margem de 0.5x ATR acima do candle
                #sl = self.ohlcv.lows[-1] + (self.ohlcv.highs[-1] - self.ohlcv.lows[-1]) * 0.25
                nearest_sup *= (1 + tolerance)
                tp = max(nearest_sup, lb, low)
            else:
                sl = ub
                tp = lb
                #tp = self.price_ref - (upperband[-1] - lowerband[-1])

            sl, tp = RiskUtils.adjust_sl_tp(signal, sl, tp, self.price_ref, atr_val, support_levels, resistance_levels)
            if tp is None:
                logging.info(f"TP demasiado próximo do preço de entrada ({signal.value}): {tp} vs {self.price_ref}")
                return SignalResult(Signal.HOLD, None, None, None, 0)

        else:
            return SignalResult(Signal.HOLD, None, None, None, 0)
        
        return SignalResult(signal, sl, tp, None, 0, 0, 0, 0,  None)
    
