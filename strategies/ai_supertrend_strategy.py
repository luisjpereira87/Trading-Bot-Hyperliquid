from commons.enums.signal_enum import Signal
from commons.models.signal_result_dclass import SignalResult
from commons.models.strategy_base_dclass import StrategyBase
from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from strategies.ai_super_trend_utils import AISuperTrendUtils
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
        signal = trend_signal[-2]

        if signal == Signal.BUY:
            sl = lowerband[-2]
            tp = upperband[-2]


        elif signal == Signal.SELL:
            sl = upperband[-2]
            tp = lowerband[-2]

        else:
            return SignalResult(Signal.HOLD, None, None, None, 0)
        
        return SignalResult(signal, sl, tp, None, 0, 0, 0, 0,  None)
    