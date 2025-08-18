import logging

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
        supertrend, trend, upperband, lowerband, supertrend_smooth, trend_signal = AISuperTrendUtils(self.ohlcv).get_supertrend()
        signal = trend_signal[-1]

        if signal == Signal.BUY:

            #sl, tp = self.calculate_sl_tp_with_risk(signal, lowerband[-1], upperband[-1], self.price_ref)

            sl = lowerband[-1]
            tp = upperband[-1]
            tp = self.price_ref + (upperband[-1] - lowerband[-1])

        elif signal == Signal.SELL:
            sl = upperband[-1]
            tp = lowerband[-1]
            tp = self.price_ref - (upperband[-1] - lowerband[-1])
            #sl, tp = self.calculate_sl_tp_with_risk(signal, lowerband[-1], upperband[-1], self.price_ref)

        else:
            return SignalResult(Signal.HOLD, None, None, None, 0)
        
        return SignalResult(signal, sl, tp, None, 0, 0, 0, 0,  None)
    
    def calculate_sl_tp_with_risk(self, signal: Signal, band_lower, band_upper, entry_price, margin_pct=0.001, risk_multiplier=2):
        """
        Calcula SL e TP usando bandas como referência, mas ajustando o TP pelo risco.
        
        :param signal: Signal.BUY ou Signal.SELL
        :param band_lower: banda inferior
        :param band_upper: banda superior
        :param entry_price: preço de entrada
        :param margin_pct: margem mínima intra-candle
        :param risk_multiplier: multiplica o risco (distância até SL) para definir TP
        :return: sl, tp
        """
        if signal == Signal.BUY:
            sl = band_lower - band_lower * margin_pct  # SL abaixo da banda
            risk = entry_price - sl
            tp = entry_price + risk * risk_multiplier
            # garante que o TP não ultrapassa a banda superior
            tp = min(tp, band_upper + band_upper * margin_pct)

        elif signal == Signal.SELL:
            sl = band_upper + band_upper * margin_pct  # SL acima da banda
            risk = sl - entry_price
            tp = entry_price - risk * risk_multiplier
            # garante que o TP não ultrapassa a banda inferior
            tp = max(tp, band_lower - band_lower * margin_pct)

        return sl, tp
    