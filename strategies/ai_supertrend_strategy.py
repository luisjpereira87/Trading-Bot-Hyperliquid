import logging

from commons.enums.signal_enum import Signal
from commons.models.signal_result_dclass import SignalResult
from commons.models.strategy_base_dclass import StrategyBase
from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.ai_supertrend.ai_super_trend_utils import AISuperTrendUtils
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
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
        signal = trend_signal[-1]

        atr = IndicatorsUtils(self.ohlcv).atr(14)
        tolerance = 0.005  # 0.5% de margem
        resistance_levels, support_levels = SupportResistanceUtils.detect_multiple_support_resistance(self.ohlcv)
        closes = self.ohlcv.closes
        highs = self.ohlcv.highs
        lows = self.ohlcv.lows
        tolerance = atr[-1] / closes[-1]  # converte ATR em percentagem relativa ao preço


        if signal == Signal.BUY:

            #sl, tp = self.calculate_sl_tp_with_risk(signal, lowerband[-1], upperband[-1], self.price_ref)

            #sl = lowerband[-1]
            #tp = upperband[-1]
            #tp = self.price_ref + (upperband[-1] - lowerband[-1])
            
            nearest_res = min([r for r in resistance_levels if r > closes[-1]], default=None)
            if nearest_res:
                sl = lows[-1] - 0.5 * atr[-1]   # SL com margem de 0.5x ATR abaixo do candle
                #sl = self.ohlcv.lows[-1] - (self.ohlcv.highs[-1] - self.ohlcv.lows[-1]) * 0.25
                nearest_res *= (1 - tolerance)
                tp = nearest_res
            else:
                sl = lowerband[-1]
                tp = self.price_ref + (upperband[-1] - lowerband[-1])
                #tp = upperband[-1]

        elif signal == Signal.SELL:
            #sl = upperband[-1]
            #tp = lowerband[-1]
            #tp = self.price_ref - (upperband[-1] - lowerband[-1])
            #sl, tp = self.calculate_sl_tp_with_risk(signal, lowerband[-1], upperband[-1], self.price_ref)

            nearest_sup = max([s for s in support_levels if s < closes[-1]], default=None)
            if nearest_sup:
                sl = highs[-1] + 0.5 * atr[-1]  # SL com margem de 0.5x ATR acima do candle
                #sl = self.ohlcv.lows[-1] + (self.ohlcv.highs[-1] - self.ohlcv.lows[-1]) * 0.25
                nearest_sup *= (1 + tolerance)
                tp = nearest_sup
            else:
                sl = upperband[-1]
                tp = self.price_ref - (upperband[-1] - lowerband[-1])
                #tp = lowerband[-1]

            

        else:
            return SignalResult(Signal.HOLD, None, None, None, 0)
        
        if (tp - self.price_ref) / self.price_ref < 0.002 or tp <= self.price_ref:
            #raise ValueError(f"TP demasiado próximo do preço de entrada: {tp} vs {self.price_ref}")
            logging.info(f"TP demasiado próximo do preço de entrada: {tp} vs {self.price_ref}")
            return SignalResult(signal, sl, tp, None, 0, 0, 0, 0,  None)
        
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
    