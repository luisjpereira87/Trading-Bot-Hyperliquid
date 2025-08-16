import logging

from commons.enums.signal_enum import Signal
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from strategies.ai_super_trend_utils import AISuperTrendUtils
from strategies.indicators import Indicators
from strategies.strategy_utils import StrategyUtils


class SignalStrategy:

    @staticmethod      
    def get_signal(ohlcv: OhlcvWrapper) -> tuple[float, float]:

        # O score varia de 0 e 1, cada indicador tem um peso entre 0 e 1 e no final é normalizadp
        score_buy = 0.0
        score_sell = 0.0
        weight_sum = 0.0

        last_closed_candle = ohlcv.get_last_closed_candle()

        supertrend, trend, upperband, lowerband, supertrend_smooth, trend_signal = AISuperTrendUtils(ohlcv).get_supertrend()

        if trend_signal[last_closed_candle.idx] == Signal.BUY:
            score_buy += 1
        elif trend_signal[last_closed_candle.idx] == Signal.SELL:
            score_sell += 1

        """
        # 1. PSAR
        score_buy, score_sell, weight_sum = SignalStrategy.psar_score(ohlcv, score_buy, score_sell, weight_sum)
        print(score_buy, score_sell)
        # 2. RSI and Stochastic
        score_buy, score_sell, weight_sum = SignalStrategy.rsi_stochastic_score(ohlcv, score_buy, score_sell, weight_sum)
        print(score_buy, score_sell)
        # 3. EMA
        #score_buy, score_sell, weight_sum = SignalStrategy.ema_score(ohlcv, score_buy, score_sell, weight_sum)
        print(score_buy, score_sell)
        # 4. Candle body
        #score_buy, score_sell, weight_sum = SignalStrategy.candle_body_score(ohlcv, score_buy, score_sell, weight_sum)
        print(score_buy, score_sell)
        # 5. Lateral market
        #score_buy, score_sell, weight_sum = SignalStrategy.lateral_market_score(ohlcv, score_buy, score_sell, weight_sum)
        print(score_buy, score_sell)
        # 6. Support and resistance
        #score_buy, score_sell, weight_sum = SignalStrategy.support_resistence_score(ohlcv, score_buy, score_sell, weight_sum)
        print(score_buy, score_sell)
        # 7. Trend
        score_buy, score_sell, weight_sum = SignalStrategy.trend_strength_score(ohlcv, score_buy, score_sell, weight_sum)
        print(score_buy, score_sell)

        """
        return score_buy / 1, score_sell / 1
    
    @staticmethod  
    def psar_score(ohlcv: OhlcvWrapper, score_buy: float, score_sell: float, weight_sum: float, weight_point: float = 1) -> tuple[float, float, float]:
        indicators = Indicators(ohlcv)

        last_closed_candle = ohlcv.get_last_closed_candle()

        indicators = Indicators(ohlcv)
        psar_values = indicators.psar()  
        last_psar = psar_values[-2]

        tolerance = 0.001  # 0.1% de tolerância

        if last_closed_candle.close > last_psar * (1 + tolerance):
            score_buy += weight_point
        elif last_closed_candle.close < last_psar * (1 - tolerance):
            score_sell += weight_point
        weight_sum += weight_point

        return score_buy, score_sell, weight_sum
    
    @staticmethod  
    def rsi_stochastic_score(ohlcv: OhlcvWrapper, score_buy: float, score_sell: float, weight_sum: float, weight_point: float = 1) -> tuple[float, float, float]:
        
        last_closed_candle = ohlcv.get_last_closed_candle()

        if StrategyUtils.is_stoch_oversold(ohlcv, last_closed_candle.idx) and StrategyUtils.is_rsi_oversold(ohlcv):
            score_buy += weight_point
        elif StrategyUtils.is_stoch_overbought(ohlcv, last_closed_candle.idx) and StrategyUtils.is_rsi_overbought(ohlcv):
            score_sell += weight_point
        
        weight_sum += weight_point

        return score_buy, score_sell, weight_sum
    
    @staticmethod  
    def ema_score(ohlcv: OhlcvWrapper, score_buy: float, score_sell: float, weight_sum: float, weight_point: float = 1) -> tuple[float, float, float]:
        if StrategyUtils.ema_signal_strict(ohlcv) == Signal.BUY:
            score_buy += 1
        elif StrategyUtils.ema_signal_strict(ohlcv) == Signal.SELL:
            score_sell += 1 
        
        weight_sum += weight_point

        return score_buy, score_sell, weight_sum
    
    @staticmethod  
    def candle_body_score(ohlcv: OhlcvWrapper, score_buy: float, score_sell: float, weight_sum: float, weight_point: float = 1) -> tuple[float, float, float]:
        if StrategyUtils.candle_body_signal(ohlcv)  == Signal.BUY:
            score_buy += weight_point
        elif StrategyUtils.candle_body_signal(ohlcv)  == Signal.SELL:
            score_sell += weight_point
        
        weight_sum += weight_point

        return score_buy, score_sell, weight_sum
    
    @staticmethod  
    def lateral_market_score(ohlcv: OhlcvWrapper, score_buy: float, score_sell: float, weight_sum: float, weight_point: float = 1) -> tuple[float, float, float]:
        if StrategyUtils.is_market_sideways_strict(ohlcv):
            score_sell -= weight_point
            score_buy -= weight_point
        else:
            score_sell += weight_point
            score_buy += weight_point
        
        weight_sum += weight_point

        return score_buy, score_sell, weight_sum
    
    @staticmethod  
    def support_resistence_score(ohlcv: OhlcvWrapper, score_buy: float, score_sell: float, weight_sum: float, weight_point: float = 1) -> tuple[float, float, float]:
        print("RATIO", StrategyUtils.ratio_support_resistence(ohlcv))
        score_sell += weight_point * StrategyUtils.ratio_support_resistence(ohlcv)
        score_buy += weight_point * (1-StrategyUtils.ratio_support_resistence(ohlcv))
        
        weight_sum += weight_point

        return score_buy, score_sell, weight_sum
    
    @staticmethod  
    def trend_strength_score(ohlcv: OhlcvWrapper, score_buy: float, score_sell: float, weight_sum: float, weight_point: float = 1) -> tuple[float, float, float]:
        if StrategyUtils.trend_strength_signal(ohlcv) == Signal.BUY:
            score_buy += weight_point
        elif StrategyUtils.trend_strength_signal(ohlcv) == Signal.SELL:
            score_sell += weight_point
        
        weight_sum += weight_point

        return score_buy, score_sell, weight_sum

    

    
    