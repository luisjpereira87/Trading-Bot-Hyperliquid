import numpy as np

from commons.enums.signal_enum import Signal
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper


class MomentumUtils:
    
    @staticmethod
    def stochastic(ohlcv: OhlcvWrapper) -> Signal:

        stoch_k, stoch_d = IndicatorsUtils(ohlcv).stochastic()
        k_now, d_now = stoch_k[-1], stoch_d[-1]
        k_prev, d_prev = stoch_k[-2], stoch_d[-2]

        if k_now > d_now and k_prev <= d_prev:
            return Signal.BUY
        elif k_now < d_now and k_prev >= d_prev:
            return Signal.SELL
        return Signal.HOLD
    
    @staticmethod
    def momentum_signal_macd_cci(ohlcv: OhlcvWrapper, cci_period: int = 20, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9) -> Signal:
        """
        Calcula sinal combinado de momentum baseado em MACD e CCI.

        Retorna:
        - Signal.BUY se MACD cruza sinal para cima e CCI > 100
        - Signal.SELL se MACD cruza sinal para baixo e CCI < -100
        - Signal.HOLD caso contrário
        """
        indicators = IndicatorsUtils(ohlcv)

        macd_line, signal_line = indicators.macd(macd_fast, macd_slow, macd_signal)
        cci = indicators.cci(cci_period)

        if len(macd_line) < 2 or len(signal_line) < 2 or len(cci) < 1:
            return Signal.HOLD

        macd_curr, macd_prev = macd_line[-1], macd_line[-2]
        signal_curr, signal_prev = signal_line[-1], signal_line[-2]
        cci_curr = cci[-1]

        # Cruzamento de MACD para cima (bullish)
        macd_bull_cross = macd_prev <= signal_prev and macd_curr > signal_curr
        # Cruzamento de MACD para baixo (bearish)
        macd_bear_cross = macd_prev >= signal_prev and macd_curr < signal_curr

        if macd_bull_cross and cci_curr > 100:
            return Signal.BUY
        elif macd_bear_cross and cci_curr < -100:
            return Signal.SELL
        else:
            return Signal.HOLD
        
    @staticmethod
    def is_weak_momentum(ohlcv: OhlcvWrapper, idx: int, ema_fast=9, ema_slow=21, threshold=0.0015) -> bool:
        """
        Retorna True se a diferença entre EMA rápida e lenta for pequena
        (momentum fraco).
        """
        close = ohlcv.get_last_closed_candle().close
        ema_fast_val = IndicatorsUtils(ohlcv).ema(ema_fast)[idx]
        ema_slow_val = IndicatorsUtils(ohlcv).ema(ema_slow)[idx]
        diff = abs(ema_fast_val - ema_slow_val) / close
        return diff < threshold
    
    @staticmethod
    def is_stoch_overbought(ohlcv: OhlcvWrapper, idx: int, k_period=14, d_period=3, overbought=80) -> bool:
        """
        Retorna True se o estocástico estiver sobrecomprado.
        """
        
        k, d = IndicatorsUtils(ohlcv).stochastic()
        return k[idx] > overbought and d[idx] > overbought

    @staticmethod
    def is_stoch_oversold(ohlcv: OhlcvWrapper, idx: int, k_period=14, d_period=3, oversold=20) -> bool:
        """
        Retorna True se o estocástico estiver sobrevendido.
        """
        k, d = IndicatorsUtils(ohlcv).stochastic()
        return k[idx] < oversold and d[idx] < oversold
    
    @staticmethod
    def is_rsi_overbought(ohlcv: OhlcvWrapper, rsi_threshold: float = 70) -> bool:
        """
        Verifica se o RSI do último candle fechado está em sobrecompra.
        
        Args:
            ohlcv: Wrapper que contém o RSI e os candles.
            rsi_threshold: Valor limite para considerar sobrecompra (default 70).
        
        Returns:
            True se RSI estiver acima do limite de sobrecompra, False caso contrário.
        """
        rsi_series = IndicatorsUtils(ohlcv).rsi()  # supõe que tens método para obter indicador
        if rsi_series is None or len(rsi_series) == 0:
            return False
        last_rsi = rsi_series[-1]
        return last_rsi >= rsi_threshold
    
    @staticmethod
    def is_rsi_oversold(ohlcv: OhlcvWrapper, rsi_threshold: float = 30) -> bool:
        """
        Verifica se o RSI do último candle fechado está em sobrevenda.
        
        Args:
            ohlcv: Wrapper que contém o RSI e os candles.
            rsi_threshold: Valor limite para considerar sobrevenda (default 30).
        
        Returns:
            True se RSI estiver abaixo do limite de sobrevenda, False caso contrário.
        """
        rsi_series = IndicatorsUtils(ohlcv).rsi() # supõe que tens método para obter indicador
        if rsi_series is None or len(rsi_series) == 0:
            return False
        last_rsi = rsi_series[-1]
        return last_rsi <= rsi_threshold
    
    @staticmethod
    def _calc_rsi(closes, period:int=14) -> float:
        deltas = np.diff(closes)
        ups = deltas.clip(min=0)
        downs = -deltas.clip(max=0)
        avg_gain = np.mean(ups[-period:])
        avg_loss = np.mean(downs[-period:])
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))
    
    
    @staticmethod
    def rsi_signal(ohlcv: OhlcvWrapper, period: int = 14) -> Signal:
        """
        Retorna 'buy' se RSI < oversold, 'sell' se RSI > overbought, senão None.
        """
        closes = ohlcv.closes
        if len(closes) < period + 1:
            return Signal.HOLD

        deltas = np.diff(closes)
        ups = deltas.clip(min=0)
        downs = -deltas.clip(max=0)

        avg_gain = np.mean(ups[-period:])
        avg_loss = np.mean(downs[-period:])

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # Inclinação do RSI
        prev_rsi = MomentumUtils._calc_rsi(closes[:-1], period)
        slope = rsi - prev_rsi

        # Lógica combinada
        if rsi >= 50 and slope >= 0:
            return Signal.BUY
        elif rsi < 50 and slope <= 0:
            return Signal.SELL
        elif slope > 0:
            return Signal.BUY
        else:
            return Signal.SELL

    @staticmethod
    def stochastic_signal(ohlcv: OhlcvWrapper, period=14) -> Signal:
        highs = ohlcv.highs
        lows = ohlcv.lows
        closes = ohlcv.closes

        if len(closes) < period + 1:
            return Signal.HOLD

        # Calcular %K atual
        highest_high = max(highs[-period:])
        lowest_low = min(lows[-period:])
        k = 100 * (closes[-1] - lowest_low) / (highest_high - lowest_low)

        # Calcular %K anterior
        prev_highest_high = max(highs[-period-1:-1])
        prev_lowest_low = min(lows[-period-1:-1])
        prev_k = 100 * (closes[-2] - prev_lowest_low) / (prev_highest_high - prev_lowest_low)

        slope = k - prev_k

        
        # Lógica combinada
        if k >= 50 and slope >= 0:
            return Signal.BUY
        elif k < 50 and slope <= 0:
            return Signal.SELL
        elif slope > 0:
            return Signal.BUY
        else:
            return Signal.SELL
        
    @staticmethod
    def ema_signal(ohlcv: OhlcvWrapper, fast_period:int=21, slow_period:int=50) -> Signal:
        closes = ohlcv.closes

        if len(closes) < slow_period:
            return Signal.HOLD

        # Calcular EMAs
        ema_fast = IndicatorsUtils(ohlcv).ema()
        ema_slow = IndicatorsUtils(ohlcv).ema(50)

        # Inclinação da EMA rápida
        slope = ema_fast[-1] - ema_fast[-2]

        # Lógica de decisão
        if ema_fast[-1] > ema_slow[-1] and slope >= 0:
            return Signal.BUY
        elif ema_fast[-1] < ema_slow[-1] and slope <= 0:
            return Signal.SELL
        elif slope > 0:
            return Signal.BUY
        else:
            return Signal.SELL
        
    @staticmethod
    def ema_signal_strict(
        ohlcv: OhlcvWrapper,
        fast_period: int = 21,
        slow_period: int = 50,
        min_slope: float = 0.0,
        min_ema_distance: float = 0.001,  # 0.1% de distância mínima
        lookback_support_resistance: int = 20
    ) -> Signal:
        closes = ohlcv.closes

        if len(closes) < slow_period:
            return Signal.HOLD

        # Calcular EMAs
        indicators = IndicatorsUtils(ohlcv)
        ema_fast = indicators.ema(fast_period)
        ema_slow = indicators.ema(slow_period)

        # Inclinação da EMA rápida
        slope = ema_fast[-1] - ema_fast[-2]

        # Distância relativa entre EMAs
        ema_distance = abs(ema_fast[-1] - ema_slow[-1]) / ema_slow[-1]

        # Preço atual
        price = closes[-1]

        # Suporte e resistência recentes
        highs = ohlcv.highs[-lookback_support_resistance:]
        lows = ohlcv.lows[-lookback_support_resistance:]
        recent_support = min(lows)
        recent_resistance = max(highs)

        # PSAR (opcional)
        psar_values = IndicatorsUtils(ohlcv).psar()
        psar_last = psar_values[-1]

        # --- Lógica de Compra ---
        if (
            ema_fast[-1] > ema_slow[-1]  # tendência de alta
            and slope > min_slope        # inclinação positiva
            and ema_distance > min_ema_distance
            and price > ema_slow[-1]     # preço acima da EMA lenta
            and price > recent_support   # preço não está colado ao suporte
            and psar_last < price        # PSAR de compra
        ):
            return Signal.BUY

        # --- Lógica de Venda ---
        elif (
            ema_fast[-1] < ema_slow[-1]  # tendência de baixa
            and slope < -min_slope       # inclinação negativa
            and ema_distance > min_ema_distance
            and price < ema_slow[-1]
            and price < recent_resistance
            and psar_last > price
        ):
            return Signal.SELL

        return Signal.HOLD