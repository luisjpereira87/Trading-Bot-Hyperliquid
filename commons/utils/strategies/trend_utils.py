import logging
from typing import List, Tuple

import numpy as np

from commons.enums.signal_enum import Signal
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from commons.utils.strategies.price_action_utils import PriceActionUtils
from commons.utils.strategies.support_resistance_utils import \
    SupportResistanceUtils


class TrendUtils:
    
    @staticmethod
    def trend_strength_signal(ohlcv: OhlcvWrapper, lookback: int = 3, adx_threshold: float = 25) -> Signal:
        """
        Retorna um sinal consolidado da tendência:
        - Signal.BUY -> tendência de alta ainda forte
        - Signal.SELL -> tendência de baixa ainda forte
        - Signal.HOLD -> tendência está enfraquecendo ou indecisa
        """
        closes = ohlcv.closes
        highs = ohlcv.highs
        lows = ohlcv.lows
        opens = ohlcv.opens
        indicators = IndicatorsUtils(ohlcv)
        
        ema_fast = indicators.ema(10)
        ema_slow = indicators.ema(50)
        adx = indicators.adx()
        
        n = len(closes)
        if n < lookback + 1 or len(adx) < lookback + 1:
            return Signal.HOLD  # dados insuficientes
        
        # Contadores de candles enfraquecidos
        weakening_count = 0
        
        for i in range(n - lookback, n):
            candle_body = abs(closes[i] - opens[i])
            candle_range = highs[i] - lows[i]
            if candle_range == 0:
                continue
            body_ratio = candle_body / candle_range
            
            weak_adx = adx[i] < adx_threshold or adx[i] < adx[i-1]
            small_candle = body_ratio < 0.4
            trend_conflict = (ema_fast[i] < ema_slow[i] and closes[i] > ema_fast[i]) or \
                            (ema_fast[i] > ema_slow[i] and closes[i] < ema_fast[i])
            
            if sum([weak_adx, small_candle, trend_conflict]) >= 2:
                weakening_count += 1

        # Se a maioria das velas no lookback está enfraquecendo, HOLD
        if weakening_count >= (lookback // 2) + 1:
            return Signal.HOLD
        
        # Senão, retorna a tendência dominante
        if ema_fast[-1] > ema_slow[-1]:
            return Signal.BUY
        elif ema_fast[-1] < ema_slow[-1]:
            return Signal.SELL
        else:
            return Signal.HOLD
        
    @staticmethod
    def detect_lateral_market(ohlcv: OhlcvWrapper, adx_threshold) -> bool:
        indicators = IndicatorsUtils(ohlcv)
        adx = indicators.adx()
        adx_now = adx[-1]
        lateral_market = adx_now < adx_threshold
        return lateral_market
    
    @staticmethod
    def trend_signal_with_adx(ohlcv: OhlcvWrapper, adx_threshold: float):
        indicators = IndicatorsUtils(ohlcv)
        ema = indicators.ema()[-1]
        prev_ema = indicators.ema()[-2]

        if TrendUtils.detect_lateral_market(ohlcv, adx_threshold):
            if ema > prev_ema:
                return 1  # buy
            elif ema < prev_ema:
                return -1  # sell
        return 0  # sem sinal
    
    @staticmethod
    def passes_volume_volatility_filter(ohlcv: OhlcvWrapper, symbol: (str | None), volume_threshold_ratio: float, atr_threshold_ratio: float):
        indicators = IndicatorsUtils(ohlcv)
        atr = indicators.atr()

        volumes = getattr(indicators, 'volumes', None)
        if volumes is None or len(volumes) < 20:
            return True  # Sem dados suficientes

        avg_volume = sum(volumes[-20:]) / 20
        current_volume = volumes[-1]

        if current_volume < volume_threshold_ratio * avg_volume:
            logging.info(f"{symbol} - Volume baixo: {current_volume:.2f} < {volume_threshold_ratio*100:.0f}% da média ({avg_volume:.2f})")
            return False

        avg_atr = sum(atr[-20:]) / 20
        current_atr = atr[-1]

        if current_atr < atr_threshold_ratio * avg_atr:
            logging.info(f"{symbol} - ATR baixo: {current_atr:.4f} < {atr_threshold_ratio*100:.0f}% da média ({avg_atr:.4f})")
            return False

        return True
    
    @staticmethod
    def calculate_higher_tf_trend(ohlcv_higher: OhlcvWrapper, adx_threshold: float):
        if len(ohlcv_higher) < 21:
            # Sem dados suficientes para timeframe maior → assume neutro
            return 0

        indicators_htf = IndicatorsUtils(ohlcv_higher)
        closes_htf = indicators_htf.closes
        atr_htf = indicators_htf.atr()
        ema_htf = indicators_htf.ema()
        adx_htf = indicators_htf.adx()

        ema_now = ema_htf[-1]
        ema_prev = ema_htf[-2]
        adx_now = adx_htf[-1]

        lateral = adx_now < adx_threshold

        if lateral:
            if ema_now > ema_prev:
                return 1
            elif ema_now < ema_prev:
                return -1
            else:
                return 0
        else:
            # Se mercado não lateral, confia na tendência EMA simples
            if ema_now > ema_prev:
                return 1
            elif ema_now < ema_prev:
                return -1
            else:
                return 0
            
    @staticmethod
    def detect_divergence(
        ohlcv: OhlcvWrapper
    ) -> Tuple[List[int], List[int]]:
        """
        Detecta divergências entre preço (pivôs) e RSI.
        Retorna listas de índices onde existem divergências de alta e baixa.
        """
        bullish_divergences = []
        bearish_divergences = []

        indicators = IndicatorsUtils(ohlcv)
        rsi = list(indicators.rsi(14))

        pivots_high, pivots_low = SupportResistanceUtils.find_pivots(ohlcv)

        # Divergência de alta: preços fazem pivôs baixos mais baixos, RSI faz pivôs baixos mais altos
        for i in range(1, len(pivots_low)):
            idx1 = pivots_low[i - 1]
            idx2 = pivots_low[i]
            price_low1 = ohlcv.lows[idx1]
            price_low2 = ohlcv.lows[idx2]
            rsi_low1 = rsi[idx1]
            rsi_low2 = rsi[idx2]

            if price_low2 < price_low1 and rsi_low2 > rsi_low1:
                bullish_divergences.append(idx2)

        # Divergência de baixa: preços fazem pivôs altos mais altos, RSI faz pivôs altos mais baixos
        for i in range(1, len(pivots_high)):
            idx1 = pivots_high[i - 1]
            idx2 = pivots_high[i]
            price_high1 = ohlcv.highs[idx1]
            price_high2 = ohlcv.highs[idx2]
            rsi_high1 = rsi[idx1]
            rsi_high2 = rsi[idx2]

            if price_high2 > price_high1 and rsi_high2 < rsi_high1:
                bearish_divergences.append(idx2)

        return bullish_divergences, bearish_divergences
    
    @staticmethod
    def is_market_manipulation(candles: OhlcvWrapper) -> bool:
        current = candles.get_current_candle()
        return (
            PriceActionUtils.has_large_wick(current)
            or PriceActionUtils.is_single_candle_pump(candles)
            or PriceActionUtils.has_price_gap(candles)
        )
    
    @staticmethod
    def get_market_manipulation_score(
        candles: OhlcvWrapper,
        lookback: int = 20,
        price_spike_threshold: float = 2.0,
        wick_ratio_threshold: float = 2.5,
        volume_spike_ratio: float = 2.0
    ) -> float:
        """
        Retorna um score de manipulação de mercado entre 0.0 e 1.0 com base em:
        - Tamanho anormal do candle (spike)
        - Pavio longo (sombra superior/inferior muito maior que o corpo)
        - Volume muito acima da média
        - Reversão após candle extremo

        Parâmetros:
        - candles: OhlcvWrapper
        - lookback: Nº de candles para calcular médias
        - thresholds: valores para deteção de anomalias

        Returns:
        - manip_score: float (0.0 a 1.0)
        """
        if len(candles) < lookback + 3:
            return 0.0

        recent_candles = candles.get_recent_closed(lookback)
        last_candle = candles.get_last_closed_candle()

        # Preço médio de corpos
        avg_body_size = sum(abs(c.open - c.close) for c in recent_candles) / lookback
        body = abs(last_candle.open - last_candle.close)

        # Comprimento do candle
        full_range = last_candle.high - last_candle.low
        upper_wick = last_candle.high - max(last_candle.open, last_candle.close)
        lower_wick = min(last_candle.open, last_candle.close) - last_candle.low

        # Volume
        avg_volume = sum(c.volume for c in recent_candles) / lookback
        volume_score = 1.0 if last_candle.volume > avg_volume * volume_spike_ratio else 0.0

        # Corpo anormal
        body_score = min(1.0, body / (avg_body_size * price_spike_threshold))

        # Wick ratio
        wick_ratio = max(upper_wick, lower_wick) / body if body > 0 else 0
        wick_score = 1.0 if wick_ratio > wick_ratio_threshold else 0.0

        # Reversão após candle forte
        candle_1 = candles.get_recent_closed(1)[-1]
        candle_2 = candles.get_recent_closed(2)[-2]

        reversal_score = 0.0
        if body_score > 0.8:
            if last_candle.close < candle_1.low < candle_2.low:
                # Engolfo de baixa após candle forte de alta
                reversal_score = 1.0
            elif last_candle.close > candle_1.high > candle_2.high:
                # Engolfo de alta após candle forte de queda
                reversal_score = 1.0

        # Média ponderada simples
        score = (
            0.4 * body_score +
            0.3 * wick_score +
            0.2 * volume_score +
            0.1 * reversal_score
        )

        return round(min(score, 1.0), 2)
    
    @staticmethod
    def calculate_volume_ratio(candles: OhlcvWrapper, window: int = 20) -> float:
        volumes = candles.volumes
        if len(volumes) < window + 1:
            return 1.0  # neutro

        avg_volume = np.mean(volumes[-window - 1:-1])
        current_volume = volumes[-1]

        ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        return ratio
    
    @staticmethod
    def calculate_atr_ratio(candles: OhlcvWrapper):
        indicators = IndicatorsUtils(candles)
        atr = indicators.atr()[-1]
        high = candles.highs[-1]
        low = candles.lows[-1]
        range_candle = high - low
        return range_candle / atr if atr != 0 else 0
    
    @staticmethod
    def calculate_volume_penalty(candles: OhlcvWrapper) -> Tuple[float, float]:
        volumes = candles.volumes
        recent_volume = volumes[-1]
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)

        if avg_volume == 0:
            return 1.0, 1.0  # evitar divisão por zero

        volume_ratio = recent_volume / avg_volume
        volume_score = min(volume_ratio, 1.5) / 1.5  # normaliza para [0, 1]

        # penalizar abaixo de 0.7 (ajustável)
        if volume_score < 0.7:
            buy_penalty = 1 - (0.7 - volume_score)
            sell_penalty = 1 - (0.7 - volume_score)
        else:
            buy_penalty = 1.0
            sell_penalty = 1.0

        # opcional: ajustar pelo tipo de candle
        closes = candles.closes
        opens = candles.opens
        if closes[-1] > opens[-1]:  # candle bullish
            sell_penalty *= 0.95
        elif closes[-1] < opens[-1]:  # candle bearish
            buy_penalty *= 0.95

        # limitar entre [0.3, 1.0] por segurança
        buy_penalty = max(0.3, min(buy_penalty, 1.0))
        sell_penalty = max(0.3, min(sell_penalty, 1.0))

        return buy_penalty, sell_penalty
    
    @staticmethod
    def is_abnormal_volume(candles: OhlcvWrapper, lookback: int = 20, threshold: float = 2.0) -> bool:
        if len(candles) < lookback + 1:
            return False

        recent = candles.get_recent_closed(lookback)
        volumes = [c.volume for c in recent]

        avg_volume = sum(volumes) / len(volumes)
        current_volume = candles.get_current_candle().volume

        return current_volume > (avg_volume * threshold)
    
    @staticmethod
    def get_volatility_moves(candles: OhlcvWrapper, lookback: int = 50) -> list[float]:
        return [
            abs(c.high - c.low)
            for c in candles.get_recent_closed(lookback)
        ]
    
    @staticmethod
    def is_strong_trend_up(ohlcv: OhlcvWrapper) -> bool:
        indicators = IndicatorsUtils(ohlcv)

        ema_fast = indicators.ema(10)
        ema_slow = indicators.ema(50)
        adx = indicators.adx()
        i = -1
        return ema_fast[i] > ema_slow[i] and adx[i] > 25

    @staticmethod
    def is_strong_trend_down(ohlcv: OhlcvWrapper) -> bool:
        indicators = IndicatorsUtils(ohlcv)

        ema_fast = indicators.ema(10)
        ema_slow = indicators.ema(50)
        adx = indicators.adx()
        i = -1
        return ema_fast[i] < ema_slow[i] and adx[i] > 25
    
    @staticmethod
    def is_pullback_in_uptrend(ohlcv: OhlcvWrapper) -> bool:
        indicators = IndicatorsUtils(ohlcv)

        closes = ohlcv.closes
        lows = ohlcv.lows
        ema_fast = indicators.ema(10)
        ema_slow = indicators.ema(50)
        i = -1
        return (
            ema_fast[i] > ema_slow[i] and
            closes[i] > ema_slow[i] and
            lows[i] < ema_fast[i] * 0.985
        )
    
    @staticmethod
    def is_pullback_in_downtrend(ohlcv: OhlcvWrapper) -> bool:
        indicators = IndicatorsUtils(ohlcv)

        closes = ohlcv.closes
        highs = ohlcv.highs
        ema_fast = indicators.ema(10)
        ema_slow = indicators.ema(50)
        i = -1
        return (
            ema_fast[i] < ema_slow[i] and
            closes[i] < ema_slow[i] and
            highs[i] > ema_fast[i] * 1.015
        )
    
    @staticmethod
    def equal_weights(keys: list[str]) -> dict[str, float]:
        count = len(keys)
        value = 1 / count
        return {k: value for k in keys}
    
    @staticmethod    
    def is_market_sideways(ohlcv: OhlcvWrapper, lookback:int=14, atr_threshold_pct:float=0.003, range_threshold_pct:float=0.003) -> bool:
        """
        Detecta mercado lateral com base em ATR e range médio.

        Args:
            ohlcv (OhlcvWrapper): dados OHLCV com arrays .highs, .lows, .closes
            lookback (int): número de candles para cálculo da média
            atr_threshold_pct (float): limite percentual para ATR (ex: 0.003 = 0.3%)
            range_threshold_pct (float): limite percentual para range (high-low) médio

        Returns:
            bool: True se mercado lateral, False caso contrário
        """
        highs = ohlcv.highs
        lows = ohlcv.lows
        closes = ohlcv.closes

        if len(highs) < lookback + 1:
            return False  # dados insuficientes

        # Calcula ATR simplificado (True Range médio)
        tr_values = []
        for i in range(-lookback, 0):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1]),
            )
            tr_values.append(tr)
        atr = sum(tr_values) / lookback

        # Calcula range médio (high - low)
        range_values = [highs[i] - lows[i] for i in range(-lookback, 0)]
        avg_range = sum(range_values) / lookback

        # Preço médio para referência percentual (podes usar close médio)
        avg_price = sum(closes[-lookback:]) / lookback

        atr_ratio = atr / avg_price
        range_ratio = avg_range / avg_price

        # Debug prints (podes tirar depois)
        print(f"ATR ratio: {atr_ratio:.5f}, Range ratio: {range_ratio:.5f}")

        return (atr_ratio < atr_threshold_pct) and (range_ratio < range_threshold_pct)
    
    @staticmethod
    def is_market_sideways_strict(ohlcv: OhlcvWrapper, lookback=20, atr_threshold_pct=0.003, max_trend_pct=0.01, max_ema_slope=0.0005):
        closes = ohlcv.closes
        highs = ohlcv.highs
        lows = ohlcv.lows

        if len(closes) < lookback + 1:
            return False

        # 1. ATR relativo
        atr = IndicatorsUtils(ohlcv).atr()
        atr_value = atr[-1]
        atr_rel = atr_value / closes[-1]

        # 2. Variação percentual do fechamento inicial para final do lookback
        price_start = closes[-lookback - 1]
        price_end = closes[-1]
        trend_pct = abs(price_end - price_start) / price_start

        # 3. Range total percentual no lookback (max - min) / preço atual
        period_max = max(highs[-lookback:])
        period_min = min(lows[-lookback:])
        range_pct = (period_max - period_min) / closes[-1]

        # 4. Inclinação EMA21 (aprox. diferença percentual entre última e penúltima EMA)
        ema21 = IndicatorsUtils(ohlcv).ema(21)
        if len(ema21) < 2:
            return False
        ema_slope = abs(ema21[-1] - ema21[-2]) / ema21[-2]

        print(f"ATR rel: {atr_rel:.5f}, Trend pct: {trend_pct:.5f}, Range pct: {range_pct:.5f}, EMA slope: {ema_slope:.5f}")

        # Verifica critérios estritos para lateralidade
        if atr_rel < atr_threshold_pct \
        and trend_pct < max_trend_pct \
        and range_pct < max_trend_pct \
        and ema_slope < max_ema_slope:
            return True
        else:
            return False