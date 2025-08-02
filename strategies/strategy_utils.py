import logging
from typing import List, Tuple

import numpy as np

from commons.enums.mode_enum import ModeEnum
from commons.enums.signal_enum import Signal
from commons.models.ohlcv_type_dclass import Ohlcv
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from strategies.indicators import Indicators


class StrategyUtils:

    @staticmethod    
    def detect_reversal_pattern(ohlcv: OhlcvWrapper, rsi_overbought: float = 70, rsi_oversold: float = 30) -> tuple[bool, bool]:
        """
        Detecta reversão de topo (baixa) ou de fundo (alta) com base em padrão de candle + RSI.
        Retorna (is_bearish_reversal, is_bullish_reversal).
        """
        indicators = Indicators(ohlcv)

        rsi = list(indicators.rsi(period=14))
        if len(ohlcv) < 2 or len(rsi) < 2:
            return False, False

        recent = ohlcv.get_recent_closed(lookback=2)
        prev = recent[-2]
        curr = recent[-1]

        body = abs(curr.close - curr.open)
        candle_range = curr.high - curr.low
        upper_wick = curr.high - max(curr.close, curr.open)
        lower_wick = min(curr.close, curr.open) - curr.low

        if candle_range == 0:
            return False, False

        body_ratio = body / candle_range
        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range

        rsi_now = rsi[-1]

        # Reversão de topo (estrela cadente + RSI sobrecomprado)
        is_bearish = (
            body_ratio < 0.3 and
            upper_wick_ratio > 0.5 and
            curr.close < curr.open and
            rsi_now >= rsi_overbought
        )

        # Reversão de fundo (martelo + RSI sobrevendido)
        is_bullish = (
            body_ratio < 0.3 and
            lower_wick_ratio > 0.5 and
            curr.close > curr.open and
            rsi_now <= rsi_oversold
        )

        # Engolfo de baixa como fallback
        if not is_bearish:
            is_bearish = StrategyUtils.is_bearish_engulfing(prev, curr)

        return is_bearish, is_bullish
    
    @staticmethod
    def is_bearish_engulfing(prev: Ohlcv, curr: Ohlcv) -> bool:
        return (
            prev.close > prev.open and  # Candle anterior é de alta
            curr.close < curr.open and  # Candle atual é de baixa
            curr.open > prev.close and  # Abertura acima do fechamento anterior
            curr.close < prev.open      # Fechamento abaixo da abertura anterior
        )

    @staticmethod
    def detect_support_resistance(
        candles: OhlcvWrapper,
        lookback: int = 50,
        tolerance_pct: float = 0.005
    ) -> tuple[float, float]:
        """
        Detecta níveis de suporte e resistência baseados em máximos/mínimos locais.

        Retorna:
        - resistência (float) se houver
        - suporte (float) se houver
        """
        #if len(candles) < lookback:
        #    return None, None

        highs = [c.high for c in candles.get_recent_closed(lookback)]
        lows = [c.low for c in candles.get_recent_closed(lookback)]

        resistance = max(highs)
        support = min(lows)

        return resistance, support

    @staticmethod
    def get_distance_to_levels(ohlcv: OhlcvWrapper, price_ref: float, lookback: int = 50) -> tuple[float, float]:
        recent = ohlcv.get_recent_closed(lookback)

        highs = [candle.high for candle in recent]
        lows = [candle.low for candle in recent]

        resistance = max(highs)
        support = min(lows)
        price = price_ref  # ou self.ohlcv.get_current_candle().close

        dist_to_res = abs(resistance - price)
        dist_to_sup = abs(price - support)

        return dist_to_res, dist_to_sup
    
    
    @staticmethod
    def is_flat_candle(ohlcv: OhlcvWrapper):
        candle = ohlcv.get_current_candle()
        return candle.open == candle.close and candle.high == candle.low
    
    @staticmethod
    def calculate_bands(ohlcv: OhlcvWrapper, multiplier):
        indicators = Indicators(ohlcv)

        closes = ohlcv.closes

        atr = indicators.atr()
        upper_band = [closes[i] + multiplier * atr[i] for i in range(len(atr))]
        lower_band = [closes[i] - multiplier * atr[i] for i in range(len(atr))]

        return upper_band, lower_band

    @staticmethod
    def detect_lateral_market(ohlcv: OhlcvWrapper, symbol,  adx_threshold):
        indicators = Indicators(ohlcv)
        adx = indicators.adx()
        adx_now = adx[-1]
        lateral_market = adx_now < adx_threshold
        logging.info(f"{symbol} - ADX: {adx_now:.2f} → Lateral: {lateral_market}")
        return lateral_market
    
    @staticmethod
    def trend_signal_with_adx(ohlcv: OhlcvWrapper, symbol: (str | None), adx_threshold: float):
        indicators = Indicators(ohlcv)
        ema = indicators.ema()[-1]
        prev_ema = indicators.ema()[-2]

        if StrategyUtils.detect_lateral_market(ohlcv, symbol, adx_threshold):
            if ema > prev_ema:
                return 1  # buy
            elif ema < prev_ema:
                return -1  # sell
        return 0  # sem sinal
    
    @staticmethod
    def detect_setup_123(ohlcv: OhlcvWrapper):

        closes = ohlcv.closes
        highs = ohlcv.highs
        lows = ohlcv.lows
        price = ohlcv.opens[0]

        if len(closes) < 5:
            return False, False

        h = highs[-5:]
        l = lows[-5:]

        # Buy Setup 123
        buy_123 = (
            l[2] < l[1] and l[2] < l[3] and  # ponto 2 é mínimo local
            h[3] > h[2] and  # ponto 3 alta local
            l[4] > l[2] and  # ponto 4 confirma subida
            price > h[3]  # preço acima de ponto 3
        )

        # Sell Setup 123
        sell_123 = (
            h[2] > h[1] and h[2] > h[3] and  # ponto 2 é máximo local
            l[3] < l[2] and  # ponto 3 baixa local
            h[4] < h[2] and  # ponto 4 confirma queda
            price < l[3]  # preço abaixo do ponto 3
        )

        return buy_123, sell_123
    
    @staticmethod
    def is_breakout_candle(ohlcv: OhlcvWrapper, idx: int, multiplier: float = 2.0, window: int = 20) -> bool:
        closes = ohlcv.closes
        opens = ohlcv.opens
        
        if idx < window:
            return False

        candle_bodies = [abs(closes[i] - opens[i]) for i in range(idx - window, idx)]
        avg_body = sum(candle_bodies) / window
        current_body = abs(closes[idx] - opens[idx])

        return current_body > multiplier * avg_body
    
    @staticmethod
    def check_price_action_signals(ohlcv: OhlcvWrapper):
        closes = ohlcv.closes
        opens = ohlcv.opens
        open_now = opens[-1]
        close_now = closes[-1]


       # Prioridade 1: Breakout candle (mais forte)
        if StrategyUtils.is_breakout_candle(ohlcv, -1):
            return "buy"  # ou True, se quiseres só booleano

        # Prioridade 2: Setup 123
        buy_123, sell_123 = StrategyUtils.detect_setup_123(ohlcv)
        if buy_123:
            return "buy"
        if sell_123:
            return "sell"

        # Prioridade 3: Cor da vela atual
        if close_now > open_now:
            return "buy"
        elif close_now < open_now:
            return "sell"

        # Se nenhuma condição for satisfeita
        return None
    
    @staticmethod
    def calculate_sl_tp(ohlcv: OhlcvWrapper, price_ref: float, side: Signal, mode: ModeEnum, sl_multiplier_aggressive: float, tp_multiplier_aggressive: float, sl_multiplier_conservative: float, tp_multiplier_conservative: float):
        indicators = Indicators(ohlcv)
        atr = indicators.atr()
        #atr_value = atr[-1]
        
        atr_avg = atr[-1]  # já está suavizado
        
        if mode == ModeEnum.AGGRESSIVE:
            sl_dist = sl_multiplier_aggressive * atr_avg
            tp_dist = tp_multiplier_aggressive * atr_avg
        else:
            sl_dist = sl_multiplier_conservative * atr_avg
            tp_dist = tp_multiplier_conservative * atr_avg

        if side == Signal.BUY:
            sl = price_ref - sl_dist
            tp = price_ref + tp_dist
        else:
            sl = price_ref + sl_dist
            tp = price_ref - tp_dist

        #print(f"entry_price: {price_ref} SL: {sl} TP:{tp}")
        return sl, tp
    
    @staticmethod
    def passes_volume_volatility_filter(ohlcv: OhlcvWrapper, symbol: (str | None), volume_threshold_ratio: float, atr_threshold_ratio: float):
        indicators = Indicators(ohlcv)
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

        indicators_htf = Indicators(ohlcv_higher)
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
    def stochastic(ohlcv: OhlcvWrapper) -> Signal:

        stoch_k, stoch_d = Indicators(ohlcv).stochastic()
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
        indicators = Indicators(ohlcv)

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
    def is_weak_confirmation_candle(candle: Ohlcv, min_body_ratio: float = 0.3) -> bool:
        """
        Verifica se o candle é fraco como confirmação (corpo pequeno em relação ao range total).
        
        Um corpo pequeno (ex: <30% do range total) sugere indecisão, fraqueza ou falta de convicção.

        Args:
            candle (Ohlcv): O candle a ser analisado.
            min_body_ratio (float): Valor mínimo aceitável para o corpo/range (ex: 0.3 = 30%).

        Returns:
            bool: True se o candle for considerado fraco (corpo pequeno), False caso contrário.
        """
        high = candle.high
        low = candle.low
        open_ = candle.open
        close = candle.close

        range_total = high - low
        body_size = abs(close - open_)

        if range_total == 0:
            return True  # evitar divisão por zero, assume candle fraco

        body_ratio = body_size / range_total
        return body_ratio < min_body_ratio

    @staticmethod
    def find_pivots(ohlcv: OhlcvWrapper, left: int = 3, right: int = 3) -> Tuple[List[int], List[int]]:
        """
        Detecta pivôs locais de alta (topos) e baixa (fundos) no gráfico.
        Retorna dois arrays de índices: (pivots_high, pivots_low)
        """
        highs = ohlcv.highs
        lows = ohlcv.lows
        length = len(highs)
        pivots_high = []
        pivots_low = []

        for i in range(left, length - right):
            high_candidate = highs[i]
            is_pivot_high = all(high_candidate > highs[j] for j in range(i - left, i)) and \
                            all(high_candidate > highs[j] for j in range(i + 1, i + right + 1))
            if is_pivot_high:
                pivots_high.append(i)

            low_candidate = lows[i]
            is_pivot_low = all(low_candidate < lows[j] for j in range(i - left, i)) and \
                           all(low_candidate < lows[j] for j in range(i + 1, i + right + 1))
            if is_pivot_low:
                pivots_low.append(i)

        return pivots_high, pivots_low

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

        indicators = Indicators(ohlcv)
        rsi = list(indicators.rsi(14))

        pivots_high, pivots_low = StrategyUtils.find_pivots(ohlcv)

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
    def get_candle_type(candle: Ohlcv) -> str:
        close_price = candle.close
        open_price = candle.open
        high_price = candle.high
        low_price = candle.low

        body = abs(close_price - open_price)
        candle_range = high_price - low_price
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price

        # Evitar divisão por zero
        if candle_range == 0:
            return "doji"

        body_ratio = body / candle_range
        upper_ratio = upper_wick / candle_range
        lower_ratio = lower_wick / candle_range

        # Classificações básicas
        if body_ratio < 0.1:
            return "doji"
        if upper_ratio < 0.1 and lower_ratio > 0.6:
            return "hammer"
        if lower_ratio < 0.1 and upper_ratio > 0.6:
            return "shooting_star"
        if body_ratio > 0.6 and close_price > open_price:
            return "bullish"
        if body_ratio > 0.6 and close_price < open_price:
            return "bearish"
        if body_ratio > 0.5 and upper_ratio > 0.2 and lower_ratio > 0.2:
            return "marubozu"

        return "neutral"


    @staticmethod
    def has_large_wick(candle, ratio: float = 2.0) -> bool:
        body = abs(candle.close - candle.open)
        upper_wick = candle.high - max(candle.close, candle.open)
        lower_wick = min(candle.close, candle.open) - candle.low
        return upper_wick > body * ratio or lower_wick > body * ratio

    @staticmethod
    def is_single_candle_pump(candles: OhlcvWrapper, threshold: float = 0.03) -> bool:
        current = candles.get_current_candle()
        move = abs(current.close - current.open) / current.open
        return move > threshold

    @staticmethod
    def has_price_gap(candles: OhlcvWrapper, threshold: float = 0.02) -> bool:
        if len(candles) < 2:
            return False
        prev = candles.get_previous_candle()
        current = candles.get_current_candle()
        gap = abs(current.open - prev.close) / prev.close
        return gap > threshold

    @staticmethod
    def is_market_manipulation(candles: OhlcvWrapper) -> bool:
        current = candles.get_current_candle()
        return (
            StrategyUtils.has_large_wick(current)
            or StrategyUtils.is_single_candle_pump(candles)
            or StrategyUtils.has_price_gap(candles)
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
        indicators = Indicators(candles)
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
    def is_exhaustion_candle(candles: OhlcvWrapper, lookback: int = 20, threshold: float = 0.95) -> tuple[bool, bool]:
        """
        Verifica se o candle atual está em zona de exaustão:
        - Topo (para penalizar BUY)
        - Fundo (para penalizar SELL)

        Parâmetros:
        - candles: OhlcvWrapper com dados OHLCV
        - lookback: Número de candles anteriores a considerar (excluindo o atual)
        - threshold: Percentil acima/abaixo do qual se considera exaustão
        """
        if len(candles) < lookback + 1:
            return False, False

        # Últimos N candles fechados
        recent_candles = candles.get_recent_closed(lookback=lookback)

        highs = [c.high for c in recent_candles]
        lows = [c.low for c in recent_candles]

        range_high = max(highs)
        range_low = min(lows)

        current_close = candles.get_current_candle().close

        # Evita divisão por zero se todos os candles forem flat
        if range_high == range_low:
            return False, False

        relative_position = (current_close - range_low) / (range_high - range_low)

        is_top_exhaustion = relative_position >= threshold
        is_bottom_exhaustion = relative_position <= (1 - threshold)

        return is_top_exhaustion, is_bottom_exhaustion
    
    @staticmethod
    def is_abnormal_volume(candles: OhlcvWrapper, lookback: int = 20, threshold: float = 2.0) -> bool:
        if len(candles) < lookback + 1:
            return False

        recent = candles.get_recent_closed(lookback)
        volumes = [c.volume for c in recent]

        avg_volume = sum(volumes) / len(volumes)
        current_volume = candles.get_current_candle().volume

        return current_volume > (avg_volume * threshold)
        
        
        

        
        