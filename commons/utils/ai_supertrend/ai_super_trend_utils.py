from typing import List

import numpy as np

from commons.enums.signal_enum import Signal
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from commons.utils.strategies.support_resistance_utils import \
    SupportResistanceUtils


class AISuperTrendUtils:
    def __init__(self, ohlcv: OhlcvWrapper):
        self.ohlcv = ohlcv  # instância de OhlcvWrapper
        self.indicators = IndicatorsUtils(ohlcv)

    def get_supertrend(self,
                   mode="adaptive",
                   base_length=10, base_mult=3.0,
                   min_len=7, max_len=21,
                   min_mult=2.0, max_mult=4.0,
                   vol_sensitivity=1.5,
                   trend_confirmation=1,
                   smooth_period=3):

        opens, highs, lows, closes, volumes = self.ohlcv.candles_to_arrays()
        hl2 = (highs + lows) / 2
        n = len(closes)

        # ATR fixo para medir volatilidade
        atr_fixed = self.indicators.atr(14)
        vol_rel = atr_fixed / closes

        # calcular arrays de length e multiplier
        if mode == "adaptive":
            atr_len = base_length * (1 / (1 + vol_sensitivity * vol_rel))
            atr_len = np.clip(atr_len, min_len, max_len)
            atr_mult = base_mult * (1 + vol_sensitivity * vol_rel)
            atr_mult = np.clip(atr_mult, min_mult, max_mult)

            # suavizar com numpy (rolling mean simples)
            def rolling_mean(arr, window=3):
                if window <= 1:
                    return arr
                kernel = np.ones(window) / window
                return np.convolve(arr, kernel, mode="same")

            atr_len = rolling_mean(atr_len, 3)
            atr_mult = rolling_mean(atr_mult, 3)
        else:
            atr_len = np.full(n, base_length)
            atr_mult = np.full(n, base_mult)

        # pré-calcular ATRs de todos os comprimentos possíveis
        atr_cache = {L: self.indicators.atr(L) for L in range(min_len, max_len + 1)}

        supertrend = np.zeros(n)
        trend = np.ones(n)
        final_upperband = np.zeros(n)
        final_lowerband = np.zeros(n)
        trend_count = 0

        for i in range(n):
            length = int(round(atr_len[i]))
            mult = atr_mult[i]
            atr_curr = atr_cache[length][i]  # usar cache

            if np.isnan(atr_curr):
                supertrend[i] = np.nan
                trend[i] = 1
                final_upperband[i] = np.nan
                final_lowerband[i] = np.nan
                continue

            upperband = hl2[i] + mult * atr_curr
            lowerband = hl2[i] - mult * atr_curr

            if i == 0 or np.isnan(supertrend[i - 1]):
                supertrend[i] = hl2[i]
                trend[i] = 1
                final_upperband[i] = upperband
                final_lowerband[i] = lowerband
                trend_count = 1
                continue

            final_upperband[i] = upperband if (upperband < final_upperband[i - 1] or closes[i - 1] > final_upperband[i - 1]) else final_upperband[i - 1]
            final_lowerband[i] = lowerband if (lowerband > final_lowerband[i - 1] or closes[i - 1] < final_lowerband[i - 1]) else final_lowerband[i - 1]

            prev_trend = trend[i - 1]
            if prev_trend == 1:
                if closes[i] <= final_upperband[i]:
                    trend_count += 1
                    if trend_count >= trend_confirmation:
                        trend[i] = -1
                        supertrend[i] = final_upperband[i]
                        trend_count = 0
                    else:
                        trend[i] = prev_trend
                        supertrend[i] = supertrend[i - 1]
                else:
                    trend[i] = 1
                    supertrend[i] = final_lowerband[i]
                    trend_count = 0
            else:
                if closes[i] >= final_lowerband[i]:
                    trend_count += 1
                    if trend_count >= trend_confirmation:
                        trend[i] = 1
                        supertrend[i] = final_lowerband[i]
                        trend_count = 0
                    else:
                        trend[i] = prev_trend
                        supertrend[i] = supertrend[i - 1]
                else:
                    trend[i] = -1
                    supertrend[i] = final_upperband[i]
                    trend_count = 0
       

        trend_signal = [Signal.HOLD] * n
        trend_signal_filtered = [Signal.HOLD] * n
        trend_signal_scored = [Signal.HOLD] * n         # sinais com score (EMA, distância, etc.)
        trend_score = np.zeros(n) 

        for i in range(1, n):
            # BUY: o candle fecha acima da banda superior, indicando mudança de tendência para alta
            if closes[i-1] <= final_upperband[i-1] and closes[i] > final_upperband[i]:
                trend_signal[i] = Signal.BUY

            # SELL: o candle fecha abaixo da banda inferior, indicando mudança de tendência para baixa
            elif closes[i-1] >= final_lowerband[i-1] and closes[i] < final_lowerband[i]:
                trend_signal[i] = Signal.SELL

      
        for i in range(2, n):
            # Confirmação de mudança de tendência
            if trend[i-2] == -1 and trend[i-1] == 1 and trend[i] == 1:
                trend_signal_filtered[i] = Signal.BUY
            elif trend[i-2] == 1 and trend[i-1] == -1 and trend[i] == -1:
                trend_signal_filtered[i] = Signal.SELL

        #############################################
        # --- Confirmação/score com EMA(21) ---
        trend_signal_cross = [Signal.HOLD] * n
        ema21 = self.indicators.ema(21)
        min_slope_pct = 0.0005
        large_body_pct = 0.01  # 1% do preço
        slope = ema21[i] - ema21[i-1]

        last_signal = None  # inicializa vazio
        for i in range(1, n):
            slope = ema21[i] - ema21[i-1]
            slope_threshold = min_slope_pct * ema21[i]

            body_high = max(opens[i], closes[i])
            body_low = min(opens[i], closes[i])
            body_size = body_high - body_low


            if body_size == 0:  # doji
               continue

            # quanto do corpo está acima/abaixo da EMA
            above = max(0, body_high - ema21[i])
            below = max(0, ema21[i] - body_low)

            above_ratio = above / body_size
            below_ratio = below / body_size

            current_signal = None

            # caso 1: corpo maioritariamente acima
            if (above_ratio > 0.5 and slope > slope_threshold):
                current_signal = Signal.BUY

            # caso 2: corpo maioritariamente abaixo
            elif (below_ratio > 0.5 and slope < -slope_threshold):
                current_signal = Signal.SELL

            # só regista se for diferente do último
            if current_signal is not None and current_signal != last_signal:
                trend_signal_cross[i] = current_signal
                last_signal = current_signal

        supertrend_smooth = self.indicators.ema_array(supertrend, smooth_period)


        return supertrend, trend, final_upperband, final_lowerband, supertrend_smooth, trend_signal_cross, trend_signal_filtered
    
    def _calculate_signal_strength(self, i, closes, opens, lows, volumes, ema21):
        score = 0

        # 1. Tamanho do corpo do candle
        body = abs(closes[i] - opens[i])
        avg_body = np.mean([abs(closes[j] - opens[j]) for j in range(i-20, i)]) if i >= 20 else body
        if body > avg_body:
            score += 1

        # 2. Volume acima da média
        avg_volume = np.mean(volumes[i-20:i]) if i >= 20 else volumes[i]
        if volumes[i] > avg_volume:
            score += 1

        # 3. Distância da EMA21
        distance = abs(closes[i] - ema21[i]) / ema21[i]
        if distance < 0.02:  # menos de 2% de distância → saudável
            score += 1

        return score