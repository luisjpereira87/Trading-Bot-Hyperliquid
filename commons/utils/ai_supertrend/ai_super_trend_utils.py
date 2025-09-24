from typing import List

import numpy as np

from commons.enums.signal_enum import Signal
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from commons.utils.strategies.price_action_utils import PriceActionUtils
from commons.utils.strategies.support_resistance_utils import \
    SupportResistanceUtils
from commons.utils.strategies.trend_utils import TrendUtils


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
       

        supertrend_smooth = self.indicators.ema_array(supertrend, smooth_period)

        return supertrend, trend, final_upperband, final_lowerband, supertrend_smooth
    
    def get_trend_signal(self):

        closes = self.ohlcv.closes
        opens = self.ohlcv.opens
        n = len(closes)
        trend_signal = [Signal.HOLD] * n

        _, _, final_upperband, final_lowerband, _ = self.get_supertrend()

        for i in range(2, n):
            # BUY: o candle fecha acima da banda superior, indicando mudança de tendência para alta
            if closes[i-1] <= final_upperband[i-1] and closes[i] > final_upperband[i]:
                trend_signal[i] = Signal.BUY

            # SELL: o candle fecha abaixo da banda inferior, indicando mudança de tendência para baixa
            elif closes[i-1] >= final_lowerband[i-1] and closes[i] < final_lowerband[i]:
                trend_signal[i] = Signal.SELL
        
        return trend_signal
    
    def get_bands_cross_signal(self):

        _, trend, final_upperband, final_lowerband, _ = self.get_supertrend()

        n = len(self.ohlcv.closes)
        trend_signal_filtered = [Signal.HOLD] * n

        for i in range(2, n):
            # Confirmação de mudança de tendência
            if trend[i-2] == -1 and trend[i-1] == 1 and trend[i] == 1:
                trend_signal_filtered[i] = Signal.BUY
            elif trend[i-2] == 1 and trend[i-1] == -1 and trend[i] == -1:
                trend_signal_filtered[i] = Signal.SELL

        return trend_signal_filtered
    

    def get_ema_cross_signal(self):
        closes = self.ohlcv.closes
        opens = self.ohlcv.opens
        highs = self.ohlcv.highs
        lows = self.ohlcv.lows
        n = len(closes)

        trend_signal = [Signal.HOLD] * n

        last_signal = None
        bands_cross_signal = self.get_bands_cross_signal()
        active_trend = None
        ema200 = self.indicators.ema(200)
        ema50 = self.indicators.ema(50)
        ema21 = self.indicators.ema(21)
        macd_line, signal_line = self.indicators.macd(fast_period=3, slow_period=10, signal_period=16)

        for i in range(1, n):

            current_signal = None
            ema_dist_prev = ema21[i-1] - ema50[i-1]
            ema_dist = ema21[i] - ema50[i]

            if last_signal == Signal.SELL and macd_line[i-1] < signal_line[i-1] and macd_line[i] > signal_line[i] and signal_line[i] < 0:
                current_signal = Signal.CLOSE
            elif last_signal == Signal.BUY and macd_line[i-1] > signal_line[i-1] and macd_line[i] < signal_line[i] and signal_line[i] > 0:
                current_signal = Signal.CLOSE

            spread = abs(ema21[i] - ema50[i])
            price = closes[i]
            # Normaliza o spread em relação ao preço (percentagem)
            spread_pct = spread / price

            if ema_dist_prev <= 0 and ema_dist > 0:
                active_trend = Signal.BUY
            elif ema_dist_prev >= 0 and ema_dist < 0:
                active_trend = Signal.SELL

            if bands_cross_signal[i] == Signal.BUY:
                active_trend = Signal.BUY
            elif bands_cross_signal[i] == Signal.SELL:
                active_trend = Signal.SELL

            if active_trend == Signal.BUY and spread_pct > 0.003 and closes[i] > ema200[i]: 
                current_signal = Signal.BUY
            elif active_trend == Signal.SELL and spread_pct > 0.003 and closes[i] < ema200[i]:
                current_signal = Signal.SELL
            
            # --- Regista sinal apenas se diferente do último ---
            if current_signal is not None and current_signal != last_signal :
                trend_signal[i] = current_signal
                last_signal = current_signal

        return trend_signal
