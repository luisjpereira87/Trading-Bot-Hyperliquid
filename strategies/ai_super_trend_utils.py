from typing import List

import numpy as np
import pandas as pd

from commons.enums.signal_enum import Signal
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from strategies.indicators import Indicators


class AISuperTrendUtils:
    def __init__(self, ohlcv: OhlcvWrapper):
        self.ohlcv = ohlcv  # instância de OhlcvWrapper
        self.indicators = Indicators(ohlcv)

    def get_supertrend(self,
                   mode="adaptive",
                   base_length=10, base_mult=3.0,
                   min_len=7, max_len=21,
                   min_mult=2.0, max_mult=4.0,
                   vol_sensitivity=1.5,
                   trend_confirmation=1,
                   smooth_period=3):
    
        opens, highs, lows, closes, volumes  = self.ohlcv.candles_to_arrays()
        hl2 = (highs + lows) / 2
        n = len(closes)

        # ATR fixo para medir volatilidade
        atr_fixed = self.indicators.atr(14)
        vol_rel = atr_fixed / closes

        if mode == "adaptive":
            atr_len = np.clip(base_length * (1 / (1 + vol_sensitivity * vol_rel)), min_len, max_len)
            atr_len = pd.Series(atr_len).rolling(3, min_periods=1).mean().to_numpy()
            atr_mult = np.clip(base_mult * (1 + vol_sensitivity * vol_rel), min_mult, max_mult)
            atr_mult = pd.Series(atr_mult).rolling(3, min_periods=1).mean().to_numpy()
        else:
            atr_len = np.full(n, base_length)
            atr_mult = np.full(n, base_mult)

        supertrend = np.zeros(n)
        trend = np.ones(n)
        final_upperband = np.zeros(n)
        final_lowerband = np.zeros(n)
        trend_count = 0

        for i in range(n):
            length = int(round(atr_len[i]))  # <-- conversão segura para inteiro
            mult = atr_mult[i]
            atr_curr = self.indicators.atr(length)[i]

            if np.isnan(atr_curr):
                supertrend[i] = np.nan
                trend[i] = 1
                final_upperband[i] = np.nan
                final_lowerband[i] = np.nan
                continue

            upperband = hl2[i] + mult * atr_curr
            lowerband = hl2[i] - mult * atr_curr

            if i == 0 or np.isnan(supertrend[i-1]):
                supertrend[i] = hl2[i]
                trend[i] = 1
                final_upperband[i] = upperband
                final_lowerband[i] = lowerband
                trend_count = 1
                continue

            final_upperband[i] = upperband if upperband < final_upperband[i-1] or closes[i-1] > final_upperband[i-1] else final_upperband[i-1]
            final_lowerband[i] = lowerband if lowerband > final_lowerband[i-1] or closes[i-1] < final_lowerband[i-1] else final_lowerband[i-1]

            prev_trend = trend[i-1]
            if prev_trend == 1:
                if closes[i] <= final_upperband[i]:
                    trend_count += 1
                    if trend_count >= trend_confirmation:
                        trend[i] = -1
                        supertrend[i] = final_upperband[i]
                        trend_count = 0
                    else:
                        trend[i] = prev_trend
                        supertrend[i] = supertrend[i-1]
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
                        supertrend[i] = supertrend[i-1]
                else:
                    trend[i] = -1
                    supertrend[i] = final_upperband[i]
                    trend_count = 0

        
        rel_threshold = 0.005  # 0.5% do preço, ajusta conforme volatilidade do ativo
        trend_signal = [Signal.HOLD] * len(trend)

        for i in range(2, len(trend)):
            # calcula threshold relativo ao preço do candle anterior
            threshold = closes[i-1] * rel_threshold

            # distância entre bandas
            band_distance = final_upperband[i-1] - final_lowerband[i-1]

            # só gera sinal se bandas suficientemente afastadas
            if band_distance < threshold:
                continue  # ignora sinais, bandas muito próximas

            # início de alta
            if trend[i-2] == -1 and trend[i-1] == 1 and trend[i] == 1:
                if closes[i-1] > final_lowerband[i-1]:
                    trend_signal[i-1] = Signal.BUY

            # início de baixa
            elif trend[i-2] == 1 and trend[i-1] == -1 and trend[i] == -1:
                if closes[i-1] < final_upperband[i-1]:
                    trend_signal[i-1] = Signal.SELL
            

        #print("trend_signal", trend_signal)
        #print("trend", trend)

        supertrend_smooth = self.indicators.ema_array(supertrend, smooth_period)

        return supertrend, trend, final_upperband, final_lowerband, supertrend_smooth, trend_signal