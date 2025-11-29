import logging
from typing import List

import numpy as np

from commons.enums.signal_enum import Signal
from commons.models.ohlcv_type_dclass import Ohlcv
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
    
    def get_trend_signal(self):

        closes = self.ohlcv.closes
        opens = self.ohlcv.opens
        n = len(closes)
        trend_signal = [Signal.HOLD] * n

        _, _, final_upperband, final_lowerband, _ = self.indicators.supertrend()

        for i in range(2, n):
            # BUY: o candle fecha acima da banda superior, indicando mudança de tendência para alta
            if closes[i-1] <= final_upperband[i-1] and closes[i] > final_upperband[i]:
                trend_signal[i] = Signal.BUY

            # SELL: o candle fecha abaixo da banda inferior, indicando mudança de tendência para baixa
            elif closes[i-1] >= final_lowerband[i-1] and closes[i] < final_lowerband[i]:
                trend_signal[i] = Signal.SELL
        
        return trend_signal
    
    def get_bands_cross_signal(self):

        _, trend, final_upperband, final_lowerband, _ = self.indicators.supertrend()

        n = len(self.ohlcv.closes)
        trend_signal_filtered = [Signal.HOLD] * n

        for i in range(2, n):
            # Confirmação de mudança de tendência
            if trend[i-2] == -1 and trend[i-1] == 1 and trend[i] == 1:
                trend_signal_filtered[i] = Signal.BUY
            elif trend[i-2] == 1 and trend[i-1] == -1 and trend[i] == -1:
                trend_signal_filtered[i] = Signal.SELL

        return trend_signal_filtered

    
    
    def get_ema_cross_signal(self, trailing_n = 3):
        closes = self.ohlcv.closes
        #opens = self.ohlcv.opens
        #highs = self.ohlcv.highs
        #lows = self.ohlcv.lows
        n = len(closes)

        trend_signal = [Signal.HOLD] * n

        last_signal = None
        #bands_cross_signal = self.get_bands_cross_signal()
        #supertrend, trend, final_upperband, final_lowerband, _ = self.indicators.supertrend()
        active_trend = None
        active_fast_trend = None
        #active_supertrend = None
        ema200 = self.indicators.ema(200)
        ema50 = self.indicators.ema(50)
        ema21 = self.indicators.ema(21)
        ema9 = self.indicators.ema(9)
        psar = self.indicators.psar()
        atr = self.indicators.atr()
        lateral = self.indicators.detect_low_volatility()
        #signal, stoch_score = self.indicators.get_stoch_signal(k_period=5, d_period=2, lookback=4)
        #rsi_signal, rsi_score, patterns = self.indicators.get_rsi_reversal_signal()
        
        #macd_line, signal_line = self.indicators.macd(3,10,16)
        entry_price = 0
        profits = []
        fee_rate = 0.0004  # 0.04% Hyperliquid round trip
        min_profit_threshold = 0.001
        cross_index = None
        cross_age = 0
        #lookback = 10
        for i in range(1, n):

            current_signal = None
            ema_dist_prev = ema21[i-1] - ema50[i-1]
            ema_dist = ema21[i] - ema50[i]
            fees_min = entry_price * fee_rate

            ema_dist_prev_fast = ema9[i-1] - ema21[i-1]
            ema_dist_fast = ema9[i] - ema21[i]
            

            # Calcula lucro atual
            if last_signal == Signal.BUY:
                current_profit_pct = (closes[i] - entry_price) / entry_price
                profits.append(current_profit_pct)
                
            elif last_signal == Signal.SELL:  # SELL
                current_profit_pct = (entry_price - closes[i]) / entry_price
                profits.append(current_profit_pct)
            
            # --- Saída antecipada com trailing_n e PSAR ---
            if len(profits) >= trailing_n and current_profit_pct > min_profit_threshold:
                # verifica se os últimos N profits estão sempre a descer
                if all(profits[-k] < profits[-(k+1)] for k in range(1, trailing_n)):
                    # validação extra com PSAR
                    if (psar[i] < closes[i] and last_signal == Signal.SELL) or \
                    (psar[i] > closes[i] and last_signal == Signal.BUY):
                        current_signal = Signal.CLOSE
            
            # --- Saida antecipada caso o mercado não tenha direção e aproveita o lucro para sair ---
            profit_pos = sum(1 for x in profits if x > 0)
            profit_neg = sum(1 for x in profits if x < 0)

            if len(profits) >= trailing_n and profit_neg >= profit_pos and current_profit_pct > min_profit_threshold:
                current_signal = Signal.CLOSE
            
            # --- Detecção de tendência via EMA ---
            spread = abs(ema21[i] - ema50[i])
            spread_pct = spread / closes[i]

            spread_fast = abs(ema9[i] - ema21[i])
            spread_fast_pct = spread_fast / closes[i]

            if ema_dist_prev <= 0 and ema_dist > 0:
                active_trend = Signal.BUY
                cross_index = i
                cross_age = 0
            elif ema_dist_prev >= 0 and ema_dist < 0:
                active_trend = Signal.SELL
                cross_index = i
                cross_age = 0
        
            if ema_dist_prev_fast <= 0 and ema_dist_fast > 0:
                active_fast_trend = Signal.BUY
                cross_index = i
                cross_age = 0
            elif ema_dist_prev_fast >= 0 and ema_dist_fast < 0:
                active_fast_trend = Signal.SELL
                cross_index = i
                cross_age = 0
            
            if cross_index is not None:
                cross_age += 1
            
            _, profile, ema_spread  = self.get_volatility_profile(atr)

            mid_ema_buy_signal = active_trend == Signal.BUY and spread_pct > ema_spread and ema21[i] > ema50[i] > ema200[i] and closes[i] > psar[i] and cross_age < 10
            mid_ema_sell_signal = active_trend == Signal.SELL and spread_pct > ema_spread and ema21[i] < ema50[i] < ema200[i] and closes[i] < psar[i] and cross_age < 10


            fast_ema_buy_signal = active_fast_trend == Signal.BUY and spread_fast_pct > ema_spread and ema9[i] > ema21[i] > ema50[i] > ema200[i] and closes[i] > psar[i] and cross_age < 10
            fast_ema_sell_signal = active_fast_trend == Signal.SELL and spread_fast_pct > ema_spread and ema9[i] < ema21[i] < ema50[i] < ema200[i] and closes[i] < psar[i] and cross_age < 10
            """
            reforce_buy_signal = spread_fast_pct > ema_spread and closes[i] > ema50[i] and closes[i] > psar[i]
            reforce_sell_signal = spread_fast_pct > ema_spread and closes[i] < ema50[i] and closes[i] < psar[i]
            """
            #reforce_stoch = stoch_score[i] > 0.5 and signal[i] == active_trend

            #print("AQUIII", i, rsi_signal[i], rsi_score[i])
            if (mid_ema_buy_signal or fast_ema_buy_signal):  
                current_signal = Signal.BUY
            elif (mid_ema_sell_signal or fast_ema_sell_signal):
                current_signal = Signal.SELL

            if lateral[i] and current_signal != Signal.CLOSE:
                current_signal = None

            if current_signal is not None and current_signal != last_signal:

                if trend_signal[i-1] == Signal.CLOSE:
                    last_signal = None

                trend_signal[i] = current_signal
                last_signal = current_signal
                entry_price = closes[i]
                active_trend = None
                profits = []

        return trend_signal

    def get_volatility_profile(self, atr: list[float], lookback: int = 50):
        """
        Mede a volatilidade média do ativo com base no ATR relativo.
        Retorna:
        - atr_rel (float): média do ATR/preço (ex: 0.018 = 1.8%)
        - profile (str): classificação qualitativa ("low", "medium", "high")
        """
        closes = np.array(self.ohlcv.closes)
        #atr = np.array(self.indicators.atr(atr_period))

        # evitar erro se houver poucos dados
        if len(closes) < lookback or len(atr) < lookback:
            atr_rel = atr[-1] / closes[-1] if len(atr) > 0 else 0
        else:
            atr_rel = np.mean(atr[-lookback:]) / closes[-1]

        # classificação qualitativa (ajusta conforme teu mercado)
        if atr_rel < 0.012:
            profile = "low"     # ex: BTC, ETH
            ema_spread = 0.002 
        elif atr_rel < 0.025:
            profile = "medium"  # ex: BNB, AVAX
            ema_spread = 0.003
        else:
            profile = "high"    # ex: SOL, meme coins
            ema_spread = 0.004

        return atr_rel, profile, ema_spread

        

    



        


