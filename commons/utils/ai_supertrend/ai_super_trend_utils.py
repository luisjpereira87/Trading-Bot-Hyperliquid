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
        opens = self.ohlcv.opens
        highs = self.ohlcv.highs
        lows = self.ohlcv.lows
        n = len(closes)

        trend_signal = [Signal.HOLD] * n

        last_signal = None
        bands_cross_signal = self.get_bands_cross_signal()
        _, trend, final_upperband, final_lowerband, _ = self.indicators.supertrend()
        active_trend = None
        active_supertrend = None
        ema200 = self.indicators.ema(200)
        ema50 = self.indicators.ema(50)
        ema21 = self.indicators.ema(21)
        ema9 = self.indicators.ema(9)
        psar = self.indicators.psar()
        atr = self.indicators.atr()
        macd_line, signal_line = self.indicators.macd(3,10,16)
        entry_price = 0
        profits = []
        fee_rate = 0.0004  # 0.04% Hyperliquid round trip
        min_profit_threshold = 0.001
        for i in range(1, n):

            current_signal = None
            ema_dist_prev = ema21[i-1] - ema50[i-1]
            ema_dist = ema21[i] - ema50[i]
            fees_min = entry_price * fee_rate

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
            
            # --- Saída por cruzamento de bandas ---
            if last_signal == Signal.BUY and bands_cross_signal[i] != Signal.BUY and closes[i] >= final_upperband[i] and current_profit_pct > min_profit_threshold:
                current_signal = Signal.CLOSE
            elif last_signal == Signal.SELL and bands_cross_signal[i] != Signal.SELL and closes[i] <= final_lowerband[i] and current_profit_pct > min_profit_threshold:
                current_signal = Signal.CLOSE
            
            # --- Detecção de tendência via EMA ---
            spread = abs(ema21[i] - ema50[i])
            spread_pct = spread / closes[i]

            spread_fast = abs(ema9[i] - ema21[i])
            spread_fast_pct = spread_fast / closes[i]

            #print("AQUIII", i, spread_pct, spread_fast_pct, abs(spread_pct-spread_fast_pct))

            if ema_dist_prev <= 0 and ema_dist > 0:
                active_trend = Signal.BUY
            elif ema_dist_prev >= 0 and ema_dist < 0:
                active_trend = Signal.SELL

            
            if bands_cross_signal[i] == Signal.BUY and ema21[i] > ema50[i]:
                active_supertrend = Signal.BUY
            elif bands_cross_signal[i] == Signal.SELL and ema21[i] < ema50[i]:
                active_supertrend = Signal.SELL

            
            _, profile, ema_spread  = self.get_volatility_profile(atr)

            fast_trigger_signal = self.fast_trigger_signal(i, macd_line, signal_line, closes[i],ema21[i], ema50[i], ema200[i])

            reforce_buy_signal = spread_pct > ema_spread and closes[i] > ema200[i] and closes[i] > psar[i]
            reforce_sell_signal = spread_pct > ema_spread and closes[i] < ema200[i] and closes[i] < psar[i]

            
             
            if (active_trend == Signal.BUY or fast_trigger_signal == Signal.BUY) and reforce_buy_signal: 
                current_signal = Signal.BUY
            elif(active_trend == Signal.SELL or fast_trigger_signal == Signal.SELL) and reforce_sell_signal:
                current_signal = Signal.SELL

            active_supertrend = None
            # --- Regista sinal apenas se diferente do último ---
            if current_signal is not None and current_signal != last_signal:

                if trend_signal[i-1] == Signal.CLOSE:
                    last_signal = None

                trend_signal[i] = current_signal
                last_signal = current_signal
                entry_price = closes[i]
                active_trend = None
                profits = []

        return trend_signal
    
    def get_macro_trend(
        self, 
        close: float, 
        ema21: float, 
        ema50: float, 
        ema200: float, 
        is_conservative=True):

        if not is_conservative:
            return Signal.BUY if close > ema200 else Signal.SELL
        else:
            if ema21 > ema50 > ema200:
                return Signal.BUY
            elif ema21 < ema50 < ema200:
                return Signal.SELL
            else:
                return Signal.HOLD
    
    def fast_trigger_signal(
        self, 
        index: int,
        macd_line: list[float], 
        signal_line: list[float], 
        close: float, 
        ema21: float, 
        ema50: float, 
        ema200: float, 
        is_conservative=True, 
        hist_absolute_threshold=0.0, 
        hist_strength_window=5
    ) -> Signal:
        """
        Retorna sinal de buy/sell baseado em cruzamento MACD, tendência macro e força do histograma.
        """
    
        # Calcula histograma
        hist = macd_line[index] - signal_line[index]

        # Calcula histograma médio dos últimos candles
        if index >= hist_strength_window:
            recent_hist = [macd_line[i] - signal_line[i] for i in range(index - hist_strength_window + 1, index + 1)]
            hist_avg = sum(abs(h) for h in recent_hist) / hist_strength_window
        else:
            hist_avg = abs(hist)
        #print("AQUIII", index,   macd_line[index-1] < macd_line[index] and signal_line[index-1] < signal_line[index])
        # Ignora sinais muito fracos
        if abs(hist) < max(hist_avg * 0.3, hist_absolute_threshold):
            return Signal.HOLD

        # Determina a tendência macro
        if not is_conservative:
            trend_signal = Signal.BUY if close > ema200 else Signal.SELL
        else:
            if ema21 > ema50 > ema200:
                trend_signal = Signal.BUY
            elif ema21 < ema50 < ema200:
                trend_signal = Signal.SELL
            else:
                trend_signal = Signal.HOLD

        # Cruzamento MACD alinhado com a tendência
        if trend_signal == Signal.BUY and macd_line[index-1] < signal_line[index-1] and macd_line[index] > signal_line[index]:
            return Signal.BUY
        elif trend_signal == Signal.SELL and macd_line[index-1] > signal_line[index-1] and macd_line[index] < signal_line[index]:
            return Signal.SELL

        return Signal.HOLD
    
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
        

    



        


