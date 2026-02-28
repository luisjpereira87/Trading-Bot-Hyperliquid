import math
from collections import deque

import numpy as np
import pandas as pd

from commons.utils.indicators.base_indicators_utils import BaseIndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper


class CustomIndicatorsUtils(BaseIndicatorsUtils):
    def __init__(self, ohlcv: OhlcvWrapper, mode='ta'):
        super().__init__(ohlcv, mode)
        self.ohlcv = ohlcv
        self.mode = mode
        self.opens = ohlcv.opens
        self.highs = ohlcv.highs
        self.lows = ohlcv.lows
        self.closes = ohlcv.closes
        self.volumes = ohlcv.volumes


    def smooth_band(self, raw_data, period):
            # Encontrar o primeiro índice onde o valor não é zero nem NaN
            start_idx = 0
            for i, val in enumerate(raw_data):
                if val != 0 and not np.isnan(val):
                    start_idx = i
                    break
            
            # Criar uma cópia para não estragar a original
            smoothed = np.array(raw_data, dtype=float)
            
            # Aplicar a EMA apenas a partir de onde os dados existem
            if start_idx < len(raw_data):
                # O trecho com dados reais
                valid_data = smoothed[start_idx:]
                # Usamos o teu método ema_list no trecho válido
                smoothed_valid = self.ema_list(valid_data, period)
                # Repomos os dados suavizados na lista final
                smoothed[start_idx:] = smoothed_valid
                
            return smoothed

    
    def double_bb_rsi_logic(self, bb_short_period = 20, bb_long_period = 80):
        opens, highs, lows, closes = self.opens, self.highs, self.lows, self.closes
        n = len(closes)
        indices = np.arange(n)
        
        # 1. Calcula as bandas "brutas"
        bb20_up_raw, bb20_basis_raw, bb20_low_raw = self.bollinger_bands(period=bb_short_period, std_dev=2.0)
        bb80_up_raw, bb80_basis_raw, bb80_low_raw = self.bollinger_bands(period=bb_long_period, std_dev=2.25)

        # 2. Suavização Inteligente (Evitando o problema do zero inicial)
        

        # 3. Aplicar a suavização corrigida
        # 1. Obter Desvio Padrão (necessário para as bandas)
        # Vamos usar a SMA20 apenas para calcular a volatilidade (desvio)
        std_20 = np.array([np.std(closes[max(0, i-19):i+1]) for i in range(n)])
        
        # 2. Calcular a Base usando Kalman (em vez de SMA)
        # O script faz: ltfBasis = kalman_filter(close, ltfLength)
        ltf_basis = self.kalman_filter(closes)
        
        # 3. Gerar as Bandas Superior e Inferior
        ltf_mult = 2.0
        bb20_up_raw = ltf_basis + (ltf_mult * std_20)
        bb20_low_raw = ltf_basis - (ltf_mult * std_20)
        
        # 4. Suavização Final (Como fizemos antes)
        bb20_up = self.smooth_band(bb20_up_raw, 5)
        bb20_low = self.smooth_band(bb20_low_raw, 5)

        bb80_up = self.smooth_band(bb80_up_raw, 20)
        bb80_low = self.smooth_band(bb80_low_raw, 20)

        super_score, ema_score = self.calculate_super_score()

        in_extreme_zone_bull = False
        in_extreme_zone_bear = False
        
        in_extreme_zone_bull = False
        in_extreme_zone_bear = False

        # Listas para Sinais de Contexto (Setas)
        context_buy_idx, context_buy_val = [], []
        context_sell_idx, context_sell_val = [], []

        # Listas para Sinais de Entrada Real (Círculos)
        entry_buy_idx, entry_buy_val = [], []
        entry_sell_idx, entry_sell_val = [], []

        trend_buy_idx, trend_buy_val = [], []
        trend_sell_idx, trend_sell_val = [], []

        rsi_gap_threshold = 2.0
        signals = np.zeros(n)
        bull_osc, bear_osc, slope_ma_14, polarity_osc, signal_line = self.calculate_lux_bb_oscillator()

        # Inicializamos os estados fora do loop
        is_bull = False
        is_bear = False
        is_trigger = False # Esta variável controla a repetição
        for i in range(1, n):

            # 1. Definimos o Regime (O Caminho)
            # Só mudamos o regime se houver força suficiente (> 20)
            if bull_osc[i] > bear_osc[i] and bull_osc[i] > 20:
                if not is_bull: # Se mudou de bear para bull agora
                    is_bull = True
                    is_bear = False
                    is_trigger = False # Reset para permitir novo sinal neste novo regime
            elif bear_osc[i] > bull_osc[i] and bear_osc[i] > 20:
                if not is_bear: # Se mudou de bull para bear agora
                    is_bear = True
                    is_bull = False
                    is_trigger = False # Reset para permitir novo sinal
            else:
                # Se cair na zona morta, limpamos tudo
                is_bull = False
                is_bear = False
                is_trigger = False

            # 2. Condições de Momentum (as tuas regras)
            is_bullish_momentum = is_bull and bull_osc[i] > signal_line[i] and signal_line[i] > 20 and not is_trigger
            is_bearish_momentum = is_bear and bear_osc[i] > signal_line[i] and signal_line[i] > 20 and not is_trigger

            # REVERSÃO PARA ALTA (BULL)
            # Se o oscilador bull começar a subir e cruzar um threshold (ex: 20 ou 50)
            if is_bullish_momentum:
                trend_buy_idx.append(i)
                trend_buy_val.append(lows[i])
                signals[i] = 1
                is_trigger = True

            # REVERSÃO PARA BAIXA (BEAR)
            if is_bearish_momentum:
                trend_sell_idx.append(i)
                trend_sell_val.append(highs[i])
                signals[i] = -1
                is_trigger = True

            # --- PASSO 1: CONTEXTO (SETAS) ---
            if lows[i] < bb80_low[i] and not in_extreme_zone_bull:
                context_buy_idx.append(i)
                context_buy_val.append(lows[i])
                in_extreme_zone_bull = True
            
            if highs[i] > bb80_up[i] and not in_extreme_zone_bear:
                context_sell_idx.append(i)
                context_sell_val.append(highs[i])
                in_extreme_zone_bear = True

            # --- PASSO 2: ENTRADA REAL (CÍRCULOS) ---
            if in_extreme_zone_bull:
                # Gatilho: Fecho acima do meio do candle anterior + RSI a subir
                if super_score[i] > ema_score[i] and ema_score[i] > 0:
                    entry_buy_idx.append(i)
                    entry_buy_val.append(lows[i] * 0.997) # Ligeiramente abaixo para não sobrepor
                   
                    in_extreme_zone_bull = False 

            elif in_extreme_zone_bear:
                # Gatilho: Fecho abaixo do meio do candle anterior + RSI a descer
                if super_score[i] < ema_score[i] and ema_score[i] < 0:
                    entry_sell_idx.append(i)
                    entry_sell_val.append(highs[i] * 1.003) # Ligeiramente acima
                    
                    in_extreme_zone_bear = False

            # --- RESET DE SEGURANÇA ---
            #if closes[i] > ltf_basis[i]: in_extreme_zone_bull = False
            #if closes[i] < ltf_basis[i]: in_extreme_zone_bear = False

        return {
            'bbshort_up': bb20_up,
            'bbshort_low': bb20_low,
            'bblong_up': bb80_up,
            'bblong_low': bb80_low,
            'signals': signals,
            'entry_buy_idx': entry_buy_idx,
            'entry_buy_val': entry_buy_val,
            'entry_sell_idx': entry_sell_idx,
            'entry_sell_val': entry_sell_val,
            'context_buy_idx': trend_buy_idx,
            'context_buy_val': trend_buy_val,
            'context_sell_idx': trend_sell_idx,
            'context_sell_val': trend_sell_val
        }
    
    def calculate_super_score(self, smooth_period=5):
        # 1. Obter indicadores
        rsi14, _ = self.rsi(14)
        rsi8, _ = self.rsi(8)
        macd, macd_signal, macd_hist = self.macd(12, 26, 9)
        stoch_k, stoch_d = self.stochastic(14, 3)
        adx = self.adx(14)
        psar = self.psar()
        
        n = len(self.closes)
        final_scores = np.zeros(n)

        # Definimos os pesos máximos para podermos normalizar depois
        # Peso total = 15+15 (RSIs) + 20+20 (MACD) + 30 (Stoch) = 100
        for i in range(1, n):
            raw_score = 0
            
            # --- RSI CONFLUENCE (30 pts) ---
            if rsi14[i] > 50: raw_score += 10
            elif rsi14[i] < 50: raw_score -= 10
            
            if rsi8[i] > 50: raw_score += 10
            elif rsi8[i] < 50: raw_score -= 10

            # --- MACD CONFLUENCE (40 pts) ---
            if macd[i] > macd_signal[i]: raw_score += 15
            elif macd[i] < macd_signal[i]: raw_score -= 15
            
            if macd_hist[i] > 0: raw_score += 15
            elif macd_hist[i] < 0: raw_score -= 15

            # --- STOCHASTIC (30 pts) ---
            if stoch_k[i] > stoch_d[i]: raw_score += 25
            elif stoch_k[i] < stoch_d[i]: raw_score -= 25

              # --- NOVO: PARABOLIC SAR (Peso: 25 pts) ---
            # Se o PSAR está ABAIXO do preço (Tendência de Alta)
            if psar[i] < self.closes[i]:
                raw_score += 25
            # Se o PSAR está ACIMA do preço (Tendência de Baixa)
            else:
                raw_score -= 25

            # --- FILTRO DE ADX (A "SAÚDE" DO SINAL) ---
            # Se o ADX for baixo (<20), o sinal é fraco por falta de tendência.
            # Reduzimos o score em 50% para evitar entradas em "choppy market"
            if adx[i] < 20:
                raw_score *= 0.8
            
            # Como o nosso raw_score máximo possível é 100 (15+15+20+20+30),
            # ele já está na escala de -100 a 100. 
            final_scores[i] = raw_score

            smooth_scores = self.ema_list(final_scores, smooth_period)

        return final_scores, smooth_scores
    
    def calculate_lux_bb_oscillator(self, length=14, mult=1.0, signal_length=7):
        src = self.ohlcv.closes
        
        # 1. Cálculos Base (EMA e Desvio Padrão)
        # Nota: LuxAlgo usa EMA aqui, o que é mais rápido que SMA
        ema = np.array(pd.Series(src).ewm(span=length, adjust=False).mean().values)
        stdev = np.array(pd.Series(src).rolling(window=length).std(ddof=0).values) * mult

        # 1. Primeiro calculas a EMA (que já faz parte do LuxAlgo)
        ema_series = pd.Series(src).ewm(span=length, adjust=False).mean()

        # 2. Depois calculas a inclinação (Slope)
        # O slope é a diferença entre o valor atual e o anterior
        # Usamos uma média de 14 para suavizar e eliminar o ruído
        slope_ma_14 = ema_series.diff().rolling(window=length).mean().values
        
        upper = ema + stdev
        lower = ema - stdev
        
        n = len(src)
        bull_pct = np.zeros(n)
        bear_pct = np.zeros(n)
        
        # 2. Loop para a soma móvel (equivalente ao 'for' do Pinescript)
        for i in range(length, n):
            # Janela de análise de 'length' candles
            src_window = src[i-length+1 : i+1]
            upper_window = upper[i-length+1 : i+1]
            lower_window = lower[i-length+1 : i+1]
            
            # bull += math.max(src[i] - upper[i], 0)
            bull_val = np.sum(np.maximum(src_window - upper_window, 0))
            # bear += math.max(lower[i] - src[i], 0)
            bear_val = np.sum(np.maximum(lower_window - src_window, 0))
            
            # denominadores (soma absoluta das distâncias)
            bull_den = np.sum(np.abs(src_window - upper_window))
            bear_den = np.sum(np.abs(lower_window - src_window))
            
            # Cálculo final com proteção contra divisão por zero
            bull_pct[i] = (bull_val / bull_den * 100) if bull_den != 0 else 0
            bear_pct[i] = (bear_val / bear_den * 100) if bear_den != 0 else 0


        # Unificamos: Bull fica positivo, Bear fica negativo
        # Se bull é 15 e bear é 0 -> Resultado 15
        # Se bull é 0 e bear é 20 -> Resultado -20
        polarity_osc = bull_pct - bear_pct

        # --- 1. Bandas de Bollinger para Volatilidade ---
        # Usar 14 ou 20 para o BBW ser estável mas reagir ao preço
        upper, basis, lower = self.bollinger_bands() 
        upper, basis, lower = np.array(upper), np.array(basis), np.array(lower)

        # 1. BBW Puro (Padrão TradingView)
        # Se usares length 20, ele segue o standard
        bbw_puro = ((upper - lower) / basis) * 1000

        # 2. Aplicação da Polaridade sem filtros (Trigger Removido)
        dynamic_signal = []

        for i in range(len(polarity_osc)):
            # O sinal apenas espelha o BBW de acordo com o lado do oscilador
            if polarity_osc[i] >= 0:
                dynamic_signal.append(bbw_puro[i])
            else:
                dynamic_signal.append(-bbw_puro[i])

        dynamic_signal = np.array(dynamic_signal)
            
        return bull_pct, bear_pct, slope_ma_14, polarity_osc, bbw_puro
        