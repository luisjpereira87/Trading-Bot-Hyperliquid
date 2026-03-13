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

    def calculate_gap(self, array1, array2, max_range=100):
        gaps = [abs(o - s) for o, s in zip(array1, array2)]
        n = len(self.closes)
        
        gap_index = np.zeros(n)
        for i in range(1, n):
            if i < max_range:
                gap_index[i] = 0
                continue
                
            # 4.1. Pegar na janela histórica de Gaps
            window = gaps[i-max_range : i+1]
            min_gap = min(window)
            max_gap = max(window)
            
            # 4.2. Normalizar o Gap atual para a escala 0-100
            if max_gap - min_gap == 0:
                index = 0
            else:
                # Formula: (Valor - Min) / (Max - Min) * 100
                index = ((gaps[i] - min_gap) / (max_gap - min_gap)) * 100
                
            gap_index[i] = index
        
        return gap_index
    
    def double_bb_rsi_logic(self, bb_short_period = 20, bb_long_period = 80, bb_short_std_dev = 2.0, bb_long_std_dev = 2.25):
        opens, highs, lows, closes = self.opens, self.highs, self.lows, self.closes
        n = len(closes)
        indices = np.arange(n)
        
        # 1. Calcula as bandas "brutas"
        bb20_up_raw, bb20_basis_raw, bb20_low_raw = self.bollinger_bands(period=bb_short_period, std_dev=bb_short_std_dev)
        bb80_up_raw, bb80_basis_raw, bb80_low_raw = self.bollinger_bands(period=bb_long_period, std_dev=bb_long_std_dev)

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

        # Listas para Sinais de Entrada Real (Círculos)
        entry_buy_idx, entry_buy_val = [], []
        entry_sell_idx, entry_sell_val = [], []

        trend_buy_idx, trend_buy_val = [], []
        trend_sell_idx, trend_sell_val = [], []

        signals = np.zeros(n)

        final_scores, smooth_scores = self.calculate_super_score()

        # 1. CHAMA O TEU NOVO MÉTODO (O Arqueólogo)
        levels = self.get_significant_levels(highs, lows, window=30)
        
        # 2. SEPARA AS MURALHAS (Últimos 5 suportes e 5 resistências)
        last_supports = [l['price'] for l in levels if l['type'] == 'S'][-5:]
        last_resistances = [l['price'] for l in levels if l['type'] == 'R'][-5:]

        lookback_momentum = 2  # Quantos candles o sinal do Score "vale"
        
        for i in range(1, n):
            
            # 1. CONTEXTO DE EXTREMO (Ainda usamos as bandas para filtrar ruído)
            #zona_venda = highs[i-1] > bb80_up[i-1] or highs[i-1] > bb20_up[i-1]
            #zona_compra = lows[i-1] < bb80_low[i-1] or lows[i-1] < bb20_low[i-1]
            extreme_buy_zone = bb80_low[i] > bb20_low[i]
            extreme_sell_zone = bb80_up[i] < bb20_up[i]


            # --- LÓGICA DE COMPRA (BUY) ---
            if extreme_buy_zone:
                # Pivot de 3 velas (O Bico)
                is_pivot_buy = lows[i-1] < lows[i-2] and lows[i-1] < lows[i]
                # Confirmação de Força e Score
                has_strength = closes[i] > opens[i-1] and closes[i] > highs[i-1]

                valid_momentum_buy = any(
                    (final_scores[j-1] <= smooth_scores[j-1] and  # Estava abaixo da média
                    final_scores[j] > smooth_scores[j] and       # Cruzou para cima
                    (final_scores[j-1] <= -60 or final_scores[j] <= -60)) # Garante que ocorreu no pânico
                    for j in range(max(1, i - lookback_momentum), i + 1)
                )

                if is_pivot_buy and has_strength and valid_momentum_buy:
                    current_bottom = lows[i-1]
                    relative_position = (current_bottom - bb20_low[i-1]) / (bb20_up[i-1] - bb20_low[i-1])
                    in_extreme_buy = relative_position <= 0.05 # Aceita até 5% acima da banda (margem dinâmica)
                    # Filtro de Memória S/R
                    close_to_support = any(abs(current_bottom - s) / s <= in_extreme_buy for s in last_supports)
                    is_fresh_low = current_bottom < min(last_supports) if last_supports else True

                    if (close_to_support or is_fresh_low):
                        entry_buy_idx.append(i)
                        entry_buy_val.append(current_bottom)
                        signals[i] = 1
                        # Atualiza memória
                        last_supports.append(current_bottom)
                        last_supports = last_supports[-5:]

            # --- LÓGICA DE VENDA (SELL) ---
            if extreme_sell_zone:
                # Pivot de 3 velas (O Bico)
                is_pivot_sell = highs[i-1] > highs[i-2] and highs[i-1] > highs[i]
                # Confirmação de Força e Score
                has_strength = closes[i] < opens[i-1] and closes[i] < closes[i-1]

                valid_momentum_sell = any(
                    (final_scores[j-1] >= smooth_scores[j-1] and  # Estava acima da média
                    final_scores[j] < smooth_scores[j] and       # Cruzou para baixo
                    (final_scores[j-1] >= 60 or final_scores[j] >= 60)) # Garante que ocorreu no extremo
                    for j in range(max(1, i - lookback_momentum), i + 1)
                )

                if is_pivot_sell and has_strength and valid_momentum_sell:
                    current_top = highs[i-1]

                    # 1. Calculas a posição relativa (igual ao Buy)
                    relative_position = (current_top - bb20_low[i-1]) / (bb20_up[i-1] - bb20_low[i-1])

                    # 2. O filtro de extremo muda para o topo
                    # Aceita se o preço estiver nos 5% finais da banda ou acima dela
                    in_extreme_sell = relative_position >= 0.95
                    
                    
                    # Filtro de Memória S/R
                    close_to_resistence = any(abs(current_top - r) / r <= in_extreme_sell for r in last_resistances)
                    is_fresh_high = current_top > max(last_resistances) if last_resistances else True

                    if close_to_resistence or is_fresh_high:
                        entry_sell_idx.append(i)
                        entry_sell_val.append(current_top)
                        signals[i] = -1
                        # Atualiza memória
                        last_resistances.append(current_top)
                        last_resistances = last_resistances[-5:]

            """
            # --- PASSO 1: CONTEXTO (SETAS) ---
            if lows[i] < bb80_low[i]:
                if bb20_low[i-1] < bb20_low[i] and closes[i] > ltf_basis[i]:
                    entry_buy_idx.append(i)
                    entry_buy_val.append(lows[i] * 0.997) # Ligeiramente abaixo para não sobrepor
                    in_extreme_zone_bull = False 

            
            if highs[i] > bb80_up[i]:
                if bb20_up[i-1] > bb20_up[i] and closes[i] < ltf_basis[i]:
                    entry_sell_idx.append(i)
                    entry_sell_val.append(highs[i] * 1.003) # Ligeiramente acima
                    in_extreme_zone_bear = False
            
            """
            """
            if bb20_low[i-1] < bb80_low[i-1] and bb20_low[i] >= bb80_low[i]:
                entry_buy_idx.append(i)
                entry_buy_val.append(lows[i] * 0.997) # Ligeiramente abaixo para não sobrepor
                in_extreme_zone_bull = False 

            
            if bb20_up[i-1] > bb80_up[i-1] and bb20_up[i] <= bb80_up[i]:
                entry_sell_idx.append(i)
                entry_sell_val.append(highs[i] * 1.003) # Ligeiramente acima
                in_extreme_zone_bear = False
            """

            # --- RESET DE SEGURANÇA ---

            #if closes[i] > ltf_basis[i]: in_extreme_zone_bull = False
            #if closes[i] < ltf_basis[i]: in_extreme_zone_bear = False


        return {
            'bbshort_up': bb20_up,
            'bbshort_low': bb20_low,
            'bbshort_mid': ltf_basis,
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
    

    # 1. Função para detetar níveis históricos (SR)
    # ltf_highs/lows são os teus dados de mercado
    def get_significant_levels(self, highs, lows, window=30):
        sr_levels = []
        for i in range(window, len(highs) - window):
            # Se for o máximo num raio de 'window' velas, é uma Resistência
            if highs[i] == max(highs[i-window : i+window]):
                sr_levels.append({'type': 'R', 'price': highs[i], 'idx': i})
            # Se for o mínimo, é um Suporte
            if lows[i] == min(lows[i-window : i+window]):
                sr_levels.append({'type': 'S', 'price': lows[i], 'idx': i})
        return sr_levels
        