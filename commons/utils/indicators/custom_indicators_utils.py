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

        is_trigger_buy = False # Esta variável controla a repetição
        is_trigger_sell = False

        bbw = self.bbw(bb_short_period, bb_short_std_dev) * 1000
        plano_zero = np.zeros(len(closes))
        gap_index = self.calculate_gap(bbw, plano_zero)

        final_scores, smooth_scores = self.calculate_super_score()
        for i in range(1, n):

            # 1. CONDIÇÃO DE ROMPIMENTO (BREAKOUT)
            # Usamos as bandas azuis (20 períodos) para o gatilho curto
            breakout_up = closes[i] > bb20_up[i]
            breakout_down = closes[i] < bb20_low[i]


            # A regra: Tem de haver um aumento de força de pelo menos 10%
            is_strong_impulse = (gap_index[i] - gap_index[i-1]) >= 15


            # 2. CÁLCULO DE INCLINAÇÃO (SLOPE)
            # Usamos uma janela de 2 para ser mais responsivo, mas filtramos a força
           
            slope_bbw = (bbw[i] - bbw[i-1]) / (bbw[i-1] if bbw[i-1] != 0 else 1)
            #volatility_confirm = bbw[i] > bbw[i-1] and slope_bbw > 0.30
            volatility_confirm = bbw[i] > bbw[i-1] and gap_index[i] > gap_index[i-1] and is_strong_impulse

            #print("AQUII", i, gap_index[i], gap_index[i] - gap_index[i-1])
            """

            # Definir o período de observação
            lookback = 3 

            # Calcular o Slope médio das últimas X velas
            # (bbw[i] - bbw[i-lookback]) dá-nos a tendência real da abertura das bandas
            avg_slope_bbw = (bbw[i] - bbw[i-lookback]) / (bbw[i-lookback] if bbw[i-lookback] != 0 else 1)

            # Confirmamos se a volatilidade está a crescer de forma sustentada
            volatility_confirm = avg_slope_bbw > 0.5
             """
        
            # 4. LÓGICA DE MOMENTUM REFINADA
            is_bullish_momentum = (
                breakout_up and 
                volatility_confirm and
                not is_trigger_buy
            )

            # (Repetir lógica similar para Bearish...)
            is_bearish_momentum = (
                breakout_down and
                volatility_confirm and
                not is_trigger_sell
            )

            # REVERSÃO PARA ALTA (BULL)
            # Se o oscilador bull começar a subir e cruzar um threshold (ex: 20 ou 50)
            """
            if is_bullish_momentum:
                trend_buy_idx.append(i)
                trend_buy_val.append(lows[i])
                signals[i] = 1
                is_trigger_buy = True

            # REVERSÃO PARA BAIXA (BEAR)
            if is_bearish_momentum:
                trend_sell_idx.append(i)
                trend_sell_val.append(highs[i])
                signals[i] = -1
                is_trigger_sell = True

            if bbw[i] < bbw[i-1]:
                is_trigger_buy = False
                is_trigger_sell = False
            """
            # Definimos os 3 candles da sequência:
            # i-2 (Passado próximo)
            # i-1 (O bico do cone)
            # i   (O candle atual de confirmação)

            # 1. CONTEXTO DE EXTREMO (Ainda usamos as bandas para filtrar ruído)
            zona_venda = highs[i-1] > bb80_up[i-1] or highs[i-1] > bb20_up[i-1]
            zona_compra = lows[i-1] < bb80_low[i-1] or lows[i-1] < bb20_low[i-1]

            # 2. PIVOT DE SELL (TOPO)
            # -2 sobe: highs[i-2] > highs[i-3] (opcional, mas confirma força)
            # -1 sobe acima do -2: highs[i-1] > highs[i-2]
            #  0 cai abaixo do -1: closes[i] < lows[i-1]  <-- A quebra da estrutura
            if zona_venda:
                if highs[i-1] > highs[i-2]: # O bico subiu
                    if closes[i] < lows[i-1]: # O atual engoliu a mínima do bico
                        entry_sell_idx.append(i)
                        entry_sell_val.append(highs[i-1] * 1.001)
                        # Opcional: signals[i] = -1

            # 3. PIVOT DE BUY (FUNDO)
            # -1 desce abaixo do -2: lows[i-1] < lows[i-2]
            #  0 sobe acima do -1: closes[i] > highs[i-1] <-- A quebra da estrutura
            if zona_compra:
                if lows[i-1] < lows[i-2]: # O bico desceu
                    if closes[i] > highs[i-1]: # O atual rompeu a máxima do bico
                        entry_buy_idx.append(i)
                        entry_buy_val.append(lows[i-1] * 0.999)
                        # Opcional: signals[i] = 1
            """
            # 1. ZONA DE CONE (Obrigatório estar fora ou tocar na BB80)
            in_extreme_zone_bear = highs[i-2] > bb80_up[i-2] or highs[i-1] > bb80_up[i-1]
            in_extreme_zone_bull = lows[i-2] < bb80_low[i-2] or lows[i-1] < bb80_low[i-1]

            # 2. LÓGICA DE VENDA (SELL) - Quebra de Estrutura no Topo
            if in_extreme_zone_bear:
                # Identificamos o Pivot High (O bico do cone) em i-1 ou i-2
                # E verificamos se o candle atual (i) quebrou a mínima do candle que fez o topo
                topo_do_cone = max(highs[i-1], highs[i-2])
                gatilho_venda = min(lows[i-1], lows[i-2]) # A mínima do bico
                
                # Quebra de Estrutura: Preço atual fecha abaixo da mínima do bico
                if closes[i] < gatilho_venda and closes[i] < ltf_basis[i]:
                    # Confirmamos que a banda 20 já começou a apontar para baixo
                    if bb20_up[i] < bb20_up[i-1]:
                        entry_sell_idx.append(i)
                        entry_sell_val.append(highs[i-1]) # Sinal visual no topo
                        in_extreme_zone_bear = False

            # 3. LÓGICA DE COMPRA (BUY) - Quebra de Estrutura no Fundo
            if in_extreme_zone_bull:
                # Identificamos o Pivot Low em i-1 ou i-2
                fundo_do_cone = min(lows[i-1], lows[i-2])
                gatilho_compra = max(highs[i-1], highs[i-2]) # A máxima do bico
                
                # Quebra de Estrutura: Preço atual fecha acima da máxima do bico
                if closes[i] > gatilho_compra and closes[i] > ltf_basis[i]:
                    # Confirmamos que a banda 20 já curvou para cima
                    if bb20_low[i] > bb20_low[i-1]:
                        entry_buy_idx.append(i)
                        entry_buy_val.append(lows[i-1])
                        in_extreme_zone_bull = False
            """
            # --- PASSO 1: CONTEXTO (SETAS) ---
            """
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
        