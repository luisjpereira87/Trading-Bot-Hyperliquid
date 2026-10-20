import math
from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from commons.enums.signal_enum import Signal
from commons.utils.indicators.base_indicators_utils import BaseIndicatorsUtils
from commons.utils.indicators.tv_indicators_utils import TvIndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper


class CustomIndicatorsUtils(BaseIndicatorsUtils):
    def __init__(self, ohlcv: OhlcvWrapper, mode='ta'):
        super().__init__(ohlcv, mode)

        self.tvIndicators = TvIndicatorsUtils(ohlcv)


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
        bb20_up = bb20_up_raw
        bb20_low = bb20_low_raw

        bb80_up = self.smooth_band(bb80_up_raw, 20)
        bb80_low = self.smooth_band(bb80_low_raw, 20)

        # Listas para Sinais de Entrada Real (Círculos)
        entry_buy_idx, entry_buy_val = [], []
        entry_sell_idx, entry_sell_val = [], []

        signals = np.zeros(n)
        rsi, rsi_ema = self.rsi()
        adx = self.adx()
        #self.set_ohlcv(ohlcv_higher)
        #rsi_higher, rsi_ema_higher = self.rsi(56)
        #super_score, super_score_ema = self.calculate_super_score()
        for i in range(1, n):
            extreme_buy_setup = bb80_low[i] > bb20_low[i] and lows[i] < bb80_low[i]
            extreme_sell_setup = bb80_up[i] < bb20_up[i] and highs[i] > bb80_up[i]

            extreme_oversold = any(rsi[j] < 30 for j in range(i - 3, i + 1))
            extreme_overbought = any(rsi[j] > 70 for j in range(i - 3, i + 1))

            if extreme_buy_setup and extreme_oversold:
                if bb20_low[i] > bb20_low[i - 1] and closes[i] > ltf_basis[i]:
                    signals[i] = 1
                    entry_buy_idx.append(i)
                    entry_buy_val.append(lows[i])

            elif extreme_sell_setup  and extreme_overbought:
                if bb20_up[i] < bb20_up[i - 1] and closes[i] < ltf_basis[i]:
                    signals[i] = -1
                    entry_sell_idx.append(i)
                    entry_sell_val.append(highs[i])

            """
            # --- SINAL EXTRA: CRUZAMENTO DE MOMENTUM ---
            if rsi[i] < rsi_ema[i] and rsi[i-1] >= rsi_ema[i-1] and rsi[i] < 60:
                if closes[i] < bb80_basis_raw[i] and abs(rsi[i] - rsi_ema[i]) > 5:  # Confirmação pelo preço/Kalman
                    signals[i] = -1
                    entry_sell_idx.append(i)
                    entry_sell_val.append(highs[i])
                    #print(f"Sinal Extra Venda no Idx {i}")

            # Compra Extra (Fundo Confirmado)
            if rsi[i] > rsi_ema[i] and rsi[i-1] <= rsi_ema[i-1] and rsi[i] > 60:
                if closes[i] > bb80_basis_raw[i] and abs(rsi[i] - rsi_ema[i]) > 5:
                    signals[i] = 1
                    entry_buy_idx.append(i)
                    entry_buy_val.append(lows[i])
                    #print(f"Sinal Extra Compra no Idx {i}")
            """
        return {
            'bbshort_up': bb20_up,
            'bbshort_low': bb20_low,
            'bbshort_mid': ltf_basis,
            'bblong_up': bb80_up,
            'bblong_low': bb80_low,
            'bblong_mid': bb80_basis_raw,
            'signals': signals,
            'entry_buy_idx': entry_buy_idx,
            'entry_buy_val': entry_buy_val,
            'entry_sell_idx': entry_sell_idx,
            'entry_sell_val': entry_sell_val
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

    def keltner_channels(self, period=20, multiplier=3.0):
        """
        Calcula os Keltner Channels baseados em ATR.
        """

        close, high, low = pd.Series(self.closes), pd.Series(self.highs), pd.Series(self.lows)
        # 1. Calcular a EMA da base (Linha Central)
        basis = close.ewm(span=period, adjust=False).mean()

        # 2. Calcular o True Range (TR) para o ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 3. Calcular o ATR (Média do TR)
        atr = tr.ewm(span=period, adjust=False).mean()

        # 4. Calcular as Bandas
        keltner_up = basis + (multiplier * atr)
        keltner_down = basis - (multiplier * atr)

        return keltner_up, basis, keltner_down

    def analyze_and_execute_candles(self, trend_threshold=0.3) -> Signal:
        """
        Equivalente em Python do método TypeScript analyzeAndExecuteCoin.
        Recebe uma lista de candles, calcula RSI, EMA do RSI, variação de tendência
        e valida a força do candle corrente antes de disparar o sinal.
        """

        # Cálculo do RSI atual (14 períodos) e respetiva EMA
        rsi, ema_rsi = self.rsi(signal_period=9)
        current_rsi = rsi[-1]
        current_ema_rsi = ema_rsi[-1]

        is_rsi_bullish = current_rsi > current_ema_rsi
        is_rsi_bearish = current_rsi < current_ema_rsi

        # Variação de tendência baseada nos últimos blocos (garantindo compatibilidade com lista de floats ou dicts)
        recent = self.closes[-5:]
        older = self.closes[-10:-5]

        # Helper seguro para extrair o valor do fecho (seja float direto ou dicionário)
        def get_close(val):
            return float(val['close']) if isinstance(val, dict) else float(val)

        recent_avg = sum(get_close(c) for c in recent) / len(recent)
        older_avg = sum(get_close(c) for c in older) / len(older)

        change_percent = ((recent_avg - older_avg) / older_avg) * 100
        threshold = trend_threshold

        trend = 'neutral'
        if change_percent > threshold:
            trend = 'up'
        elif change_percent < -threshold:
            trend = 'down'

        """
        print(
            f"KLINE: 📊 Variação: {change_percent:.4f}% | RSI: {current_rsi:.1f} (EMA: {current_ema_rsi:.1f}) | Tendência: {trend.upper()}"
        )
        """
        if trend == 'neutral':
            return Signal.HOLD

        # 🛡️ FILTRO DE RSI: Validar se o momento técnico apoia a tendência
        if trend == 'up' and not is_rsi_bullish:
            """
            print(
                f"WARN: ⏳ Sinal UP ignorado: Variação positiva mas RSI ({current_rsi:.1f}) abaixo da EMA ({current_ema_rsi:.1f}) - Falta de momentum comprador."
            )
            """
            return Signal.HOLD
        if trend == 'down' and not is_rsi_bearish:
            """
            print(
                f"WARN: ⏳ Sinal DOWN ignorado: Variação negativa mas RSI ({current_rsi:.1f}) acima da EMA ({current_ema_rsi:.1f}) - Falta de momentum vendedor."
            )
            """
            return Signal.HOLD

        # 🕯️ FILTRO DE ANÁLISE DO CORPO E DIREÇÃO DO CANDLE CORRENTE
        candle_open = float(self.opens[-1]) if isinstance(self.opens[-1], (int, float, str)) else float(
            self.opens[-1].get('open', 0))
        candle_close = get_close(self.closes[-1])
        candle_high = float(self.highs[-1]) if isinstance(self.highs[-1], (int, float, str)) else float(
            self.highs[-1].get('high', 0))
        candle_low = float(self.lows[-1]) if isinstance(self.lows[-1], (int, float, str)) else float(
            self.lows[-1].get('low', 0))

        body_size = abs(candle_close - candle_open)
        total_range = candle_high - candle_low

        # 1. Evitar velas de indecisão (Dojis / corpos muito esmagados face ao pavio)
        min_body_ratio = 0.25
        if total_range > 0 and (body_size / total_range) < min_body_ratio:
            #print("WARN: ⏳ Ignorado: Candle atual sem corpo expressivo (Doji/Indecisão). A aguardar força real.")
            return Signal.HOLD

        # 2. Validar se o corpo do candle está na direção correta do sinal
        if trend == 'up' and candle_close <= candle_open:
            #print(f"WARN: ⏳ Sinal UP ignorado: Candle atual está vermelho (Open: ${candle_open} > Close: ${candle_close}).")
            return Signal.HOLD
        if trend == 'down' and candle_close >= candle_open:
            #print(f"WARN: ⏳ Sinal DOWN ignorado: Candle atual está verde (Open: ${candle_open} < Close: ${candle_close}).")
            return Signal.HOLD

        #print(f"SIGNAL: ✅ Todos os filtros técnicos validados com sucesso para tendência {trend.upper()}!")

        if trend == 'up':
            return Signal.BUY
        elif trend == 'down':
            return Signal.SELL

        return Signal.HOLD

        # Ponto de disparo para a ordem na Hyperliquid...

    def is_healthy_acceleration(self, variations, trend='up', min_candles=5, max_allowed_spike=1.5):
        if len(variations) < min_candles:
            return False

        recent_vars = variations[-min_candles:]

        # 1. Verificar picos de exaustão (usando o valor absoluto, pois tanto serve para cima como para baixo)
        if abs(recent_vars[-1]) >= max_allowed_spike:
            return False

        mid = len(recent_vars) // 2
        first_half_avg = sum(recent_vars[:mid]) / mid
        second_half_avg = sum(recent_vars[mid:]) / (len(recent_vars) - mid)

        if trend == 'up':
            # Tendência de alta: variações positivas, a segunda metade deve ser maior que a primeira
            is_accelerating = second_half_avg > first_half_avg
            is_valid_end = recent_vars[-1] > 0
            return is_accelerating and is_valid_end

        elif trend == 'down':
            # Tendência de baixa: variações negativas.
            # Aceleração significa a média ficar *mais negativa* (menor numericamente na segunda metade)
            is_accelerating = second_half_avg < first_half_avg
            is_valid_end = recent_vars[-1] < 0
            return is_accelerating and is_valid_end

        return False

    def analyze_candles_history__(
            self,
            trend_threshold: float = 0.3
    ) -> list[Signal] | tuple[list[Signal], list[Any]]:  # Retorna uma lista de Signal (ex: Signal.BUY, Signal.SELL, Signal.HOLD)
        """
        Varre o histórico completo de candles de forma iterativa, aplicando
        os mesmos filtros de tendência, RSI, EMA e proporção de corpo de vela,
        e devolve uma lista de sinais correspondente a cada período.
        """
        signals = []
        change_percent_list = []

        # Precisamos de pelo menos 20+10 candles para começar a ter histórico válido para RSI e médias móveis
        min_required_len = 30
        total_candles = len(self.closes)

        if total_candles < min_required_len:
            return [Signal.HOLD] * total_candles

        # Pré-calculamos o RSI e a EMA do RSI para toda a série histórica de uma só vez (otimização)
        # (Assumindo que os teus métodos internos conseguem calcular com base numa fatia ou no total)
        rsi_full, ema_rsi_full = self.rsi(signal_period=9)

        for i in range(min_required_len, total_candles):
            # Fatiamos os dados até ao índice atual `i` para evitar "look-ahead bias" (viés de olhar o futuro)
            slice_closes = self.closes[:i + 1]

            current_rsi = rsi_full[i]
            current_ema_rsi = ema_rsi_full[i]

            is_rsi_bullish = current_rsi > current_ema_rsi and current_rsi > 50
            is_rsi_bearish = current_rsi < current_ema_rsi and current_rsi < 50

            # Variação de tendência baseada nos blocos anteriores até ao momento `i`
            recent = slice_closes[-5:]
            older = slice_closes[-10:-5]

            def get_val(val, key='close'):
                if isinstance(val, dict):
                    return float(val.get(key, 0))
                return float(val)

            recent_avg = sum(get_val(c) for c in recent) / len(recent)
            older_avg = sum(get_val(c) for c in older) / len(older)

            change_percent = ((recent_avg - older_avg) / older_avg) * 100
            change_percent_list.append(change_percent)

            trend = 'neutral'
            if change_percent > trend_threshold:
                trend = 'up'
            elif change_percent < -trend_threshold:
                trend = 'down'

            if trend == 'neutral':
                signals.append(Signal.HOLD)
                continue

            # 🛡️ FILTRO DE RSI
            if trend == 'up' and not is_rsi_bullish:
                signals.append(Signal.HOLD)
                continue
            if trend == 'down' and not is_rsi_bearish:
                signals.append(Signal.HOLD)
                continue

            # 🕯️ FILTRO DE ANÁLISE DO CORPO E DIREÇÃO DO CANDLE ATUAL (no índice `i`)
            c_open = get_val(self.opens[i], 'open') if isinstance(self.opens[i], (dict, list)) else float(self.opens[i])
            c_close = get_val(self.closes[i], 'close')
            c_high = get_val(self.highs[i], 'high') if isinstance(self.highs[i], (dict, list)) else float(self.highs[i])
            c_low = get_val(self.lows[i], 'low') if isinstance(self.lows[i], (dict, list)) else float(self.lows[i])

            body_size = abs(c_close - c_open)
            total_range = c_high - c_low

            # 1. Evitar Dojis / velas de indecisão
            min_body_ratio = 0.25
            if total_range > 0 and (body_size / total_range) < min_body_ratio:
                signals.append(Signal.HOLD)
                continue

            # 2. Direção do candle
            if trend == 'up' and c_close <= c_open:
                signals.append(Signal.HOLD)
                continue
            if trend == 'down' and c_close >= c_open:
                signals.append(Signal.HOLD)
                continue

            # 3. 🚨 NOVO FILTRO: Evitar Candle com Corpo Demasiado Esticado (Comparado com o passado recente)
            # Em vez de olhar para o range total (que inclui pavios), olhamos para o CORPO da vela (`body_size`).
            # Se o corpo atual for desproporcionadamente maior que a média dos corpos anteriores, é um estirão de exaustão.
            """
            lookback_bodies = 5
            if i >= lookback_bodies:
                recent_bodies = [
                    abs(get_val(self.closes[j], 'close') - get_val(self.opens[j], 'open'))
                    for j in range(i - lookback_bodies, i)
                ]
                avg_body_size = sum(recent_bodies) / len(recent_bodies)

                # Se o corpo da vela atual for mais do que 2.0x ou 2.5x o tamanho médio dos corpos recentes, rejeita!
                max_body_multiplier = 2.0
                if avg_body_size > 0 and body_size > (avg_body_size * max_body_multiplier):
                    signals.append(Signal.HOLD)
                    continue
            """
            # Se passou por todos os filtros neste exato candle:
            if trend == 'up':
                signals.append(Signal.BUY)
            else:
                signals.append(Signal.SELL)

        # Preenche os primeiros candles iniciais (onde não havia histórico suficiente) com HOLD
        padding = [Signal.HOLD] * (total_candles - len(signals))
        return padding + signals, change_percent_list

    def analyze_candles_history_old(
            self
    ) -> tuple[list[Signal], list[float], list[float]]:
        """
        Varre o histórico de candles aplicando RSI (ignição), quebra de
        estrutura (BOS), filtro de Doji e cálculo dinâmico de SL e TP.
        """
        signals = []
        stop_losses = []
        take_profits = []

        min_required_len = 30
        total_candles = len(self.closes)

        if total_candles < min_required_len:
            hold_pad = [Signal.HOLD] * total_candles
            return hold_pad, [0.0] * total_candles, [0.0] * total_candles

        rsi_full, ema_rsi_full = self.rsi(signal_period=9)

        # Obter estrutura de mercado
        last_HL, last_LH = self.tvIndicators.market_structure(2, 2)

        # 🛡️ SALVAGUARDA: Garantir que propagamos os valores se houver NaNs (Forward Fill)
        # (Assume que last_LH e last_HL podem ser listas ou pandas Series)
        def clean_structural_series(series):
            cleaned = []
            last_valid = 0.0
            for val in series:
                if val is not None and not (isinstance(val, float) and str(val).lower() == 'nan'):
                    last_valid = float(val)
                cleaned.append(last_valid)
            return cleaned

        safe_last_LH = clean_structural_series(last_LH)
        safe_last_HL = clean_structural_series(last_HL)  # Ou last_HL consoante o teu indicador

        for i in range(min_required_len, total_candles):
            slice_closes = self.closes[:i + 1]

            current_rsi = rsi_full[i]
            current_ema_rsi = ema_rsi_full[i]

            # Níveis estruturais atuais e anteriores com valores limpos
            lh_curr = safe_last_LH[i]
            lh_prev = safe_last_LH[i - 1]

            hl_curr = safe_last_HL[i]
            hl_prev = safe_last_HL[i - 1]

            # 🛡️ FILTRO DE IGNIÇÃO DE RSI
            is_rsi_ignition_bullish = current_rsi > current_ema_rsi
            is_rsi_ignition_bearish = current_rsi < current_ema_rsi

            def get_val(val, key='close'):
                if isinstance(val, dict):
                    return float(val.get(key, 0))
                return float(val)

            c_open = get_val(self.opens[i], 'open') if isinstance(self.opens[i], (dict, list)) else float(self.opens[i])
            c_close = get_val(self.closes[i], 'close')
            c_high = get_val(self.highs[i], 'high') if isinstance(self.highs[i], (dict, list)) else float(self.highs[i])
            c_low = get_val(self.lows[i], 'low') if isinstance(self.lows[i], (dict, list)) else float(self.lows[i])

            prev_close = get_val(self.closes[i - 1], 'close')

            body_size = abs(c_close - c_open)
            total_range = c_high - c_low

            # 1. Evitar Dojis / velas de indecisão
            min_body_ratio = 0.25
            if total_range > 0 and (body_size / total_range) < min_body_ratio:
                signals.append(Signal.HOLD)
                stop_losses.append(0.0)
                take_profits.append(0.0)
                continue

            # Determinar a direção pelo corpo do candle atual
            if c_close > c_open:
                trend = 'up'
            elif c_close < c_open:
                trend = 'down'
            else:
                signals.append(Signal.HOLD)
                stop_losses.append(0.0)
                take_profits.append(0.0)
                continue

            # 2. FILTRO DE RSI
            if trend == 'up' and not is_rsi_ignition_bullish:
                signals.append(Signal.HOLD)
                stop_losses.append(0.0)
                take_profits.append(0.0)
                continue
            if trend == 'down' and not is_rsi_ignition_bearish:
                signals.append(Signal.HOLD)
                stop_losses.append(0.0)
                take_profits.append(0.0)
                continue

            # 🚀 3. QUEBRA DE ESTRUTURA REAL (BOS)
            if trend == 'up':
                # O preço estava abaixo ou na resistência (Lower High) e rompeu estritamente para cima
                if not (prev_close <= lh_prev and c_close > lh_curr):
                    signals.append(Signal.HOLD)
                    stop_losses.append(0.0)
                    take_profits.append(0.0)
                    continue
            elif trend == 'down':
                # O preço estava acima ou no suporte (Higher Low) e rompeu estritamente para baixo
                if not (prev_close >= hl_prev and c_close < hl_curr):
                    signals.append(Signal.HOLD)
                    stop_losses.append(0.0)
                    take_profits.append(0.0)
                    continue

            # 🎯 4. CÁLCULO DINÂMICO DE SL E TP
            risk_reward_ratio = 2.0

            if trend == 'up':
                sl = c_low * 0.999
                risk = c_close - sl
                tp = c_close + (risk * risk_reward_ratio)

                signals.append(Signal.BUY)
                stop_losses.append(sl)
                take_profits.append(tp)
            else:
                sl = c_high * 1.001
                risk = sl - c_close
                tp = c_close - (risk * risk_reward_ratio)

                signals.append(Signal.SELL)
                stop_losses.append(sl)
                take_profits.append(tp)

        padding_len = total_candles - len(signals)
        return (
            [Signal.HOLD] * padding_len + signals,
            [0.0] * padding_len + stop_losses,
            [0.0] * padding_len + take_profits
        )

    def analyze_candles_history(
            self,
            breakout_lookback: int = 5,  # Número de velas anteriores a verificar para a quebra de extremos
            min_trailing_activation_pct: float = 0.008
    ) -> tuple[list[Signal], list[float], list[float]]:
        """
        Varre o histórico filtrando ruído: exige quebra de máximos/mínimos de um
        período maior (ex: 5 velas), ignição sólida do RSI (eixo 50) e filtro de Doji.
        """
        signals = []
        stop_losses = []
        take_profits = []

        min_required_len = max(30, breakout_lookback + 5)
        total_candles = len(self.closes)

        if total_candles < min_required_len:
            hold_pad = [Signal.HOLD] * total_candles
            return hold_pad, [0.0] * total_candles, [0.0] * total_candles

        rsi_full, ema_rsi_full = self.rsi(signal_period=9)

        for i in range(min_required_len, total_candles):
            def get_val(val, key='close'):
                if isinstance(val, dict):
                    return float(val.get(key, 0))
                return float(val)

            c_open = get_val(self.opens[i], 'open') if isinstance(self.opens[i], (dict, list)) else float(self.opens[i])
            c_close = get_val(self.closes[i], 'close')
            c_high = get_val(self.highs[i], 'high') if isinstance(self.highs[i], (dict, list)) else float(self.highs[i])
            c_low = get_val(self.lows[i], 'low') if isinstance(self.lows[i], (dict, list)) else float(self.lows[i])

            # 🛡️ FILTRO DE IGNIÇÃO DE RSI RIGOROSO (Eixo 50)
            current_rsi = rsi_full[i]
            current_ema_rsi = ema_rsi_full[i]

            is_rsi_ignition_bullish = (current_rsi > 50) and (current_ema_rsi < 50 or current_rsi > current_ema_rsi)
            is_rsi_ignition_bearish = (current_rsi < 50) and (current_ema_rsi > 50 or current_rsi < current_ema_rsi)

            body_size = abs(c_close - c_open)
            total_range = c_high - c_low

            # 1. Evitar Dojis / velas de indecisão
            min_body_ratio = 0.25
            if total_range > 0 and (body_size / total_range) < min_body_ratio:
                signals.append(Signal.HOLD)
                stop_losses.append(0.0)
                take_profits.append(0.0)
                continue

            # Determinar a direção pelo corpo do candle atual
            if c_close > c_open:
                trend = 'up'
            elif c_close < c_open:
                trend = 'down'
            else:
                signals.append(Signal.HOLD)
                stop_losses.append(0.0)
                take_profits.append(0.0)
                continue

            # 2. FILTRO DE RSI
            if trend == 'up' and not is_rsi_ignition_bullish:
                signals.append(Signal.HOLD)
                stop_losses.append(0.0)
                take_profits.append(0.0)
                continue
            if trend == 'down' and not is_rsi_ignition_bearish:
                signals.append(Signal.HOLD)
                stop_losses.append(0.0)
                take_profits.append(0.0)
                continue

            # 🚀 3. QUEBRA DOS MÁXIMOS/MÍNIMOS DAS ÚLTIMAS 'N' VELAS
            # Recolher os máximos e mínimos do período anterior definido em breakout_lookback
            past_highs = []
            past_lows = []
            for j in range(1, breakout_lookback + 1):
                idx = i - j
                past_highs.append(
                    get_val(self.highs[idx], 'high') if isinstance(self.highs[idx], (dict, list)) else float(
                        self.highs[idx]))
                past_lows.append(get_val(self.lows[idx], 'low') if isinstance(self.lows[idx], (dict, list)) else float(
                    self.lows[idx]))

            max_past_high = max(past_highs)
            min_past_low = min(past_lows)

            if trend == 'up':
                # O fecho atual tem de ultrapassar o MAIOR máximo das últimas N velas
                if c_close <= max_past_high:
                    signals.append(Signal.HOLD)
                    stop_losses.append(0.0)
                    take_profits.append(0.0)
                    continue
            elif trend == 'down':
                # O fecho atual tem de furar o MENOR mínimo das últimas N velas
                if c_close >= min_past_low:
                    signals.append(Signal.HOLD)
                    stop_losses.append(0.0)
                    take_profits.append(0.0)
                    continue

            # 🎯 4. CÁLCULO DINÂMICO DE SL E TP (Com salvaguarda do trailing)
            risk_reward_ratio = 2.0

            if trend == 'up':
                sl = c_low * 0.999
                risk = c_close - sl
                tp_calculated = c_close + (risk * risk_reward_ratio)

                min_tp_allowed = c_close * (1.0 + min_trailing_activation_pct)
                tp = max(tp_calculated, min_tp_allowed)

                signals.append(Signal.BUY)
                stop_losses.append(sl)
                take_profits.append(tp)
            else:
                sl = c_high * 1.001
                risk = sl - c_close
                tp_calculated = c_close - (risk * risk_reward_ratio)

                min_tp_allowed = c_close * (1.0 - min_trailing_activation_pct)
                tp = min(tp_calculated, min_tp_allowed)

                signals.append(Signal.SELL)
                stop_losses.append(sl)
                take_profits.append(tp)

        padding_len = total_candles - len(signals)
        return (
            [Signal.HOLD] * padding_len + signals,
            [0.0] * padding_len + stop_losses,
            [0.0] * padding_len + take_profits
        )