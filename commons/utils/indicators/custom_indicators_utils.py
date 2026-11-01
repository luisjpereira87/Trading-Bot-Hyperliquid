import math
from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from commons.enums.candle_type_enum import CandleType
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
        is_strong_signals = np.zeros(n, dtype=bool)
        is_enter = False

        rsi, rsi_ema = self.rsi()
        for i in range(1, n):

            extreme_buy_setup = bb80_low[i] > bb20_low[i]
            extreme_sell_setup = bb80_up[i] < bb20_up[i]

            # Métricas da vela atual
            body = abs(closes[i] - opens[i])
            total_range = highs[i] - lows[i]

            if total_range == 0:
                continue

            # Evitar divisão por zero ou corpos estéreis (Doji perfeito)
            # Se o corpo for quase zero, exigimos um pequeno mínimo ou tratamos a proporção
            effective_body = max(body, 0.00001)  # Proteção contra divisão por zero se precisares

            lower_shadow = min(opens[i], closes[i]) - lows[i]
            upper_shadow = highs[i] - max(opens[i], closes[i])

            # Critério Agressivo: A sombra tem de ser pelo menos 3x superior ao corpo
            is_strong_lower_wick = lower_shadow >= (body * 2.0)
            is_strong_upper_wick = upper_shadow >= (body * 2.0)

            if extreme_buy_setup:
                # Via 1: Fura a zona extrema (Rutura Direta)
                if bb20_low[i] > bb20_low[i - 1] and closes[i] > ltf_basis[i] and lows[i] < bb80_low[i]:
                    signals[i] = 1
                    entry_buy_idx.append(i)
                    entry_buy_val.append(lows[i])
                    is_enter = True


                # Via 2: NÃO fura a zona (fica de fora), mas deixa um pavio forte de rejeição perto da banda
                if not is_enter and closes[i] > closes[i - 1] and is_strong_lower_wick and rsi[i] <= 30:
                    signals[i] = 1
                    entry_buy_idx.append(i)
                    entry_buy_val.append(lows[i])
                    is_strong_signals[i] = True

            elif extreme_sell_setup:
                # Via 1: Fura a zona extrema para cima (Rutura Direta)
                if bb20_up[i] < bb20_up[i - 1] and closes[i] < ltf_basis[i] and highs[i] > bb80_up[i]:
                    signals[i] = -1
                    entry_sell_idx.append(i)
                    entry_sell_val.append(highs[i])
                    is_enter = True


                # Via 2: NÃO fura a zona, mas deixa um pavio superior forte de rejeição
                if not is_enter and closes[i] < closes[i - 1] and is_strong_upper_wick and rsi[i] >= 70:
                    signals[i] = -1
                    entry_sell_idx.append(i)
                    entry_sell_val.append(highs[i])
                    is_strong_signals[i] = True

            else:
                is_enter = False


        return {
            'bbshort_up': bb20_up,
            'bbshort_low': bb20_low,
            'bbshort_mid': ltf_basis,
            'bblong_up': bb80_up,
            'bblong_low': bb80_low,
            'bblong_mid': bb80_basis_raw,
            'signals': signals,
            'is_strong_signals': is_strong_signals,
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

    def analyze_candles_history_teste(
            self,
            breakout_lookback: int = 5,
            min_trailing_activation_pct: float = 0.008
    ) -> tuple[list[Signal], list[float], list[float]]:
        """
        Varre o histórico filtrando ruído: exige cruzamento do Andean Oscillator,
        validação opcional de RSI/Doji e mantém o alinhamento correto dos arrays.
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
        ema200 = self.ema(200)
        ema50 = self.ema(50)
        andean_oscillator = self.tvIndicators.calculate_andean_oscillator()
        bull = andean_oscillator['bull']
        bear = andean_oscillator['bear']

        for i in range(min_required_len, total_candles):
            def get_val(val, key='close'):
                if isinstance(val, dict):
                    return float(val.get(key, 0))
                return float(val)

            c_open = get_val(self.opens[i], 'open') if isinstance(self.opens[i], (dict, list)) else float(self.opens[i])
            c_close = get_val(self.closes[i], 'close')
            c_high = get_val(self.highs[i], 'high') if isinstance(self.highs[i], (dict, list)) else float(self.highs[i])
            c_low = get_val(self.lows[i], 'low') if isinstance(self.lows[i], (dict, list)) else float(self.lows[i])

            # 🛡️ FILTROS OPCIONAIS DE QUALIDADE
            current_rsi = rsi_full[i]
            current_ema_rsi = ema_rsi_full[i]
            is_rsi_ignition_bullish = (current_rsi > 50) and (current_ema_rsi < 50 or current_rsi > current_ema_rsi)
            is_rsi_ignition_bearish = (current_rsi < 50) and (current_ema_rsi > 50 or current_rsi < current_ema_rsi)

            body_size = abs(c_close - c_open)
            total_range = c_high - c_low
            is_doji = total_range > 0 and (body_size / total_range) < 0.05  # Evitar velas de indecisão

            # 1. Identificar tendência pelo Andean Oscillator (Cruzamento)
            trend = 'hold'
            if bull[i] >= bear[i] and bull[i - 1] < bear[i - 1]:
                trend = 'up'
            elif bull[i] <= bear[i] and bull[i - 1] > bear[i - 1]:
                trend = 'down'

            risk_reward_ratio = 2.0

            # Valores por defeito para velas sem sinal (HOLD)
            current_signal = Signal.HOLD
            sl = 0.0
            tp = 0.0

            # 🎯 4. CÁLCULO DINÂMICO DE SL E TP
            if trend == 'up' and not is_doji:
                sl = c_low * 0.999
                risk = c_close - sl
                tp_calculated = c_close + (risk * risk_reward_ratio)

                min_tp_allowed = c_close * (1.0 + min_trailing_activation_pct)
                tp = max(tp_calculated, min_tp_allowed)

                current_signal = Signal.BUY

            elif trend == 'down' and not is_doji:
                sl = c_high * 1.001
                risk = sl - c_close
                tp_calculated = c_close - (risk * risk_reward_ratio)

                # CORRIGIDO: Adicionado o operador de subtração (-) em falta
                min_tp_allowed = c_close * (1.0 - min_trailing_activation_pct)
                tp = min(tp_calculated, min_tp_allowed)

                current_signal = Signal.SELL

            # Adiciona sempre o resultado de cada vela para manter os tamanhos sincronizados
            signals.append(current_signal)
            stop_losses.append(sl)
            take_profits.append(tp)

        padding_len = total_candles - len(signals)
        return (
            [Signal.HOLD] * padding_len + signals,
            [0.0] * padding_len + stop_losses,
            [0.0] * padding_len + take_profits
        )

    def analyze_candles_history__(
            self,
            mode: str = "pullback",  # "breakout" ou "pullback"
            ema_fast: int = 50,
            ema_slow: int = 200,
            ema_pullback: int = 20,
            donchian: int = 20,
            atr_period: int = 14,
            stop_atr: float = 2.0,  # SL = fecho - stop_atr * ATR
            tp_r: float = 3.0,  # TP = fecho + tp_r * risco (ver nota sobre trailing)
            cost_roundtrip_pct: float = 0.003,  # 2 x (comissão 0,1% + slippage 0,05%)
            min_risk_over_cost: float = 3.0,  # risco% >= 3x custo ida e volta
            cooldown_bars: int = 5  # evita sinais em velas consecutivas
    ):
        """
        Long-only, pensado para velas de 1h/4h (alimenta o bot com esse timeframe).
        Sinal na vela i usa SÓ dados até i (vela fechada). A entrada realista é na
        abertura da vela i+1: o teu executor deve fazê-lo.
        """
        opens = [float(x) for x in self.opens]
        highs = [float(x) for x in self.highs]
        lows = [float(x) for x in self.lows]
        closes = [float(x) for x in self.closes]
        n = len(closes)

        signals = [Signal.HOLD] * n
        stops = [0.0] * n
        tps = [0.0] * n

        warmup = max(ema_slow, donchian, atr_period) + 5
        if n <= warmup:
            return signals, stops, tps

        ema_f = self.ema( ema_fast)
        ema_s = self.ema( ema_slow)
        ema_p = self.ema( ema_pullback)
        atr = self.atr( atr_period)

        last_signal = -10 ** 9
        for i in range(warmup, n):
            if atr[i] is None or atr[i] <= 0:
                continue
            c = closes[i]

            # 1) FILTRO DE TENDÊNCIA (regime)
            if not (c > ema_s[i] and ema_f[i] > ema_s[i]):
                continue

            # 2) GATILHO (um só, sem indicadores redundantes)
            if mode == "breakout":
                prior_high = max(highs[i - donchian:i])  # exclui a vela atual
                if c <= prior_high:
                    continue
            elif mode == "pullback":
                dipped = any(closes[j] < ema_p[j] for j in range(i - 5, i))
                if not (dipped and c > ema_p[i] and c > highs[i - 1]):
                    continue
            else:
                raise ValueError("mode deve ser 'breakout' ou 'pullback'")

            # 3) STOP EM ATR e filtro de custos
            sl = c - stop_atr * atr[i]
            risk = c - sl
            if risk <= 0 or (risk / c) < min_risk_over_cost * cost_roundtrip_pct:
                continue

            # 4) COOLDOWN
            if i - last_signal < cooldown_bars:
                continue
            last_signal = i

            signals[i] = Signal.BUY
            stops[i] = sl
            tps[i] = c + tp_r * risk

        return signals, stops, tps

    """
    Estratégia: reteste de EMA em tendência, velas de 15 min (long e short).

      Regime  : LONG  -> fecho > EMA(800) e EMA(200) > EMA(800)   (~ EMA200/EMA50 de 1h)
                SHORT -> fecho < EMA(800) e EMA(200) < EMA(800)
      Gatilho : recuo (long) / repique (short) à zona EMA20/EMA50 sem a perder no fecho,
                e vela de confirmação:
                  long : fecho > EMA20 e > máximo da vela anterior
                  short: fecho < EMA20 e < mínimo da vela anterior
      Stop    : entrada -/+ stop_atr * ATR(1h)   (ATR de velas de 1h JÁ FECHADAS)
      TP      : entrada +/- tp_r * risco
      Filtros : risco% >= min_risk_over_cost * custo ida-e-volta; cooldown

    Tudo é causal: o sinal na vela i só usa dados até à vela i (prefixo-invariante).
    Assume `Signal` (BUY/SELL/HOLD) já importado no teu código.
      BUY  = abrir long  (SL abaixo, TP acima)
      SELL = abrir short (SL acima,  TP abaixo)
    """

    # ------------------------------------------------------------ auxiliares
    def _ema(self,values, period):
        k = 2.0 / (period + 1)
        out, prev = [], None
        for v in values:
            prev = v if prev is None else prev + k * (v - prev)
            out.append(prev)
        return out

    def _atr_wilder(self,highs, lows, closes, period):
        n = len(closes)
        out = [None] * n
        if n < period + 1:
            return out
        tr = [highs[0] - lows[0]]
        for i in range(1, n):
            tr.append(max(highs[i] - lows[i],
                          abs(highs[i] - closes[i - 1]),
                          abs(lows[i] - closes[i - 1])))
        val = sum(tr[:period]) / period
        out[period - 1] = val
        for i in range(period, n):
            val = (val * (period - 1) + tr[i]) / period
            out[i] = val
        return out

    def _aggregate(self,highs, lows, closes, n):
        """Agrega n velas por posição (bloco k = velas n*k .. n*k+n-1).
        Só devolve blocos completos. Alinha o início dos dados a uma hora cheia
        para o bloco coincidir com as velas reais de 1h."""
        m = len(closes) // n
        h = [max(highs[k * n:(k + 1) * n]) for k in range(m)]
        l = [min(lows[k * n:(k + 1) * n]) for k in range(m)]
        c = [closes[(k + 1) * n - 1] for k in range(m)]
        return h, l, c

    # -------------------------------------------------------------- método
    def analyze_candles_history(
            self,
            enable_long: bool = True,
            enable_short: bool = True,
            ema_regime_fast: int = 200,
            ema_regime_slow: int = 800,
            ema_a: int = 20,
            ema_b: int = 50,
            retest_window: int = 6,  # velas onde o recuo pode ter ocorrido
            tf_factor: int = 4,  # 4 x 15m = 1h
            atr_period: int = 14,
            stop_atr: float = 2.0,
            tp_r: float = 3.0,
            cost_roundtrip_pct: float = 0.003,  # AJUSTA à tua corretora
            min_risk_over_cost: float = 3.0,
            cooldown_bars: int = 8,
    ):
        closes = [float(x) for x in self.closes]
        highs = [float(x) for x in self.highs]
        lows = [float(x) for x in self.lows]
        n = len(closes)

        signals = [Signal.HOLD] * n
        stops = [0.0] * n
        tps = [0.0] * n
        funnel = {f"{t}_{k}": 0 for t in ("long", "short")
                  for k in ("regime", "retest", "confirmed", "cost_ok", "signals")}
        self.last_funnel = funnel

        warmup = ema_regime_slow + tf_factor * (atr_period + 2)
        if n <= warmup:
            return signals, stops, tps

        e_fast = self._ema(closes, ema_regime_fast)
        e_slow = self._ema(closes, ema_regime_slow)
        e_a = self._ema(closes, ema_a)
        e_b = self._ema(closes, ema_b)

        hh, hl, hc = self._aggregate(highs, lows, closes, tf_factor)
        atr_1h = self._atr_wilder(hh, hl, hc, atr_period)

        last_signal = -10 ** 9
        for i in range(warmup, n):
            c = closes[i]

            # 1) REGIME (long e short são mutuamente exclusivos)
            if c > e_slow[i] and e_fast[i] > e_slow[i]:
                side = 1
            elif c < e_slow[i] and e_fast[i] < e_slow[i]:
                side = -1
            else:
                continue
            if (side == 1 and not enable_long) or (side == -1 and not enable_short):
                continue
            tag = "long" if side == 1 else "short"
            funnel[tag + "_regime"] += 1

            # 2) RECUO / REPIQUE à zona EMA20/EMA50 sem a perder no fecho
            touched, broke = False, False
            for j in range(i - retest_window, i + 1):
                if side == 1:
                    if lows[j] <= max(e_a[j], e_b[j]):
                        touched = True
                    if closes[j] < min(e_a[j], e_b[j]):
                        broke = True
                else:
                    if highs[j] >= min(e_a[j], e_b[j]):
                        touched = True
                    if closes[j] > max(e_a[j], e_b[j]):
                        broke = True
            if not touched or broke:
                continue
            funnel[tag + "_retest"] += 1

            # 3) CONFIRMAÇÃO
            if side == 1:
                ok = c > e_a[i] and c > highs[i - 1]
            else:
                ok = c < e_a[i] and c < lows[i - 1]
            if not ok:
                continue
            funnel[tag + "_confirmed"] += 1

            # 4) ATR de 1h usando apenas blocos de 1h já fechados até à vela i
            done = (i + 1) // tf_factor
            a = atr_1h[done - 1] if done >= 1 else None
            if a is None or a <= 0:
                continue

            risk = stop_atr * a
            sl = c - side * risk
            if (risk / c) < min_risk_over_cost * cost_roundtrip_pct:
                continue
            funnel[tag + "_cost_ok"] += 1

            # 5) COOLDOWN
            if i - last_signal < cooldown_bars:
                continue
            last_signal = i

            signals[i] = Signal.BUY if side == 1 else Signal.SELL
            stops[i] = sl
            tps[i] = c + side * tp_r * risk
            funnel[tag + "_signals"] += 1

        return signals, stops, tps

    # --------------------------------------------- avaliação rápida em R
    def _stats(self, R, bars, risks):
        if not R:
            return {"n_trades": 0}
        wins = [r for r in R if r > 0]
        losses = [r for r in R if r <= 0]
        cum, peak, dd = 0.0, 0.0, 0.0
        for r in R:
            cum += r
            peak = max(peak, cum)
            dd = max(dd, peak - cum)
        return {
            "n_trades": len(R),
            "win_rate": round(len(wins) / len(R), 3),
            "expectancy_R": round(sum(R) / len(R), 3),
            "avg_win_R": round(sum(wins) / len(wins), 2) if wins else 0.0,
            "avg_loss_R": round(sum(losses) / len(losses), 2) if losses else 0.0,
            "profit_factor": round(sum(wins) / -sum(losses), 2) if losses and sum(losses) < 0 else None,
            "max_dd_R": round(dd, 2),
            "avg_risk_pct": round(100 * sum(risks) / len(risks), 2),
            "avg_bars": round(sum(bars) / len(bars), 1),
        }

    def quick_eval_R(self, opens, highs, lows, closes, signals, stops, tps,
                     fee=0.001, slippage=0.0005, entry="next_open",
                     buy=None, sell=None, side="both", return_trades=False,
                     allow_overlap=False):
        """Simula SL/TP fixos (sem trailing e sem funding) para medir a expectância.
        Entrada na abertura da vela seguinte (realista) ou no fecho do sinal.
        SL e TP na mesma vela -> assume SL primeiro. Uma posição de cada vez.
        side: "both", "long" ou "short". Fee=0 e slippage=0 dão o resultado BRUTO.
        Devolve métricas globais + 'by_side' com long e short separados.
        allow_overlap=True avalia CADA sinal de forma independente, mesmo com outro trade
        aberto (dá mais amostras para diagnóstico, mas trades sobrepostos são correlacionados;
        não uses essas métricas como desempenho real)."""
        enum = type(signals[0])
        buy = buy if buy is not None else enum.BUY
        sell = sell if sell is not None else getattr(enum, "SELL", None)
        n = len(closes)
        trades = []  # (direção, R, barras, risco%)
        next_free = 0
        for i in range(n - 1):
            if i < next_free and not allow_overlap:
                continue
            if signals[i] == buy and side in ("both", "long"):
                d = 1
            elif sell is not None and signals[i] == sell and side in ("both", "short"):
                d = -1
            else:
                continue
            sl, tp = stops[i], tps[i]
            px = opens[i + 1] if entry == "next_open" else closes[i]
            ent = px * (1 + d * slippage)  # long paga mais, short vende mais baixo
            risk = d * (ent - sl)
            if risk <= 0 or d * (tp - ent) <= 0:
                continue
            ex, k = None, None
            for j in range(i + 1, n):
                if d == 1:
                    if lows[j] <= sl:
                        ex, k = min(opens[j], sl), j
                        break
                    if highs[j] >= tp:
                        ex, k = tp, j
                        break
                else:
                    if highs[j] >= sl:
                        ex, k = max(opens[j], sl), j
                        break
                    if lows[j] <= tp:
                        ex, k = tp, j
                        break
            if ex is None:
                ex, k = closes[-1], n - 1
            ex_eff = ex * (1 - d * slippage)
            pnl = d * (ex_eff - ent) - fee * (ent + ex_eff)
            trades.append((d, pnl / risk, k - i, risk / ent, i, pnl / ent))
            next_free = k

        def pick(self,dirn=None):
            t = [x for x in trades if dirn is None or x[0] == dirn]
            return self._stats([x[1] for x in t], [x[2] for x in t], [x[3] for x in t])

        out = pick()
        out["by_side"] = {"long": pick(1), "short": pick(-1)}
        if return_trades:
            out["trades"] = [{"i": x[4], "dir": x[0], "R": round(x[1], 4), "ret": round(x[5], 6)} for x in trades]
        return out

    # ------------------------------------- que TP é alcançável? (grelha de TP)
    def tp_grid(self, opens, highs, lows, closes, signals, stops, ks=(1.0, 1.5, 2.0, 3.0),
                fee=0.00045, slippage=0.0002, buy=None, sell=None):
        """Para os MESMOS sinais e stops, varia só a distância do TP (em R) e mede:
          hit_rate   : fração de trades que acaba em lucro (≈ chega ao TP)
          benchmark  : 1/(1+k) = taxa de acerto esperada SEM edge, num mercado sem drift
          expectancy : expectância líquida em R (SL e TP fixos, sem trailing)
        Se hit_rate ≈ benchmark, a entrada não acrescenta nada. Se hit_rate fica
        claramente acima e a expectância líquida é > 0, há edge nessa distância de TP.
        Defaults de custos: perps da Hyperliquid, escalão base, como taker."""
        enum = type(signals[0])
        buy = buy if buy is not None else enum.BUY
        sell = sell if sell is not None else getattr(enum, "SELL", None)
        out = []
        for k in ks:
            tps = [0.0] * len(closes)
            for i, s in enumerate(signals):
                if s == buy:
                    d = 1
                elif sell is not None and s == sell:
                    d = -1
                else:
                    continue
                tps[i] = closes[i] + d * k * abs(closes[i] - stops[i])
            r = self.quick_eval_R(opens, highs, lows, closes, signals, stops, tps,
                             fee=fee, slippage=slippage, buy=buy, sell=sell)
            out.append({
                "tp_R": k,
                "n_trades": r["n_trades"],
                "hit_rate": r.get("win_rate"),
                "benchmark": round(1 / (1 + k), 3),
                "expectancy_R": r.get("expectancy_R"),
                "avg_bars": r.get("avg_bars"),
            })
        return out

    # ------------------------------- a tendência está a esgotar-se? (medidas)
    def signal_features(self,opens, highs, lows, closes, signals, buy=None, sell=None,
                        ema_regime_fast=200, ema_regime_slow=800, tf_factor=4,
                        atr_period=14, look=96):
        """Para cada sinal, calcula medidas de maturidade/esgotamento da tendência
        usando SÓ dados até à vela do sinal. Todas são normalizadas pelo ATR de 1h e
        orientadas pela direção do trade (positivo = a favor da tendência).
          ext_atr             : distância do preço à EMA lenta (em ATR). Alto = esticado.
          regime_age_h        : horas desde que o regime atual começou. Alto = tendência velha.
          slope_atr           : subida da EMA rápida nas últimas `look` velas (24h por defeito).
          gap_change_atr      : variação da distância EMA rápida-EMA lenta. Negativo = a fechar.
          accel_atr           : momento atual menos o momento anterior. Negativo = a desacelerar.
          pullback_depth_atr  : profundidade do recuo face ao extremo recente. Alto = recuo fundo.
        Devolve {índice_do_sinal: {medidas}}."""
        enum = type(signals[0])
        buy = buy if buy is not None else enum.BUY
        sell = sell if sell is not None else getattr(enum, "SELL", None)
        n = len(closes)
        e_fast = self._ema(closes, ema_regime_fast)
        e_slow = self._ema(closes, ema_regime_slow)
        hh, hl, hc = self._aggregate(highs, lows, closes, tf_factor)
        atr_1h = self._atr_wilder(hh, hl, hc, atr_period)

        age, prev = [0] * n, 0
        for i in range(n):
            if closes[i] > e_slow[i] and e_fast[i] > e_slow[i]:
                side = 1
            elif closes[i] < e_slow[i] and e_fast[i] < e_slow[i]:
                side = -1
            else:
                side = 0
            age[i] = (age[i - 1] + 1 if (i > 0 and side == prev) else 1) if side else 0
            prev = side

        out = {}
        for i in range(2 * look, n):
            if signals[i] == buy:
                d = 1
            elif sell is not None and signals[i] == sell:
                d = -1
            else:
                continue
            done = (i + 1) // tf_factor
            a = atr_1h[done - 1] if done >= 1 else None
            if a is None or a <= 0:
                continue
            mom_now = d * (closes[i] - closes[i - look])
            mom_prev = d * (closes[i - look] - closes[i - 2 * look])
            extreme = max(highs[i - look + 1:i + 1]) if d == 1 else min(lows[i - look + 1:i + 1])
            gap_now = d * (e_fast[i] - e_slow[i]) / a
            gap_before = d * (e_fast[i - look] - e_slow[i - look]) / a
            out[i] = {
                "dir": d,
                "ext_atr": d * (closes[i] - e_slow[i]) / a,
                "regime_age_h": age[i] * 15 / 60,
                "slope_atr": d * (e_fast[i] - e_fast[i - look]) / a,
                "gap_change_atr": gap_now - gap_before,
                "accel_atr": (mom_now - mom_prev) / a,
                "pullback_depth_atr": d * (extreme - closes[i]) / a,
            }
        return out

    def feature_report(self,features, trades, min_per_bucket=8):
        """Junta as medidas com o resultado de cada trade (em R) e, para cada medida,
        divide os trades em tercis (baixo/médio/alto) e mostra nº, taxa de acerto e
        expectância em R. Uma medida só serve se os tercis forem claramente diferentes
        E isso se repetir noutros ativos e períodos. `trades` vem de
        quick_eval_R(..., return_trades=True)["trades"]."""
        rows = [(features[t["i"]], t["R"], t.get("pnl")) for t in trades if t["i"] in features]
        if len(rows) < 3 * min_per_bucket:
            return {"aviso": f"poucos trades ({len(rows)}): precisas de mais dados"}
        report = []
        for name in next(iter(features.values())):
            if name == "dir":
                continue
            ordered = sorted(rows, key=lambda r: r[0][name])
            k = len(ordered) // 3
            buckets = [ordered[:k], ordered[k:2 * k], ordered[2 * k:]]
            report.append({
                "medida": name,
                "tercis (baixo, médio, alto)": [
                    {"n": len(b),
                     "valor_mediano": round(sorted(r[0][name] for r in b)[len(b) // 2], 2),
                     "acerto": round(sum(1 for r in b if r[1] > 0) / len(b), 3),
                     "exp_R": round(sum(r[1] for r in b) / len(b), 3),
                     **({"pnl_total": round(sum(r[2] for r in b), 1),
                         "pnl_medio": round(sum(r[2] for r in b) / len(b), 2)}
                        if all(r[2] is not None for r in b) else {})}
                    for b in buckets],
            })
        return report

    # ----------------------------- filtro opcional de fim de tendência (A/B)
    def exhaustion_filter(self,opens, highs, lows, closes, signals, stops, tps,
                          buy=None, sell=None, look=96,
                          max_ext_atr=None,  # veta se o preço está esticado demais face à EMA lenta
                          max_regime_age_h=None,  # veta se o regime já dura demasiadas horas
                          min_slope_atr=None,  # veta se a EMA rápida quase não sobe/desce
                          min_gap_change_atr=None,  # veta se as EMAs não estão a abrir o suficiente
                          min_accel_atr=None,  # veta se o momento está a desacelerar demais
                          max_pullback_atr=None):  # veta se o recuo é demasiado fundo
        """Pós-filtro sobre os sinais já gerados. Todos os limiares a None = sem filtro
        (devolve os sinais iguais). Usa só dados até à vela do sinal (causal).
        Devolve (signals, stops, tps, n_vetados). Define os limiares a partir do
        feature_report (tercis) num período e confirma noutro; os valores de exemplo
        não são recomendações."""
        enum = type(signals[0])
        hold = enum.HOLD
        feats = self.signal_features(opens, highs, lows, closes, signals,
                                buy=buy, sell=sell, look=look)
        s2, sl2, tp2 = list(signals), list(stops), list(tps)
        vetoed = 0
        for i, f in feats.items():
            veto = (
                    (max_ext_atr is not None and f["ext_atr"] > max_ext_atr) or
                    (max_regime_age_h is not None and f["regime_age_h"] > max_regime_age_h) or
                    (min_slope_atr is not None and f["slope_atr"] < min_slope_atr) or
                    (min_gap_change_atr is not None and f["gap_change_atr"] < min_gap_change_atr) or
                    (min_accel_atr is not None and f["accel_atr"] < min_accel_atr) or
                    (max_pullback_atr is not None and f["pullback_depth_atr"] > max_pullback_atr)
            )
            if veto:
                s2[i], sl2[i], tp2[i] = hold, 0.0, 0.0
                vetoed += 1
        return s2, sl2, tp2, vetoed

    # -------------------- as piores entradas distinguem-se nalguma medida?
    def worst_trades_profile(self,features, trades, n=10, min_trades=20):
        """Para os n piores trades (menor R), mostra em que PERCENTIL cada medida
        se situa face a todos os trades (0 = o valor mais baixo de todos, 100 = o mais alto).
        Se os piores trades se concentram num extremo de alguma medida (média do percentil
        perto de 100 ou de 0), essa medida pode servir de filtro. Se a média fica por volta
        de 50, a medida não os distingue e o filtro não os apanha.
        Referência: a média dos percentis de n trades ao acaso varia ~29/sqrt(n) pontos em
        torno de 50 (n=10: ±9). Só valores claramente fora de 35-65 merecem atenção, e
        precisam de confirmação noutros ativos e períodos.
        `trades` vem de quick_eval_R(..., return_trades=True)["trades"]."""
        rows = [(t, features[t["i"]]) for t in trades if t["i"] in features]
        if len(rows) < max(n, min_trades):
            # devolve sempre as mesmas chaves, para o código que chama não rebentar
            return {"aviso": f"poucos trades com medidas ({len(rows)}); precisas de pelo menos {max(n, min_trades)}",
                    "n_trades": len(rows), "n_piores": 0,
                    "percentil_medio_dos_piores": None, "detalhe": []}
        names = [k for k in rows[0][1] if k != "dir"]
        sorted_vals = {nm: sorted(f[nm] for _, f in rows) for nm in names}
        worst = sorted(rows, key=lambda r: r[0]["R"])[:n]
        detail = []
        for t, f in worst:
            pct = {nm: round(100 * sum(1 for v in sorted_vals[nm] if v <= f[nm]) / len(rows))
                   for nm in names}
            detail.append({"i": t["i"], "dir": f["dir"], "R": t["R"], "pnl": t.get("pnl"), "percentis": pct})
        summary = {nm: round(sum(d["percentis"][nm] for d in detail) / len(detail)) for nm in names}
        return {"n_trades": len(rows), "n_piores": n,
                "percentil_medio_dos_piores": summary, "detalhe": detail}

    # ------------------- onde muda o desempenho? (varredura de limiar por medida)
    def threshold_scan(self,features, trades, name, thresholds, group_fn=None, tp_r=3.0, episode_fn=None):
        """Para cada limiar T compara os sinais com `name` > T (acima) e <= T (abaixo):
        n, taxa de acerto, expectância em R e z da taxa de acerto face à referência
        1/(1+tp_r) (sem edge). group_fn(chave_do_sinal) -> rótulo (ex.: o ativo ou a
        metade do período) repete o grupo 'acima' por rótulo, para veres se o efeito
        se repete. episode_fn(chave, medidas) -> id do episódio (ex.: onde o regime começou):
        acrescenta o nº de episódios independentes e a expectância média por episódio, porque
        sinais do mesmo episódio partilham o mesmo caminho de preço (amostra efetiva menor).
        `trades` vem de quick_eval_R(..., return_trades=True)['trades'].
        Cuidado: comparar muitos limiares e medidas gera falsos positivos; valida
        sempre noutros ativos/períodos antes de usares um limiar."""
        import math
        rows = [(t["i"], features[t["i"]][name], t["R"], t.get("pnl")) for t in trades if t["i"] in features]
        bench = 1 / (1 + tp_r)

        def agg(sub):
            n = len(sub)
            if n == 0:
                return {"n": 0}
            hit = sum(1 for x in sub if x[2] > 0) / n
            res = {"n": n, "acerto": round(hit, 3),
                   "exp_R": round(sum(x[2] for x in sub) / n, 3),
                   "z": round((hit - bench) / math.sqrt(bench * (1 - bench) / n), 1)}
            if all(x[3] is not None for x in sub):
                res["pnl_total"] = round(sum(x[3] for x in sub), 1)
                res["pnl_medio"] = round(sum(x[3] for x in sub) / n, 2)
            if episode_fn:
                eps = {}
                for x in sub:
                    eps.setdefault(episode_fn(x[0], features[x[0]]), []).append(x[2])
                res["episodios"] = len(eps)
                res["exp_R_por_episodio"] = round(sum(sum(v) / len(v) for v in eps.values()) / len(eps), 3)
            return res

        out = []
        for T in thresholds:
            item = {"limiar": T,
                    "acima": agg([x for x in rows if x[1] > T]),
                    "abaixo": agg([x for x in rows if x[1] <= T])}
            if group_fn:
                grupos = sorted({group_fn(x[0]) for x in rows})
                item["acima_por_grupo"] = {g: agg([x for x in rows if x[1] > T and group_fn(x[0]) == g])
                                           for g in grupos}
                item["abaixo_por_grupo"] = {g: agg([x for x in rows if x[1] <= T and group_fn(x[0]) == g])
                                            for g in grupos}
            out.append(item)
        return out

    # ------------- simulador das regras de saída do bot (momentum + escada + TP 5%)
    LADDER_PADRAO = ((0.02, 0.015), (0.01, 0.006), (0.004, 0.001), (0.002, 0.0))

    def _ladder_secure(self,p, tiers, wake, giveback, dynamic=True):
        """Lucro a garantir (fração do preço, ou múltiplos de R se as unidades forem R) para um PnL p.
        None = não mexer. Abaixo de `wake` não faz nada; até ao maior alvo usa a tabela; acima,
        garante (1 - giveback) do PnL (dynamic=True) ou mantém o último degrau (dynamic=False)."""
        if p < wake:
            return None
        if p <= tiers[0][0]:
            for target, secure in tiers:
                if p >= target:
                    return secure
        return p * (1 - giveback) if dynamic else tiers[0][1]

    def quick_eval_exits(self,opens, highs, lows, closes, signals, stops, tps,
                         fee=0.00045, slippage=0.0002, buy=None, sell=None, side="both",
                         momentum=True, mom_activation=0.006, mom_giveback=0.0025,
                         ladder=True, ladder_wake=0.008, ladder_tiers=LADDER_PADRAO,
                         ladder_giveback=0.30, tp_after_ladder=0.05, eval_mode="close",
                         units="pct", ladder_dynamic=True):
        """Simula as regras de saída do teu bot sobre os sinais, uma posição de cada vez:
          SL e TP do sinal (ordens na exchange; SL primeiro se ambos na mesma vela);
          momentum: com pico >= mom_activation, fecha a mercado se o PnL recuar mom_giveback do pico;
          escada: sobe o SL para entrada*(1+/-secure) conforme o PnL (só sobe) e, na 1.ª subida,
          substitui o TP por tp_after_ladder (None = mantém o TP do sinal). SEM timeout.
        eval_mode='close': momentum e escada avaliados ao fecho de cada vela (backtest típico);
        eval_mode='intrabar': pior caso ~tick a tick (pico pelas máximas/mínimas da vela, o recuo
        pode ocorrer na própria vela). Efeitos da escada aplicam-se a partir da vela seguinte.
        units='pct': limiares em fração do preço (como no bot). units='R': limiares em múltiplos do
        risco inicial do trade (mom_activation=1.0 => +1R; tiers/wake/tp_after_ladder também em R).
        Devolve estatísticas + 'trades' [{i, dir, R, ret, motivo, barras}] e 'motivos'."""
        enum = type(signals[0])
        buy = buy if buy is not None else enum.BUY
        sell = sell if sell is not None else getattr(enum, "SELL", None)
        n = len(closes)
        out_trades, next_free = [], 0
        for i in range(n - 1):
            if i < next_free:
                continue
            if signals[i] == buy and side in ("both", "long"):
                d = 1
            elif sell is not None and signals[i] == sell and side in ("both", "short"):
                d = -1
            else:
                continue
            sl0, tp0 = stops[i], tps[i]
            ent = opens[i + 1] * (1 + d * slippage)
            risk = d * (ent - sl0)
            if risk <= 0 or d * (tp0 - ent) <= 0:
                continue
            sc = risk / ent if units == "R" else 1.0  # converte limiares em R para fração do preço
            m_act, m_gb, l_wake = mom_activation * sc, mom_giveback * sc, ladder_wake * sc
            tiers = tuple((t_ * sc, s_ * sc) for t_, s_ in ladder_tiers)
            tp_ov = tp_after_ladder * sc if tp_after_ladder else None
            sl, tp, peak, last_adj = sl0, tp0, 0.0, None
            ex = k = motivo = None
            for j in range(i + 1, n):
                if d == 1:
                    if lows[j] <= sl:
                        ex, k, motivo = min(opens[j], sl), j, "SL"
                        break
                    if highs[j] >= tp:
                        ex, k, motivo = tp, j, "TP"
                        break
                else:
                    if highs[j] >= sl:
                        ex, k, motivo = max(opens[j], sl), j, "SL"
                        break
                    if lows[j] <= tp:
                        ex, k, motivo = tp, j, "TP"
                        break
                if eval_mode == "close":
                    p_now = d * (closes[j] - ent) / ent
                    peak = max(peak, p_now)
                    if momentum and peak >= m_act and peak - p_now >= m_gb:
                        ex, k, motivo = closes[j], j, "MOMENTUM"
                        break
                    p_lad = p_now
                else:
                    fav = highs[j] if d == 1 else lows[j]
                    adv = lows[j] if d == 1 else highs[j]
                    peak = max(peak, d * (fav - ent) / ent)
                    if momentum and peak >= m_act and peak - d * (adv - ent) / ent >= m_gb:
                        ex, k, motivo = ent * (1 + d * (peak - m_gb)), j, "MOMENTUM"
                        break
                    p_lad = peak
                if ladder:
                    adj = self._ladder_secure(p_lad, tiers, l_wake, ladder_giveback, ladder_dynamic)
                    if adj is not None and (last_adj is None or adj > last_adj):
                        last_adj = adj
                        new_sl = ent * (1 + d * adj)
                        if d * (new_sl - sl) > 0:
                            sl = new_sl
                        if tp_ov:
                            tp = ent * (1 + d * tp_ov)
            if ex is None:
                ex, k, motivo = closes[-1], n - 1, "FIM"
            ex_eff = ex * (1 - d * slippage)
            pnl = d * (ex_eff - ent) - fee * (ent + ex_eff)
            out_trades.append({"i": i, "dir": d, "R": round(pnl / risk, 4), "ret": round(pnl / ent, 6),
                               "motivo": motivo, "barras": k - i, "risk_pct": risk / ent,
                               "entrada": ent, "saida": ex, "j": k})
            next_free = k
        res = self._stats([t["R"] for t in out_trades], [t["barras"] for t in out_trades],
                     [t["risk_pct"] for t in out_trades])
        res["trades"] = out_trades
        res["motivos"] = {}
        for t in out_trades:
            res["motivos"][t["motivo"]] = res["motivos"].get(t["motivo"], 0) + 1
        return res