import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import argrelextrema, find_peaks

from commons.enums.mode_enum import ModeEnum
from commons.enums.signal_enum import Signal
from commons.models.ohlcv_type_dclass import Ohlcv
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from strategies.indicators import Indicators


class StrategyUtils:

    @staticmethod    
    def detect_reversal_pattern(ohlcv: OhlcvWrapper, rsi_overbought: float = 70, rsi_oversold: float = 30) -> tuple[bool, bool]:
        """
        Detecta reversão de topo (baixa) ou de fundo (alta) com base em padrão de candle + RSI.
        Retorna (is_bearish_reversal, is_bullish_reversal).
        """
        indicators = Indicators(ohlcv)

        rsi = list(indicators.rsi(period=14))
        if len(ohlcv) < 2 or len(rsi) < 2:
            return False, False

        recent = ohlcv.get_recent_closed(lookback=2)
        prev = recent[-2]
        curr = recent[-1]

        body = abs(curr.close - curr.open)
        candle_range = curr.high - curr.low
        upper_wick = curr.high - max(curr.close, curr.open)
        lower_wick = min(curr.close, curr.open) - curr.low

        if candle_range == 0:
            return False, False

        body_ratio = body / candle_range
        upper_wick_ratio = upper_wick / candle_range
        lower_wick_ratio = lower_wick / candle_range

        rsi_now = rsi[-1]

        # Reversão de topo (estrela cadente + RSI sobrecomprado)
        is_bearish = (
            body_ratio < 0.3 and
            upper_wick_ratio > 0.5 and
            curr.close < curr.open and
            rsi_now >= rsi_overbought
        )

        # Reversão de fundo (martelo + RSI sobrevendido)
        is_bullish = (
            body_ratio < 0.3 and
            lower_wick_ratio > 0.5 and
            curr.close > curr.open and
            rsi_now <= rsi_oversold
        )

        # Engolfo de baixa como fallback
        if not is_bearish:
            is_bearish = StrategyUtils.is_bearish_engulfing(prev, curr)

        return is_bearish, is_bullish
    
    @staticmethod
    def is_bearish_engulfing(prev: Ohlcv, curr: Ohlcv) -> bool:
        return (
            prev.close > prev.open and  # Candle anterior é de alta
            curr.close < curr.open and  # Candle atual é de baixa
            curr.open > prev.close and  # Abertura acima do fechamento anterior
            curr.close < prev.open      # Fechamento abaixo da abertura anterior
        )
    @staticmethod
    def is_bullish_engulfing(prev: Ohlcv, curr: Ohlcv) -> bool:
        return (
            prev.close < prev.open and  # Candle anterior é de baixa
            curr.close > curr.open and  # Candle atual é de alta
            curr.open < prev.close and  # Abertura abaixo do fechamento anterior
            curr.close > prev.open      # Fechamento acima da abertura anterior
        )

    @staticmethod
    def detect_support_resistance(
        candles: OhlcvWrapper,
        lookback: int = 50,
        tolerance_pct: float = 0.01  # 1%
    ) -> tuple[float, float]:
        highs = [c.high for c in candles.get_recent_closed(lookback)]
        lows = [c.low for c in candles.get_recent_closed(lookback)]

        resistance = max(highs)
        support = min(lows)

        return resistance, support
    
    @staticmethod
    def detect_multiple_support_resistance(
        candles: OhlcvWrapper,
        lookback: int = 50,
        tolerance_pct: float = 0.005
    ) -> tuple[list[float], list[float]]:
        """
        Detecta múltiplos níveis de suporte e resistência com base em extremos locais.
        Os níveis próximos são agrupados com base em `tolerance_pct`.

        :return: (lista de resistências, lista de suportes)
        """
        highs = [c.high for c in candles.get_recent_closed(lookback)]
        lows = [c.low for c in candles.get_recent_closed(lookback)]

        resistances = []
        supports = []

        for i in range(2, len(highs) - 2):
            # Máximo local
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                price = highs[i]
                if not any(abs(price - r) / price < tolerance_pct for r in resistances):
                    resistances.append(price)

            # Mínimo local
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                price = lows[i]
                if not any(abs(price - s) / price < tolerance_pct for s in supports):
                    supports.append(price)

        return sorted(resistances, reverse=True), sorted(supports)

    @staticmethod
    def get_distance_to_levels(ohlcv: OhlcvWrapper, price_ref: float, lookback: int = 50) -> tuple[float, float]:
        recent = ohlcv.get_recent_closed(lookback)

        highs = [candle.high for candle in recent]
        lows = [candle.low for candle in recent]

        resistance = max(highs)
        support = min(lows)
        price = price_ref  # ou self.ohlcv.get_current_candle().close

        dist_to_res = abs(resistance - price)
        dist_to_sup = abs(price - support)

        return dist_to_res, dist_to_sup
    
    
    @staticmethod
    def is_flat_candle(ohlcv: OhlcvWrapper):
        candle = ohlcv.get_current_candle()
        return candle.open == candle.close and candle.high == candle.low
    
    @staticmethod
    def calculate_bands(ohlcv: OhlcvWrapper, multiplier):
        indicators = Indicators(ohlcv)

        closes = ohlcv.closes

        atr = indicators.atr()
        upper_band = [closes[i] + multiplier * atr[i] for i in range(len(atr))]
        lower_band = [closes[i] - multiplier * atr[i] for i in range(len(atr))]

        return upper_band, lower_band

    @staticmethod
    def detect_lateral_market(ohlcv: OhlcvWrapper, adx_threshold) -> bool:
        indicators = Indicators(ohlcv)
        adx = indicators.adx()
        adx_now = adx[-1]
        lateral_market = adx_now < adx_threshold
        return lateral_market
    
    @staticmethod
    def trend_signal_with_adx(ohlcv: OhlcvWrapper, adx_threshold: float):
        indicators = Indicators(ohlcv)
        ema = indicators.ema()[-1]
        prev_ema = indicators.ema()[-2]

        if StrategyUtils.detect_lateral_market(ohlcv, adx_threshold):
            if ema > prev_ema:
                return 1  # buy
            elif ema < prev_ema:
                return -1  # sell
        return 0  # sem sinal
    
    @staticmethod
    def detect_setup_123(ohlcv: OhlcvWrapper):

        closes = ohlcv.closes
        highs = ohlcv.highs
        lows = ohlcv.lows
        price = ohlcv.opens[0]

        if len(closes) < 5:
            return False, False

        h = highs[-5:]
        l = lows[-5:]

        # Buy Setup 123
        buy_123 = (
            l[2] < l[1] and l[2] < l[3] and  # ponto 2 é mínimo local
            h[3] > h[2] and  # ponto 3 alta local
            l[4] > l[2] and  # ponto 4 confirma subida
            price > h[3]  # preço acima de ponto 3
        )

        # Sell Setup 123
        sell_123 = (
            h[2] > h[1] and h[2] > h[3] and  # ponto 2 é máximo local
            l[3] < l[2] and  # ponto 3 baixa local
            h[4] < h[2] and  # ponto 4 confirma queda
            price < l[3]  # preço abaixo do ponto 3
        )

        return buy_123, sell_123
    
    @staticmethod
    def is_breakout_candle(ohlcv: OhlcvWrapper, idx: int, multiplier: float = 2.0, window: int = 20) -> bool:
        closes = ohlcv.closes
        opens = ohlcv.opens
        
        if idx < window:
            return False

        candle_bodies = [abs(closes[i] - opens[i]) for i in range(idx - window, idx)]
        avg_body = sum(candle_bodies) / window
        current_body = abs(closes[idx] - opens[idx])

        return current_body > multiplier * avg_body
    
    @staticmethod
    def check_price_action_signals(ohlcv: OhlcvWrapper) -> Signal:
        closes = ohlcv.closes
        opens = ohlcv.opens
        open_now = opens[-1]
        close_now = closes[-1]


       # Prioridade 1: Breakout candle (mais forte)
        if StrategyUtils.is_breakout_candle(ohlcv, -1):
            return Signal.BUY  # ou True, se quiseres só booleano

        # Prioridade 2: Setup 123
        buy_123, sell_123 = StrategyUtils.detect_setup_123(ohlcv)
        if buy_123:
            return Signal.BUY
        if sell_123:
            return Signal.SELL

        # Prioridade 3: Cor da vela atual
        if close_now > open_now:
            return Signal.BUY
        elif close_now < open_now:
            return Signal.SELL

        # Se nenhuma condição for satisfeita
        return Signal.HOLD
    
    @staticmethod
    def calculate_sl_tp(ohlcv: OhlcvWrapper, price_ref: float, side: Signal, mode: ModeEnum, sl_multiplier_aggressive: float, tp_multiplier_aggressive: float, sl_multiplier_conservative: float, tp_multiplier_conservative: float):
        indicators = Indicators(ohlcv)
        atr = indicators.atr()
        #atr_value = atr[-1]
        
        atr_avg = atr[-1]  # já está suavizado
        
        if mode == ModeEnum.AGGRESSIVE:
            sl_dist = sl_multiplier_aggressive * atr_avg
            tp_dist = tp_multiplier_aggressive * atr_avg
        else:
            sl_dist = sl_multiplier_conservative * atr_avg
            tp_dist = tp_multiplier_conservative * atr_avg

        if side == Signal.BUY:
            sl = price_ref - sl_dist
            tp = price_ref + tp_dist
        else:
            sl = price_ref + sl_dist
            tp = price_ref - tp_dist

        #print(f"entry_price: {price_ref} SL: {sl} TP:{tp}")
        return sl, tp
    
   
    @staticmethod
    def get_dynamic_sl_tp(
        ohlcv: OhlcvWrapper,
        entry_price: float,
        signal: Signal,
        buy_score: float,
        sell_score: float,
        base_rr_target: float = 2.0,
        fib_levels=[0.382, 0.618, 0.786],
        support_lookback: int = 3,
        swing_lookback: int = 20,
        sl_buffer_pct: float = 0.003,
        max_sl_pct: float = 0.01,
        atr_lookback: int = 14,
        min_rr_target: float = 1.2,
        max_tp_pct: float = 0.01,    # Limite máximo TP 1% acima/abaixo da entrada
        max_tp_atr_mult: float = 3   # Limite máximo TP a 3 vezes o ATR da entrada
    ) -> tuple[float, float]:
        """
        Calcula SL e TP dinâmicos com base em suporte, risco, níveis Fibonacci,
        limitando o SL máximo e ajustando o RR alvo baseado na volatilidade (ATR),
        e limitando o TP máximo tanto em % como por múltiplos do ATR.
        """

        indicators = Indicators(ohlcv)
        atr = indicators.atr()[-1]  # assumir que é array e pegar último valor

        atr_pct = atr / entry_price

        # Ajustar RR alvo conforme ATR
        if atr_pct > 0.01:  # se volatilidade > 1%
            rr_target = max(min_rr_target, base_rr_target * 0.5)
        elif atr_pct > 0.005:
            rr_target = max(min_rr_target, base_rr_target * 0.75)
        else:
            rr_target = base_rr_target

        print("AQUIIIIII TP", rr_target)

        if signal == Signal.BUY:
            rr_target = rr_target * buy_score

            lows = ohlcv.lows[-support_lookback:]
            base_sl = min(lows)
            sl_price = base_sl * (1 - sl_buffer_pct)

            # Limitar SL para no máximo max_sl_pct abaixo da entrada
            sl_price_min = entry_price * (1 - max_sl_pct)
            if sl_price < sl_price_min:
                sl_price = sl_price_min

            # Garantir SL abaixo da entrada e do candle de entrada
            sl_price = min(sl_price, entry_price * (1 - 0.001))

            risco = entry_price - sl_price
            tp_init = entry_price + rr_target * risco

            swing_high = max(ohlcv.highs[-swing_lookback:])
            fib_prices = [swing_high - (swing_high - entry_price) * level for level in fib_levels]

            tp_price = tp_init
            closest_diff = float('inf')
            for fib_price in fib_prices:
                if fib_price > entry_price:
                    diff = abs(fib_price - tp_init)
                    if diff < closest_diff:
                        closest_diff = diff
                        tp_price = fib_price

            # Limites máximos para o TP
            max_tp_price_pct = entry_price * (1 + max_tp_pct)
            max_tp_price_atr = entry_price + atr * max_tp_atr_mult

            tp_price = max(tp_price, tp_init)
            tp_price = min(tp_price, max_tp_price_pct, max_tp_price_atr)

            print(f"[BUY] ATR: {atr:.4f}, ATR%: {atr_pct:.4f}, RR alvo ajustado: {rr_target:.2f}")
            print(f"[BUY] SL: {sl_price:.4f}, Risco: {risco:.4f}, TP inicial: {tp_init:.4f}")
            print(f"[BUY] Swing high: {swing_high:.4f}, Fibonacci levels: {[round(p,4) for p in fib_prices]}")
            print(f"[BUY] TP máximo %: {max_tp_price_pct:.4f}, TP máximo ATR: {max_tp_price_atr:.4f}")
            print(f"[BUY] TP final: {tp_price:.4f}")

            return round(sl_price, 4), round(tp_price, 4)

        elif signal == Signal.SELL:
            rr_target = rr_target * sell_score

            highs = ohlcv.highs[-support_lookback:]
            base_sl = max(highs)
            sl_price = base_sl * (1 + sl_buffer_pct)

            # Limitar SL para no máximo max_sl_pct acima da entrada
            sl_price_max = entry_price * (1 + max_sl_pct)
            if sl_price > sl_price_max:
                sl_price = sl_price_max

            # Garantir SL acima da entrada e do candle de entrada
            sl_price = max(sl_price, entry_price * (1 + 0.001))

            risco = sl_price - entry_price
            tp_init = entry_price - rr_target * risco

            swing_low = min(ohlcv.lows[-swing_lookback:])
            fib_prices = [swing_low + (entry_price - swing_low) * level for level in fib_levels]

            tp_price = tp_init
            closest_diff = float('inf')
            for fib_price in fib_prices:
                if fib_price < entry_price:
                    diff = abs(fib_price - tp_init)
                    if diff < closest_diff:
                        closest_diff = diff
                        tp_price = fib_price

            # Limites máximos para o TP
            max_tp_price_pct = entry_price * (1 - max_tp_pct)
            max_tp_price_atr = entry_price - atr * max_tp_atr_mult

            tp_price = min(tp_price, tp_init)
            tp_price = max(tp_price, max_tp_price_pct, max_tp_price_atr)

            print(f"[SELL] ATR: {atr:.4f}, ATR%: {atr_pct:.4f}, RR alvo ajustado: {rr_target:.2f}")
            print(f"[SELL] SL: {sl_price:.4f}, Risco: {risco:.4f}, TP inicial: {tp_init:.4f}")
            print(f"[SELL] Swing low: {swing_low:.4f}, Fibonacci levels: {[round(p,4) for p in fib_prices]}")
            print(f"[SELL] TP máximo %: {max_tp_price_pct:.4f}, TP máximo ATR: {max_tp_price_atr:.4f}")
            print(f"[SELL] TP final: {tp_price:.4f}")

            return round(sl_price, 4), round(tp_price, 4)

        else:
            raise ValueError("Signal inválido. Use Signal.BUY ou Signal.SELL.")

    
    @staticmethod   
    def calculate_hybrid_sl_tp(
        ohlcv: OhlcvWrapper, 
        signal: Signal, 
        score: float = 1.0,  # score entre 0 e 1
        atr_period=14, 
        psar_acceleration=0.02, 
        psar_maximum=0.2, 
        tp_multiplier=1.5,
        tolerance_pct=0.01,
        max_tp_atr_multiplier=2.0,
        max_support_resistance_dist_pct=0.02,
        max_sl_atr_multiplier=1.0,  # máximo distância SL = 1x ATR do preço
        max_sl_dist_pct=0.015       # máximo distância SL 1.5% do preço
    ):
        closes = ohlcv.closes
        highs = ohlcv.highs
        lows = ohlcv.lows

        psar_values = Indicators(ohlcv).psar(psar_acceleration, psar_maximum)
        last_psar = psar_values[-1]

        score = max(0.3, score)

        tr_values = []
        for i in range(1, len(highs)):
            tr = max(highs[i], closes[i-1]) - min(lows[i], closes[i-1])
            tr_values.append(tr)
        atr = np.mean(tr_values[-atr_period:]) if len(tr_values) >= atr_period else np.mean(tr_values)

        y = closes[-atr_period:]
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]
        slope_factor = max(0.5, 1 + slope * 10)

        last_close = closes[-1]

        # TP base
        if signal == Signal.BUY:
            tp_price = last_close + tp_multiplier * atr * slope_factor
        elif signal == Signal.SELL:
            tp_price = last_close - tp_multiplier * atr * slope_factor
        else:
            raise ValueError("signal deve ser Signal.BUY ou Signal.SELL")

        resistances, supports = StrategyUtils.detect_multiple_support_resistance(ohlcv, lookback=50, tolerance_pct=tolerance_pct)

        # Ajuste do TP como antes
        if signal == Signal.BUY:
            possible_rts = [r for r in resistances if last_close < r < tp_price and (r - last_close)/last_close <= max_support_resistance_dist_pct]
            if possible_rts:
                nearest_rt = min(possible_rts)
                tp_price = min(tp_price, nearest_rt)
            max_tp = last_close + max_tp_atr_multiplier * atr
            tp_price = min(tp_price, max_tp)
        elif signal == Signal.SELL:
            possible_sps = [s for s in supports if tp_price < s < last_close and (last_close - s)/last_close <= max_support_resistance_dist_pct]
            if possible_sps:
                nearest_sp = max(possible_sps)
                tp_price = max(tp_price, nearest_sp)
            max_tp = last_close - max_tp_atr_multiplier * atr
            tp_price = max(tp_price, max_tp)

        # Ajuste do SL
        if signal == Signal.BUY:
            # SL inicial pelo PSAR
            sl_price = min(last_psar, last_close * (1 - max_sl_dist_pct))
            # SL não pode ficar abaixo do suporte próximo muito longe
            #possible_sps = [s for s in supports if s < last_close and (last_close - s)/last_close <= max_sl_dist_pct]
            #if possible_sps:
            #    nearest_sp = max(possible_sps)
            #    sl_price = max(sl_price, nearest_sp)
            # Garantir que SL não ultrapasse 1x ATR do preço
            #sl_price = max(sl_price, last_close - max_sl_atr_multiplier * atr)
        else:  # SELL
            sl_price = max(last_psar, last_close * (1 + max_sl_dist_pct))
            #possible_rts = [r for r in resistances if r > last_close and (r - last_close)/last_close <= max_sl_dist_pct]
            #if possible_rts:
            #    nearest_rt = min(possible_rts)
            #    sl_price = min(sl_price, nearest_rt)
            #sl_price = min(sl_price, last_close + max_sl_atr_multiplier * atr)

        tp_price = last_close + (tp_price - last_close) * score
        sl_price = last_close + (sl_price - last_close) * (2 - score) / 2  # Exemplo: SL mais apertado se score baixo
        return sl_price , tp_price 


    @staticmethod
    def passes_volume_volatility_filter(ohlcv: OhlcvWrapper, symbol: (str | None), volume_threshold_ratio: float, atr_threshold_ratio: float):
        indicators = Indicators(ohlcv)
        atr = indicators.atr()

        volumes = getattr(indicators, 'volumes', None)
        if volumes is None or len(volumes) < 20:
            return True  # Sem dados suficientes

        avg_volume = sum(volumes[-20:]) / 20
        current_volume = volumes[-1]

        if current_volume < volume_threshold_ratio * avg_volume:
            logging.info(f"{symbol} - Volume baixo: {current_volume:.2f} < {volume_threshold_ratio*100:.0f}% da média ({avg_volume:.2f})")
            return False

        avg_atr = sum(atr[-20:]) / 20
        current_atr = atr[-1]

        if current_atr < atr_threshold_ratio * avg_atr:
            logging.info(f"{symbol} - ATR baixo: {current_atr:.4f} < {atr_threshold_ratio*100:.0f}% da média ({avg_atr:.4f})")
            return False

        return True
    
    @staticmethod
    def calculate_higher_tf_trend(ohlcv_higher: OhlcvWrapper, adx_threshold: float):
        if len(ohlcv_higher) < 21:
            # Sem dados suficientes para timeframe maior → assume neutro
            return 0

        indicators_htf = Indicators(ohlcv_higher)
        closes_htf = indicators_htf.closes
        atr_htf = indicators_htf.atr()
        ema_htf = indicators_htf.ema()
        adx_htf = indicators_htf.adx()

        ema_now = ema_htf[-1]
        ema_prev = ema_htf[-2]
        adx_now = adx_htf[-1]

        lateral = adx_now < adx_threshold

        if lateral:
            if ema_now > ema_prev:
                return 1
            elif ema_now < ema_prev:
                return -1
            else:
                return 0
        else:
            # Se mercado não lateral, confia na tendência EMA simples
            if ema_now > ema_prev:
                return 1
            elif ema_now < ema_prev:
                return -1
            else:
                return 0
            
    @staticmethod
    def stochastic(ohlcv: OhlcvWrapper) -> Signal:

        stoch_k, stoch_d = Indicators(ohlcv).stochastic()
        k_now, d_now = stoch_k[-1], stoch_d[-1]
        k_prev, d_prev = stoch_k[-2], stoch_d[-2]

        if k_now > d_now and k_prev <= d_prev:
            return Signal.BUY
        elif k_now < d_now and k_prev >= d_prev:
            return Signal.SELL
        return Signal.HOLD
    
    @staticmethod
    def momentum_signal_macd_cci(ohlcv: OhlcvWrapper, cci_period: int = 20, macd_fast: int = 12, macd_slow: int = 26, macd_signal: int = 9) -> Signal:
        """
        Calcula sinal combinado de momentum baseado em MACD e CCI.

        Retorna:
        - Signal.BUY se MACD cruza sinal para cima e CCI > 100
        - Signal.SELL se MACD cruza sinal para baixo e CCI < -100
        - Signal.HOLD caso contrário
        """
        indicators = Indicators(ohlcv)

        macd_line, signal_line = indicators.macd(macd_fast, macd_slow, macd_signal)
        cci = indicators.cci(cci_period)

        if len(macd_line) < 2 or len(signal_line) < 2 or len(cci) < 1:
            return Signal.HOLD

        macd_curr, macd_prev = macd_line[-1], macd_line[-2]
        signal_curr, signal_prev = signal_line[-1], signal_line[-2]
        cci_curr = cci[-1]

        # Cruzamento de MACD para cima (bullish)
        macd_bull_cross = macd_prev <= signal_prev and macd_curr > signal_curr
        # Cruzamento de MACD para baixo (bearish)
        macd_bear_cross = macd_prev >= signal_prev and macd_curr < signal_curr

        if macd_bull_cross and cci_curr > 100:
            return Signal.BUY
        elif macd_bear_cross and cci_curr < -100:
            return Signal.SELL
        else:
            return Signal.HOLD
        
    @staticmethod
    def is_weak_confirmation_candle(candle: Ohlcv, min_body_ratio: float = 0.3) -> bool:
        """
        Verifica se o candle é fraco como confirmação (corpo pequeno em relação ao range total).
        
        Um corpo pequeno (ex: <30% do range total) sugere indecisão, fraqueza ou falta de convicção.

        Args:
            candle (Ohlcv): O candle a ser analisado.
            min_body_ratio (float): Valor mínimo aceitável para o corpo/range (ex: 0.3 = 30%).

        Returns:
            bool: True se o candle for considerado fraco (corpo pequeno), False caso contrário.
        """
        high = candle.high
        low = candle.low
        open_ = candle.open
        close = candle.close

        range_total = high - low
        body_size = abs(close - open_)

        if range_total == 0:
            return True  # evitar divisão por zero, assume candle fraco

        body_ratio = body_size / range_total
        return body_ratio < min_body_ratio
        
    @staticmethod
    def find_local_extrema_swings_psar(ohlcv: OhlcvWrapper, sequential: bool = True):
        closes = ohlcv.closes
        timestamps = ohlcv.timestamps
        psar = Indicators(ohlcv).psar()

        high_pivots = []
        low_pivots = []
        pivots_high_index = []
        pivots_low_index = []

        last_pivot_type = None  # "high" ou "low"

        for i in range(1, len(psar)):
            # PSAR acima -> abaixo do preço = fundo
            if psar[i-1] > closes[i-1] and psar[i] < closes[i]:
                if (not sequential or last_pivot_type != "low"):
                    if not pivots_low_index or pivots_low_index[-1] != i:
                        low_pivots.append(timestamps[i])
                        pivots_low_index.append(i)
                        last_pivot_type = "low"

            # PSAR abaixo -> acima do preço = topo
            elif psar[i-1] < closes[i-1] and psar[i] > closes[i]:
                if (not sequential or last_pivot_type != "high"):
                    if not pivots_high_index or pivots_high_index[-1] != i:
                        high_pivots.append(timestamps[i])
                        pivots_high_index.append(i)
                        last_pivot_type = "high"

        return high_pivots, low_pivots, pivots_high_index, pivots_low_index
    
    @staticmethod
    def filter_close_pivots(pivots_idx, prices, min_distance=0.005):
        filtered = []
        if not pivots_idx:
            return filtered
        pivots_idx = sorted(pivots_idx)
        group = [pivots_idx[0]]

        for idx in pivots_idx[1:]:
            # Distância relativa no preço entre este pivot e último do grupo
            dist = abs(prices[idx] - prices[group[-1]]) / prices[group[-1]]
            if dist < min_distance:
                # Mantém só o pivot mais extremo do grupo
                if prices[idx] > prices[group[-1]]:
                    group[-1] = idx
            else:
                filtered.extend(group)
                group = [idx]
        filtered.extend(group)
        return filtered
    
    @staticmethod
    def filter_close_pivots_with_volume(pivots_idx, prices, volumes, avg_volume, min_distance=0.005):
        """
        Filtra pivots muito próximos, mantém só os mais extremos e com volume acima da média.

        pivots_idx: lista de índices dos pivots
        prices: lista de preços (highs ou lows)
        volumes: lista de volumes
        avg_volume: valor médio do volume de referência
        min_distance: distância mínima relativa no preço para separar pivots
        """
        filtered = []
        if not pivots_idx:
            return filtered

        pivots_idx = sorted(pivots_idx)
        group = []

        for idx in pivots_idx:
            # Ignora pivots com volume baixo
            if volumes[idx] < avg_volume:
                continue

            if not group:
                group = [idx]
                continue

            dist = abs(prices[idx] - prices[group[-1]]) / prices[group[-1]]
            if dist < min_distance:
                # Mantém só o pivot mais extremo no grupo
                if prices[idx] > prices[group[-1]]:
                    group[-1] = idx
            else:
                filtered.extend(group)
                group = [idx]

        filtered.extend(group)
        return filtered


    @staticmethod
    def find_pivots(ohlcv: OhlcvWrapper, left: int = 3, right: int = 3) -> Tuple[List[int], List[int]]:
        """
        Detecta pivôs locais de alta (topos) e baixa (fundos) no gráfico.
        Retorna dois arrays de índices: (pivots_high, pivots_low)
        """
        highs = ohlcv.highs
        lows = ohlcv.lows
        length = len(highs)
        pivots_high = []
        pivots_low = []

        for i in range(left, length - right):
            high_candidate = highs[i]
            is_pivot_high = all(high_candidate > highs[j] for j in range(i - left, i)) and \
                            all(high_candidate > highs[j] for j in range(i + 1, i + right + 1))
            if is_pivot_high:
                pivots_high.append(i)

            low_candidate = lows[i]
            is_pivot_low = all(low_candidate < lows[j] for j in range(i - left, i)) and \
                           all(low_candidate < lows[j] for j in range(i + 1, i + right + 1))
            if is_pivot_low:
                pivots_low.append(i)

        return pivots_high, pivots_low

    @staticmethod
    def detect_divergence(
        ohlcv: OhlcvWrapper
    ) -> Tuple[List[int], List[int]]:
        """
        Detecta divergências entre preço (pivôs) e RSI.
        Retorna listas de índices onde existem divergências de alta e baixa.
        """
        bullish_divergences = []
        bearish_divergences = []

        indicators = Indicators(ohlcv)
        rsi = list(indicators.rsi(14))

        pivots_high, pivots_low = StrategyUtils.find_pivots(ohlcv)

        # Divergência de alta: preços fazem pivôs baixos mais baixos, RSI faz pivôs baixos mais altos
        for i in range(1, len(pivots_low)):
            idx1 = pivots_low[i - 1]
            idx2 = pivots_low[i]
            price_low1 = ohlcv.lows[idx1]
            price_low2 = ohlcv.lows[idx2]
            rsi_low1 = rsi[idx1]
            rsi_low2 = rsi[idx2]

            if price_low2 < price_low1 and rsi_low2 > rsi_low1:
                bullish_divergences.append(idx2)

        # Divergência de baixa: preços fazem pivôs altos mais altos, RSI faz pivôs altos mais baixos
        for i in range(1, len(pivots_high)):
            idx1 = pivots_high[i - 1]
            idx2 = pivots_high[i]
            price_high1 = ohlcv.highs[idx1]
            price_high2 = ohlcv.highs[idx2]
            rsi_high1 = rsi[idx1]
            rsi_high2 = rsi[idx2]

            if price_high2 > price_high1 and rsi_high2 < rsi_high1:
                bearish_divergences.append(idx2)

        return bullish_divergences, bearish_divergences
    
    @staticmethod
    def get_candle_type(candle: Ohlcv) -> str:
        close_price = candle.close
        open_price = candle.open
        high_price = candle.high
        low_price = candle.low

        body = abs(close_price - open_price)
        candle_range = high_price - low_price
        upper_wick = high_price - max(open_price, close_price)
        lower_wick = min(open_price, close_price) - low_price

        # Evitar divisão por zero
        if candle_range == 0:
            return "doji"

        body_ratio = body / candle_range
        upper_ratio = upper_wick / candle_range
        lower_ratio = lower_wick / candle_range

        # Classificações básicas
        if body_ratio < 0.1:
            return "doji"
        if upper_ratio < 0.1 and lower_ratio > 0.6:
            return "hammer"
        if lower_ratio < 0.1 and upper_ratio > 0.6:
            return "shooting_star"
        if body_ratio > 0.6 and close_price > open_price:
            return "bullish"
        if body_ratio > 0.6 and close_price < open_price:
            return "bearish"
        if body_ratio > 0.5 and upper_ratio > 0.2 and lower_ratio > 0.2:
            return "marubozu"

        return "neutral"


    @staticmethod
    def has_large_wick(candle, ratio: float = 2.0) -> bool:
        body = abs(candle.close - candle.open)
        upper_wick = candle.high - max(candle.close, candle.open)
        lower_wick = min(candle.close, candle.open) - candle.low
        return upper_wick > body * ratio or lower_wick > body * ratio

    @staticmethod
    def is_single_candle_pump(candles: OhlcvWrapper, threshold: float = 0.03) -> bool:
        current = candles.get_current_candle()
        move = abs(current.close - current.open) / current.open
        return move > threshold

    @staticmethod
    def has_price_gap(candles: OhlcvWrapper, threshold: float = 0.02) -> bool:
        if len(candles) < 2:
            return False
        prev = candles.get_previous_candle()
        current = candles.get_current_candle()
        gap = abs(current.open - prev.close) / prev.close
        return gap > threshold

    @staticmethod
    def is_market_manipulation(candles: OhlcvWrapper) -> bool:
        current = candles.get_current_candle()
        return (
            StrategyUtils.has_large_wick(current)
            or StrategyUtils.is_single_candle_pump(candles)
            or StrategyUtils.has_price_gap(candles)
        )
    
    @staticmethod
    def get_market_manipulation_score(
        candles: OhlcvWrapper,
        lookback: int = 20,
        price_spike_threshold: float = 2.0,
        wick_ratio_threshold: float = 2.5,
        volume_spike_ratio: float = 2.0
    ) -> float:
        """
        Retorna um score de manipulação de mercado entre 0.0 e 1.0 com base em:
        - Tamanho anormal do candle (spike)
        - Pavio longo (sombra superior/inferior muito maior que o corpo)
        - Volume muito acima da média
        - Reversão após candle extremo

        Parâmetros:
        - candles: OhlcvWrapper
        - lookback: Nº de candles para calcular médias
        - thresholds: valores para deteção de anomalias

        Returns:
        - manip_score: float (0.0 a 1.0)
        """
        if len(candles) < lookback + 3:
            return 0.0

        recent_candles = candles.get_recent_closed(lookback)
        last_candle = candles.get_last_closed_candle()

        # Preço médio de corpos
        avg_body_size = sum(abs(c.open - c.close) for c in recent_candles) / lookback
        body = abs(last_candle.open - last_candle.close)

        # Comprimento do candle
        full_range = last_candle.high - last_candle.low
        upper_wick = last_candle.high - max(last_candle.open, last_candle.close)
        lower_wick = min(last_candle.open, last_candle.close) - last_candle.low

        # Volume
        avg_volume = sum(c.volume for c in recent_candles) / lookback
        volume_score = 1.0 if last_candle.volume > avg_volume * volume_spike_ratio else 0.0

        # Corpo anormal
        body_score = min(1.0, body / (avg_body_size * price_spike_threshold))

        # Wick ratio
        wick_ratio = max(upper_wick, lower_wick) / body if body > 0 else 0
        wick_score = 1.0 if wick_ratio > wick_ratio_threshold else 0.0

        # Reversão após candle forte
        candle_1 = candles.get_recent_closed(1)[-1]
        candle_2 = candles.get_recent_closed(2)[-2]

        reversal_score = 0.0
        if body_score > 0.8:
            if last_candle.close < candle_1.low < candle_2.low:
                # Engolfo de baixa após candle forte de alta
                reversal_score = 1.0
            elif last_candle.close > candle_1.high > candle_2.high:
                # Engolfo de alta após candle forte de queda
                reversal_score = 1.0

        # Média ponderada simples
        score = (
            0.4 * body_score +
            0.3 * wick_score +
            0.2 * volume_score +
            0.1 * reversal_score
        )

        return round(min(score, 1.0), 2)
    
    @staticmethod
    def calculate_volume_ratio(candles: OhlcvWrapper, window: int = 20) -> float:
        volumes = candles.volumes
        if len(volumes) < window + 1:
            return 1.0  # neutro

        avg_volume = np.mean(volumes[-window - 1:-1])
        current_volume = volumes[-1]

        ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        return ratio
    
    @staticmethod
    def calculate_atr_ratio(candles: OhlcvWrapper):
        indicators = Indicators(candles)
        atr = indicators.atr()[-1]
        high = candles.highs[-1]
        low = candles.lows[-1]
        range_candle = high - low
        return range_candle / atr if atr != 0 else 0
    
    @staticmethod
    def calculate_volume_penalty(candles: OhlcvWrapper) -> Tuple[float, float]:
        volumes = candles.volumes
        recent_volume = volumes[-1]
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)

        if avg_volume == 0:
            return 1.0, 1.0  # evitar divisão por zero

        volume_ratio = recent_volume / avg_volume
        volume_score = min(volume_ratio, 1.5) / 1.5  # normaliza para [0, 1]

        # penalizar abaixo de 0.7 (ajustável)
        if volume_score < 0.7:
            buy_penalty = 1 - (0.7 - volume_score)
            sell_penalty = 1 - (0.7 - volume_score)
        else:
            buy_penalty = 1.0
            sell_penalty = 1.0

        # opcional: ajustar pelo tipo de candle
        closes = candles.closes
        opens = candles.opens
        if closes[-1] > opens[-1]:  # candle bullish
            sell_penalty *= 0.95
        elif closes[-1] < opens[-1]:  # candle bearish
            buy_penalty *= 0.95

        # limitar entre [0.3, 1.0] por segurança
        buy_penalty = max(0.3, min(buy_penalty, 1.0))
        sell_penalty = max(0.3, min(sell_penalty, 1.0))

        return buy_penalty, sell_penalty
    
    
    @staticmethod
    def is_exhaustion_candle(candles: OhlcvWrapper, lookback: int = 20, threshold: float = 0.95) -> tuple[bool, bool]:
        """
        Verifica se o candle atual está em zona de exaustão:
        - Topo (para penalizar BUY)
        - Fundo (para penalizar SELL)

        Parâmetros:
        - candles: OhlcvWrapper com dados OHLCV
        - lookback: Número de candles anteriores a considerar (excluindo o atual)
        - threshold: Percentil acima/abaixo do qual se considera exaustão
        """
        if len(candles) < lookback + 1:
            return False, False

        # Últimos N candles fechados
        recent_candles = candles.get_recent_closed(lookback=lookback)

        highs = [c.high for c in recent_candles]
        lows = [c.low for c in recent_candles]

        range_high = max(highs)
        range_low = min(lows)

        current_close = candles.get_current_candle().close

        # Evita divisão por zero se todos os candles forem flat
        if range_high == range_low:
            return False, False

        relative_position = (current_close - range_low) / (range_high - range_low)

        is_top_exhaustion = relative_position >= threshold
        is_bottom_exhaustion = relative_position <= (1 - threshold)

        return is_top_exhaustion, is_bottom_exhaustion
    
    @staticmethod
    def is_abnormal_volume(candles: OhlcvWrapper, lookback: int = 20, threshold: float = 2.0) -> bool:
        if len(candles) < lookback + 1:
            return False

        recent = candles.get_recent_closed(lookback)
        volumes = [c.volume for c in recent]

        avg_volume = sum(volumes) / len(volumes)
        current_volume = candles.get_current_candle().volume

        return current_volume > (avg_volume * threshold)
    
    @staticmethod
    def linear_slope(values: List[float]) -> float:
        """
        Calcula a inclinação da reta (slope) que melhor ajusta os pontos fornecidos.
        Usa regressão linear simples.
        
        :param values: lista de valores (ex: preços ou bandas)
        :return: coeficiente angular da regressão linear (slope)
        """
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        y = np.array(values)
        A = np.vstack([x, np.ones(len(x))]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0] # type: ignore
        return slope
    
    @staticmethod
    def get_volatility_moves(candles: OhlcvWrapper, lookback: int = 50) -> list[float]:
        return [
            abs(c.high - c.low)
            for c in candles.get_recent_closed(lookback)
        ]
    
    @staticmethod
    def get_early_signal(ohlcv: OhlcvWrapper) -> Signal | None:
        indicators = Indicators(ohlcv)
        rsi = indicators.rsi()
        stoch_k, stoch_d = indicators.stochastic()
        volume = ohlcv.volumes
        close = ohlcv.closes
        high = ohlcv.highs
        low = ohlcv.lows

        i = -1

        avg_volume = np.mean(volume[-20:])
        signals = []

        # Early BUY
        if (
            rsi[i] > 50 and rsi[i-1] < rsi[i] and
            stoch_k[i] > stoch_d[i] and
            volume[i] > avg_volume * 1.05 and
            close[i] > high[i-1] * 0.995  # breakout com folga
        ):
            signals.append(Signal.BUY)

        # Early SELL
        if (
            rsi[i] < 50 and rsi[i-1] > rsi[i] and
            stoch_k[i] < stoch_d[i] and
            volume[i] > avg_volume * 1.05 and
            close[i] < low[i-1] * 1.005  # breakdown com tolerância
        ):
            signals.append(Signal.SELL)

        return signals[0] if signals else None
    
    @staticmethod
    def is_late_entry(ohlcv: OhlcvWrapper, signal: Signal) -> bool:
        indicators = Indicators(ohlcv)
        rsi = indicators.rsi()

        is_top, is_bottom = StrategyUtils.is_exhaustion_candle(ohlcv)

        if signal == Signal.BUY:
            return rsi[-1] > 70 and is_top
        elif signal == Signal.SELL:
            return rsi[-1] < 30 and is_bottom
        return False
    

    @staticmethod
    def is_strong_trend_up(ohlcv: OhlcvWrapper) -> bool:
        indicators = Indicators(ohlcv)

        ema_fast = indicators.ema(10)
        ema_slow = indicators.ema(50)
        adx = indicators.adx()
        i = -1
        return ema_fast[i] > ema_slow[i] and adx[i] > 25

    @staticmethod
    def is_strong_trend_down(ohlcv: OhlcvWrapper) -> bool:
        indicators = Indicators(ohlcv)

        ema_fast = indicators.ema(10)
        ema_slow = indicators.ema(50)
        adx = indicators.adx()
        i = -1
        return ema_fast[i] < ema_slow[i] and adx[i] > 25
    
    @staticmethod
    def is_pullback_in_uptrend(ohlcv: OhlcvWrapper) -> bool:
        indicators = Indicators(ohlcv)

        closes = ohlcv.closes
        lows = ohlcv.lows
        ema_fast = indicators.ema(10)
        ema_slow = indicators.ema(50)
        i = -1
        return (
            ema_fast[i] > ema_slow[i] and
            closes[i] > ema_slow[i] and
            lows[i] < ema_fast[i] * 0.985
        )
    
    @staticmethod
    def is_pullback_in_downtrend(ohlcv: OhlcvWrapper) -> bool:
        indicators = Indicators(ohlcv)

        closes = ohlcv.closes
        highs = ohlcv.highs
        ema_fast = indicators.ema(10)
        ema_slow = indicators.ema(50)
        i = -1
        return (
            ema_fast[i] < ema_slow[i] and
            closes[i] < ema_slow[i] and
            highs[i] > ema_fast[i] * 1.015
        )
    
    @staticmethod
    def equal_weights(keys: list[str]) -> dict[str, float]:
        count = len(keys)
        value = 1 / count
        return {k: value for k in keys}
    
    @staticmethod
    def is_weak_momentum(ohlcv: OhlcvWrapper, idx: int, ema_fast=9, ema_slow=21, threshold=0.0015) -> bool:
        """
        Retorna True se a diferença entre EMA rápida e lenta for pequena
        (momentum fraco).
        """
        close = ohlcv.get_last_closed_candle().close
        ema_fast_val = Indicators(ohlcv).ema(ema_fast)[idx]
        ema_slow_val = Indicators(ohlcv).ema(ema_slow)[idx]
        diff = abs(ema_fast_val - ema_slow_val) / close
        return diff < threshold

    @staticmethod
    def is_stoch_overbought(ohlcv: OhlcvWrapper, idx: int, k_period=14, d_period=3, overbought=80) -> bool:
        """
        Retorna True se o estocástico estiver sobrecomprado.
        """
        
        k, d = Indicators(ohlcv).stochastic()
        return k[idx] > overbought and d[idx] > overbought

    @staticmethod
    def is_stoch_oversold(ohlcv: OhlcvWrapper, idx: int, k_period=14, d_period=3, oversold=20) -> bool:
        """
        Retorna True se o estocástico estiver sobrevendido.
        """
        k, d = Indicators(ohlcv).stochastic()
        return k[idx] < oversold and d[idx] < oversold
    
    @staticmethod
    def is_rsi_overbought(ohlcv: OhlcvWrapper, rsi_threshold: float = 70) -> bool:
        """
        Verifica se o RSI do último candle fechado está em sobrecompra.
        
        Args:
            ohlcv: Wrapper que contém o RSI e os candles.
            rsi_threshold: Valor limite para considerar sobrecompra (default 70).
        
        Returns:
            True se RSI estiver acima do limite de sobrecompra, False caso contrário.
        """
        rsi_series = Indicators(ohlcv).rsi()  # supõe que tens método para obter indicador
        if rsi_series is None or len(rsi_series) == 0:
            return False
        last_rsi = rsi_series[-1]
        return last_rsi >= rsi_threshold
    
    @staticmethod
    def is_rsi_oversold(ohlcv: OhlcvWrapper, rsi_threshold: float = 30) -> bool:
        """
        Verifica se o RSI do último candle fechado está em sobrevenda.
        
        Args:
            ohlcv: Wrapper que contém o RSI e os candles.
            rsi_threshold: Valor limite para considerar sobrevenda (default 30).
        
        Returns:
            True se RSI estiver abaixo do limite de sobrevenda, False caso contrário.
        """
        rsi_series = Indicators(ohlcv).rsi() # supõe que tens método para obter indicador
        if rsi_series is None or len(rsi_series) == 0:
            return False
        last_rsi = rsi_series[-1]
        return last_rsi <= rsi_threshold
    
    @staticmethod
    def _calc_rsi(closes, period:int=14) -> float:
        deltas = np.diff(closes)
        ups = deltas.clip(min=0)
        downs = -deltas.clip(max=0)
        avg_gain = np.mean(ups[-period:])
        avg_loss = np.mean(downs[-period:])
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))
    
    
    @staticmethod
    def rsi_signal(ohlcv: OhlcvWrapper, period: int = 14) -> Signal:
        """
        Retorna 'buy' se RSI < oversold, 'sell' se RSI > overbought, senão None.
        """
        closes = ohlcv.closes
        if len(closes) < period + 1:
            return Signal.HOLD

        deltas = np.diff(closes)
        ups = deltas.clip(min=0)
        downs = -deltas.clip(max=0)

        avg_gain = np.mean(ups[-period:])
        avg_loss = np.mean(downs[-period:])

        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        # Inclinação do RSI
        prev_rsi = StrategyUtils._calc_rsi(closes[:-1], period)
        slope = rsi - prev_rsi

        # Lógica combinada
        if rsi >= 50 and slope >= 0:
            return Signal.BUY
        elif rsi < 50 and slope <= 0:
            return Signal.SELL
        elif slope > 0:
            return Signal.BUY
        else:
            return Signal.SELL

    @staticmethod
    def stochastic_signal(ohlcv: OhlcvWrapper, period=14) -> Signal:
        highs = ohlcv.highs
        lows = ohlcv.lows
        closes = ohlcv.closes

        if len(closes) < period + 1:
            return Signal.HOLD

        # Calcular %K atual
        highest_high = max(highs[-period:])
        lowest_low = min(lows[-period:])
        k = 100 * (closes[-1] - lowest_low) / (highest_high - lowest_low)

        # Calcular %K anterior
        prev_highest_high = max(highs[-period-1:-1])
        prev_lowest_low = min(lows[-period-1:-1])
        prev_k = 100 * (closes[-2] - prev_lowest_low) / (prev_highest_high - prev_lowest_low)

        slope = k - prev_k

        
        # Lógica combinada
        if k >= 50 and slope >= 0:
            return Signal.BUY
        elif k < 50 and slope <= 0:
            return Signal.SELL
        elif slope > 0:
            return Signal.BUY
        else:
            return Signal.SELL
        
    @staticmethod
    def ema_signal(ohlcv: OhlcvWrapper, fast_period:int=21, slow_period:int=50) -> Signal:
        closes = ohlcv.closes

        if len(closes) < slow_period:
            return Signal.HOLD

        # Calcular EMAs
        ema_fast = Indicators(ohlcv).ema()
        ema_slow = Indicators(ohlcv).ema(50)

        # Inclinação da EMA rápida
        slope = ema_fast[-1] - ema_fast[-2]

        # Lógica de decisão
        if ema_fast[-1] > ema_slow[-1] and slope >= 0:
            return Signal.BUY
        elif ema_fast[-1] < ema_slow[-1] and slope <= 0:
            return Signal.SELL
        elif slope > 0:
            return Signal.BUY
        else:
            return Signal.SELL
        
    @staticmethod
    def ema_signal_strict(
        ohlcv: OhlcvWrapper,
        fast_period: int = 21,
        slow_period: int = 50,
        min_slope: float = 0.0,
        min_ema_distance: float = 0.001,  # 0.1% de distância mínima
        lookback_support_resistance: int = 20
    ) -> Signal:
        closes = ohlcv.closes

        if len(closes) < slow_period:
            return Signal.HOLD

        # Calcular EMAs
        indicators = Indicators(ohlcv)
        ema_fast = indicators.ema(fast_period)
        ema_slow = indicators.ema(slow_period)

        # Inclinação da EMA rápida
        slope = ema_fast[-1] - ema_fast[-2]

        # Distância relativa entre EMAs
        ema_distance = abs(ema_fast[-1] - ema_slow[-1]) / ema_slow[-1]

        # Preço atual
        price = closes[-1]

        # Suporte e resistência recentes
        highs = ohlcv.highs[-lookback_support_resistance:]
        lows = ohlcv.lows[-lookback_support_resistance:]
        recent_support = min(lows)
        recent_resistance = max(highs)

        # PSAR (opcional)
        psar_values = Indicators(ohlcv).psar()
        psar_last = psar_values[-1]

        # --- Lógica de Compra ---
        if (
            ema_fast[-1] > ema_slow[-1]  # tendência de alta
            and slope > min_slope        # inclinação positiva
            and ema_distance > min_ema_distance
            and price > ema_slow[-1]     # preço acima da EMA lenta
            and price > recent_support   # preço não está colado ao suporte
            and psar_last < price        # PSAR de compra
        ):
            return Signal.BUY

        # --- Lógica de Venda ---
        elif (
            ema_fast[-1] < ema_slow[-1]  # tendência de baixa
            and slope < -min_slope       # inclinação negativa
            and ema_distance > min_ema_distance
            and price < ema_slow[-1]
            and price < recent_resistance
            and psar_last > price
        ):
            return Signal.SELL

        return Signal.HOLD
    
    @staticmethod
    def candle_body_signal(ohlcv:  OhlcvWrapper) -> Signal:
        
        idx = len(ohlcv.closes) - 1
        if idx < 1:
            return Signal.HOLD  # Não há candle anterior suficiente

        open_curr, close_curr = ohlcv.opens[idx], ohlcv.closes[idx]
        open_prev, close_prev = ohlcv.opens[idx-1], ohlcv.closes[idx-1]

        body_curr = abs(close_curr - open_curr)
        body_prev = abs(close_prev - open_prev)

        if close_curr > open_curr and body_curr > body_prev:
            return Signal.BUY
        elif close_curr < open_curr and body_curr > body_prev:
            return Signal.SELL
        else:
            return Signal.HOLD

    @staticmethod    
    def is_market_sideways(ohlcv: OhlcvWrapper, lookback:int=14, atr_threshold_pct:float=0.003, range_threshold_pct:float=0.003) -> bool:
        """
        Detecta mercado lateral com base em ATR e range médio.

        Args:
            ohlcv (OhlcvWrapper): dados OHLCV com arrays .highs, .lows, .closes
            lookback (int): número de candles para cálculo da média
            atr_threshold_pct (float): limite percentual para ATR (ex: 0.003 = 0.3%)
            range_threshold_pct (float): limite percentual para range (high-low) médio

        Returns:
            bool: True se mercado lateral, False caso contrário
        """
        highs = ohlcv.highs
        lows = ohlcv.lows
        closes = ohlcv.closes

        if len(highs) < lookback + 1:
            return False  # dados insuficientes

        # Calcula ATR simplificado (True Range médio)
        tr_values = []
        for i in range(-lookback, 0):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1]),
            )
            tr_values.append(tr)
        atr = sum(tr_values) / lookback

        # Calcula range médio (high - low)
        range_values = [highs[i] - lows[i] for i in range(-lookback, 0)]
        avg_range = sum(range_values) / lookback

        # Preço médio para referência percentual (podes usar close médio)
        avg_price = sum(closes[-lookback:]) / lookback

        atr_ratio = atr / avg_price
        range_ratio = avg_range / avg_price

        # Debug prints (podes tirar depois)
        print(f"ATR ratio: {atr_ratio:.5f}, Range ratio: {range_ratio:.5f}")

        return (atr_ratio < atr_threshold_pct) and (range_ratio < range_threshold_pct)
    
    @staticmethod
    def is_market_sideways_strict(ohlcv: OhlcvWrapper, lookback=20, atr_threshold_pct=0.003, max_trend_pct=0.01, max_ema_slope=0.0005):
        closes = ohlcv.closes
        highs = ohlcv.highs
        lows = ohlcv.lows

        if len(closes) < lookback + 1:
            return False

        # 1. ATR relativo
        atr = Indicators(ohlcv).atr()
        atr_value = atr[-1]
        atr_rel = atr_value / closes[-1]

        # 2. Variação percentual do fechamento inicial para final do lookback
        price_start = closes[-lookback - 1]
        price_end = closes[-1]
        trend_pct = abs(price_end - price_start) / price_start

        # 3. Range total percentual no lookback (max - min) / preço atual
        period_max = max(highs[-lookback:])
        period_min = min(lows[-lookback:])
        range_pct = (period_max - period_min) / closes[-1]

        # 4. Inclinação EMA21 (aprox. diferença percentual entre última e penúltima EMA)
        ema21 = Indicators(ohlcv).ema(21)
        if len(ema21) < 2:
            return False
        ema_slope = abs(ema21[-1] - ema21[-2]) / ema21[-2]

        print(f"ATR rel: {atr_rel:.5f}, Trend pct: {trend_pct:.5f}, Range pct: {range_pct:.5f}, EMA slope: {ema_slope:.5f}")

        # Verifica critérios estritos para lateralidade
        if atr_rel < atr_threshold_pct \
        and trend_pct < max_trend_pct \
        and range_pct < max_trend_pct \
        and ema_slope < max_ema_slope:
            return True
        else:
            return False

    @staticmethod   
    def ratio_support_resistence(ohlcv: OhlcvWrapper) -> float:
        resistance, support = StrategyUtils.detect_support_resistance(ohlcv, lookback=20, tolerance_pct=0.02)
        close_price = ohlcv.get_last_closed_candle().close

        channel_height = resistance - support

        channel_position = 0
        if channel_height > 0:
            # evitar divisão por zero, devolve penalização neutra (0) ou outro valor que faças sentido
            channel_position = (close_price - support) / channel_height
        return channel_position    
    
    @staticmethod
    def calculate_sl_tp_dynamic_channel(
        ohlcv: OhlcvWrapper,
        price_entry: float,
        side: Signal,
        risk_reward_ratio: float = 2.0,
        band_multiplier: float = 2.0,
        tp_band_limit: float = 0.95,  # máximo TP a 95% do canal para evitar TP no limite exato
        sl_band_buffer: float = 0.005  # buffer 0.5% para SL abaixo/ acima da banda para BUY/SELL
    ) -> tuple[float, float]:
        """
        Calcula SL e TP baseados nas bandas do canal para limitar distâncias.
        SL fica próximo da banda inferior (BUY) ou superior (SELL) com buffer.
        TP fica limitado para nunca ultrapassar 95% do canal, respeitando risk_reward_ratio.

        Args:
            ohlcv (OhlcvWrapper): Dados OHLCV.
            price_entry (float): Preço de entrada.
            side (Signal): Signal.BUY ou Signal.SELL.
            risk_reward_ratio (float): Razão risco/recompensa (default=2.0).
            band_multiplier (float): Multiplicador para calcular bandas (default=2.0).
            tp_band_limit (float): Limite máximo para TP dentro do canal (default=0.95).
            sl_band_buffer (float): Buffer percentual para SL abaixo/ acima da banda (default=0.5%).

        Returns:
            tuple: (sl_price, tp_price)
        """

        atr = Indicators(ohlcv).atr()
        window = 4
        atr_ema = StrategyUtils.ema(atr[-window:], window)
        price = ohlcv.get_last_closed_candle().close

        base_multiplier = 3.0
        min_buffer = 0.005

        band_multiplier = max(base_multiplier * (atr_ema / price), min_buffer)

        upper_band, lower_band = StrategyUtils.calculate_bands(ohlcv, multiplier=band_multiplier)
        if not upper_band or not lower_band or len(upper_band) < 1:
            # fallback simples caso as bandas não estejam disponíveis
            sl_price = price_entry * (0.98 if side == Signal.BUY else 1.02)
            tp_price = price_entry * (1.04 if side == Signal.BUY else 0.96)
            return sl_price, tp_price

        channel_high = upper_band[-1]
        channel_low = lower_band[-1]

        channel_high, channel_low = StrategyUtils.detect_support_resistance(ohlcv, 20)

        print("RESISTANCE E SUPPORT", channel_high, channel_low)

        if side == Signal.BUY:
            # SL um pouco abaixo da banda inferior, com buffer
            sl_price = channel_low * (1 - sl_band_buffer)
            sl_price = min(sl_price, price_entry * 0.995)  # nunca acima do entry (buffer 0.5%)

            risk_distance = price_entry - sl_price

            print("RISK AQUII", risk_distance)
            tp_target = price_entry + risk_distance * risk_reward_ratio

            # Limitar TP para não ultrapassar 95% da distância até o topo do canal
            max_tp = price_entry + (channel_high - price_entry) * tp_band_limit
            tp_price = min(tp_target, max_tp)

            print("RISK AQUII 2", tp_target, max_tp, price_entry)

        elif side == Signal.SELL:
            # SL um pouco acima da banda superior, com buffer
            sl_price = channel_high * (1 + sl_band_buffer)
            sl_price = max(sl_price, price_entry * 1.005)  # nunca abaixo do entry (buffer 0.5%)

            risk_distance = sl_price - price_entry
            tp_target = price_entry - risk_distance * risk_reward_ratio

            # Limitar TP para não ultrapassar 95% da distância até o fundo do canal
            min_tp = price_entry - (price_entry - channel_low) * tp_band_limit
            tp_price = max(tp_target, min_tp)

        else:
            raise ValueError("side deve ser Signal.BUY ou Signal.SELL")

        return sl_price, tp_price
    
    @staticmethod
    def calculate_sl_tp_simple(
        ohlcv: OhlcvWrapper,
        price_entry: float,
        side: Signal,
        atr_mult: float = 1.5,
        lookback_support_resistance: int = 3
    ) -> tuple[float, float]:

        highs = ohlcv.highs
        lows = ohlcv.lows
        atr = Indicators(ohlcv).atr(period=14)
        last_candle_range = highs[-1] - lows[-1]

        # tamanho base para TP (ATR ou candle range)
        base_range = last_candle_range if last_candle_range > 0 else atr[-1]

        if StrategyUtils.trend_strength_signal(ohlcv) != side:
            atr_mult *= 0.5  # reduz TP pela metade se a tendência estiver fraca

        if side == Signal.BUY:
            support = min(lows[-lookback_support_resistance:])
            sl_price = support * 0.998
            tp_price = price_entry + base_range * atr_mult

        elif side == Signal.SELL:
            resistance = max(highs[-lookback_support_resistance:])
            sl_price = resistance * 1.002
            tp_price = price_entry - base_range * atr_mult

        else:
            raise ValueError("side deve ser Signal.BUY ou Signal.SELL")

        return sl_price, tp_price
    
    @staticmethod
    def ema(values, window):
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        return np.convolve(values, weights, mode='valid')[-1]
    
    @staticmethod
    def channel_position_normalized(ohlcv: OhlcvWrapper, multiplier: float) -> float:
        """
        Retorna a posição do preço dentro do canal normalizada entre 0 e 1.
        0 = preço na base do canal
        1 = preço no topo do canal
        """
        upper_band, lower_band = StrategyUtils.calculate_bands(ohlcv, multiplier=multiplier)
        close = ohlcv.get_last_closed_candle().close

        if not upper_band or not lower_band or len(upper_band) < 1:
            return 0.0

        top = upper_band[-1]
        bottom = lower_band[-1]
        band_range = top - bottom
        if band_range == 0:
            return 0.0
        print("BANDS", top, bottom)
        # Distância relativa dentro do canal
        position = (close - bottom) / band_range

        # Limitar a [0, 1]
        return max(min(position, 1.0), 0.0)
    
    @staticmethod
    def trend_strength_signal(ohlcv: OhlcvWrapper, lookback: int = 3, adx_threshold: float = 25) -> Signal:
        """
        Retorna um sinal consolidado da tendência:
        - Signal.BUY -> tendência de alta ainda forte
        - Signal.SELL -> tendência de baixa ainda forte
        - Signal.HOLD -> tendência está enfraquecendo ou indecisa
        """
        closes = ohlcv.closes
        highs = ohlcv.highs
        lows = ohlcv.lows
        opens = ohlcv.opens
        indicators = Indicators(ohlcv)
        
        ema_fast = indicators.ema(10)
        ema_slow = indicators.ema(50)
        adx = indicators.adx()
        
        n = len(closes)
        if n < lookback + 1 or len(adx) < lookback + 1:
            return Signal.HOLD  # dados insuficientes
        
        # Contadores de candles enfraquecidos
        weakening_count = 0
        
        for i in range(n - lookback, n):
            candle_body = abs(closes[i] - opens[i])
            candle_range = highs[i] - lows[i]
            if candle_range == 0:
                continue
            body_ratio = candle_body / candle_range
            
            weak_adx = adx[i] < adx_threshold or adx[i] < adx[i-1]
            small_candle = body_ratio < 0.4
            trend_conflict = (ema_fast[i] < ema_slow[i] and closes[i] > ema_fast[i]) or \
                            (ema_fast[i] > ema_slow[i] and closes[i] < ema_fast[i])
            
            if sum([weak_adx, small_candle, trend_conflict]) >= 2:
                weakening_count += 1

        # Se a maioria das velas no lookback está enfraquecendo, HOLD
        if weakening_count >= (lookback // 2) + 1:
            return Signal.HOLD
        
        # Senão, retorna a tendência dominante
        if ema_fast[-1] > ema_slow[-1]:
            return Signal.BUY
        elif ema_fast[-1] < ema_slow[-1]:
            return Signal.SELL
        else:
            return Signal.HOLD

       
        
        
        

        
        