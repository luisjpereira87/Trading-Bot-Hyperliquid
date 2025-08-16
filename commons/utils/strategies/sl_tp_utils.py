import numpy as np

from commons.enums.mode_enum import ModeEnum
from commons.enums.signal_enum import Signal
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from commons.utils.strategies.support_resistance_utils import \
    SupportResistanceUtils
from commons.utils.strategies.trend_utils import TrendUtils


class SlTpUtils:
    
    
    @staticmethod
    def calculate_sl_tp_simple(
        ohlcv: OhlcvWrapper,
        entry_price: float,
        side: Signal,
        atr_mult: float = 1.5,
        min_pct: float = 0.002,
        lookback_support_resistance: int = 5
    ) -> tuple[float, float]:

        highs = ohlcv.highs
        lows = ohlcv.lows
        atr = IndicatorsUtils(ohlcv).atr(period=14)
        last_candle_range = highs[-1] - lows[-1]

        # tamanho base para TP (ATR ou candle range)
        base_range = last_candle_range if last_candle_range > 0 else atr[-1]

        if TrendUtils.trend_strength_signal(ohlcv) != side:
            atr_mult *= 0.5  # reduz TP pela metade se a tendência estiver fraca
        else:
            atr_mult *= 1.5

        if side == Signal.BUY:
            support = min(lows[-lookback_support_resistance:])
            sl_price = support * 0.997
            tp_price = entry_price + base_range * atr_mult

            if (tp_price - entry_price) / entry_price < min_pct or tp_price <= entry_price:
                raise ValueError(f"TP demasiado próximo do preço de entrada: {tp_price} vs {entry_price}")

        elif side == Signal.SELL:
            resistance = max(highs[-lookback_support_resistance:])
            sl_price = resistance * 1.003
            tp_price = entry_price - base_range * atr_mult

            if (entry_price - tp_price) / entry_price < min_pct or tp_price >= entry_price:
                raise ValueError(f"TP demasiado próximo do preço de entrada: {tp_price} vs {entry_price}")

        else:
            raise ValueError("side deve ser Signal.BUY ou Signal.SELL")

        return sl_price, tp_price
    

    @staticmethod
    def calculate_sl_tp(ohlcv: OhlcvWrapper, price_ref: float, side: Signal, mode: ModeEnum, sl_multiplier_aggressive: float, tp_multiplier_aggressive: float, sl_multiplier_conservative: float, tp_multiplier_conservative: float):
        indicators = IndicatorsUtils(ohlcv)
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

        indicators = IndicatorsUtils(ohlcv)
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

        psar_values = IndicatorsUtils(ohlcv).psar(psar_acceleration, psar_maximum)
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

        resistances, supports = SupportResistanceUtils.detect_multiple_support_resistance(ohlcv, lookback=50, tolerance_pct=tolerance_pct)

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