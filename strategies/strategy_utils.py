import logging

from commons.enums.mode_enum import ModeEnum
from commons.enums.signal_enum import Signal
from commons.models.ohlcv_type import Ohlcv
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
    def detect_support_resistance(
        candles: OhlcvWrapper,
        lookback: int = 50,
        tolerance_pct: float = 0.005
    ) -> tuple[float | None, float | None]:
        """
        Detecta níveis de suporte e resistência baseados em máximos/mínimos locais.

        Retorna:
        - resistência (float) se houver
        - suporte (float) se houver
        """
        if len(candles) < lookback:
            return None, None

        highs = [c.high for c in candles.get_recent_closed(lookback)]
        lows = [c.low for c in candles.get_recent_closed(lookback)]

        resistance = max(highs)
        support = min(lows)

        return resistance, support

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
    def detect_lateral_market(ohlcv: OhlcvWrapper, symbol,  adx_threshold):
        indicators = Indicators(ohlcv)
        adx = indicators.adx()
        adx_now = adx[-1]
        lateral_market = adx_now < adx_threshold
        logging.info(f"{symbol} - ADX: {adx_now:.2f} → Lateral: {lateral_market}")
        return lateral_market
    
    @staticmethod
    def trend_signal_with_adx(ohlcv: OhlcvWrapper, symbol: (str | None), adx_threshold: float, ema_now, ema_prev):
        if StrategyUtils.detect_lateral_market(ohlcv, symbol, adx_threshold):
            if ema_now > ema_prev:
                return 1  # buy
            elif ema_now < ema_prev:
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
    def check_price_action_signals(ohlcv: OhlcvWrapper):
        closes = ohlcv.closes
        opens = ohlcv.opens
        open_now = opens[-1]
        close_now = closes[-1]


       # Prioridade 1: Breakout candle (mais forte)
        if StrategyUtils.is_breakout_candle(ohlcv, -1):
            return "buy"  # ou True, se quiseres só booleano

        # Prioridade 2: Setup 123
        buy_123, sell_123 = StrategyUtils.detect_setup_123(ohlcv)
        if buy_123:
            return "buy"
        if sell_123:
            return "sell"

        # Prioridade 3: Cor da vela atual
        if close_now > open_now:
            return "buy"
        elif close_now < open_now:
            return "sell"

        # Se nenhuma condição for satisfeita
        return None
    
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

        
        