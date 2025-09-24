from commons.enums.signal_enum import Signal
from commons.models.ohlcv_type_dclass import Ohlcv
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper


class PriceActionUtils:

    @staticmethod    
    def detect_reversal_pattern(ohlcv: OhlcvWrapper, rsi_overbought: float = 70, rsi_oversold: float = 30) -> tuple[bool, bool]:
        """
        Detecta reversão de topo (baixa) ou de fundo (alta) com base em padrão de candle + RSI.
        Retorna (is_bearish_reversal, is_bullish_reversal).
        """
        indicators = IndicatorsUtils(ohlcv)

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
            is_bearish = PriceActionUtils.is_bearish_engulfing(prev, curr)

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
    def is_flat_candle(ohlcv: OhlcvWrapper):
        candle = ohlcv.get_current_candle()
        return candle.open == candle.close and candle.high == candle.low
    
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
        if PriceActionUtils.is_breakout_candle(ohlcv, -1):
            return Signal.BUY  # ou True, se quiseres só booleano

        # Prioridade 2: Setup 123
        buy_123, sell_123 = PriceActionUtils.detect_setup_123(ohlcv)
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
    def is_hammer(ohlcv:  Ohlcv) -> bool:
        open = ohlcv.open
        close = ohlcv.close
        high = ohlcv.high
        low = ohlcv.low

        body = abs(close - open)
        candle_range = high - low
        lower_wick = min(open, close) - low
        upper_wick = high - max(open, close)
        return body < candle_range * 0.3 and lower_wick > body * 2 and upper_wick < body

    @staticmethod  
    def is_shooting_star(ohlcv:  Ohlcv) -> bool:
        open = ohlcv.open
        close = ohlcv.close
        high = ohlcv.high
        low = ohlcv.low

        body = abs(close - open)
        candle_range = high - low
        upper_wick = high - max(open, close)
        lower_wick = min(open, close) - low
        return body < candle_range * 0.3 and upper_wick > body * 2 and lower_wick < body

    @staticmethod  
    def is_pinbar_bullish(ohlcv:  Ohlcv) -> bool:
        open = ohlcv.open
        close = ohlcv.close
        high = ohlcv.high
        low = ohlcv.low

        body = abs(close - open)
        lower_wick = min(open, close) - low
        upper_wick = high - max(open, close)
        return lower_wick > body * 2 and upper_wick < body

    @staticmethod  
    def is_pinbar_bearish(ohlcv:  Ohlcv) -> bool: 
        open = ohlcv.open
        close = ohlcv.close
        high = ohlcv.high
        low = ohlcv.low

        body = abs(close - open)
        upper_wick = high - max(open, close)
        lower_wick = min(open, close) - low
        return upper_wick > body * 2 and lower_wick < body
    
    @staticmethod
    def is_bearish_engulfing_below_band(prev: Ohlcv, curr: Ohlcv, band_upper: float, margin=0.001):
        """Engolfo de baixa válido apenas se candle abrir e fechar abaixo da banda superior."""
        if prev is None or curr is None:
            return False
        
        engulfing = (
            prev.close > prev.open and       # candle anterior bullish
            curr.close < curr.open and       # candle atual bearish
            curr.open > prev.close and       # abertura acima do fecho anterior
            curr.close < prev.open           # fecho abaixo da abertura anterior
        )
        
        # Só valida se **tanto abertura quanto fecho** estiverem abaixo da banda (com margem)
        below_band = curr.close < band_upper * (1 - margin) and curr.open < band_upper * (1 - margin)
        
        return engulfing and below_band
    
    @staticmethod
    def is_bullish_engulfing_above_band(prev: Ohlcv, curr: Ohlcv, band_lower: float, margin=0.001):
        """Engolfo de alta válido apenas se candle abrir e fechar acima da banda inferior."""
        if prev is None or curr is None:
            return False
        
        engulfing = (
            prev.close < prev.open and       # candle anterior bearish
            curr.close > curr.open and       # candle atual bullish
            curr.open < prev.close and       # abertura abaixo do fecho anterior
            curr.close > prev.open           # fecho acima da abertura anterior
        )
        
        # Só valida se **tanto abertura quanto fecho** estiverem acima da banda inferior (com margem)
        above_band = curr.close > band_lower * (1 + margin) and curr.open > band_lower * (1 + margin)
        
        return engulfing and above_band
    
    
    @staticmethod
    def is_bullish_rejection(prev: Ohlcv, curr: Ohlcv, min_wick_ratio=1.5):
        """
        Detecta reversão de sell:
        - prev: candle vermelho grande
        - curr: candle verde pequeno com pavio inferior longo
        """
        body_curr = abs(curr.close - curr.open)
        lower_wick = curr.open - curr.low
        if body_curr == 0:
            return False
        wick_ratio = lower_wick / body_curr
        prev_body = abs(prev.close - prev.open)
        
        if prev.close < prev.open and curr.close > curr.open and wick_ratio >= min_wick_ratio:
            return True
        return False
    