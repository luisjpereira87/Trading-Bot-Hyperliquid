import numpy as np

from commons.enums.mode_enum import ModeEnum
from commons.enums.signal_enum import Signal
from commons.models.signal_result_dclass import SignalResult
from commons.models.strategy_base_dclass import StrategyBase
from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from trading_bot.exchange_client import ExchangeClient


class CrossEmaStrategy(StrategyBase):

    def __init__(self, exchange: ExchangeClient):
        super().__init__()
    
        self.exchange = exchange
        self.ohlcv: OhlcvWrapper
        self.ohlcv_higher: OhlcvWrapper
        self.symbol = None


    def required_init(self, ohlcv: OhlcvWrapper, ohlcv_higher: OhlcvWrapper, symbol: str, price_ref: float):
        self.ohlcv = ohlcv
        self.ohlcv_higher = ohlcv_higher
        self.symbol = symbol
        self.price_ref = price_ref
        self.indicators = IndicatorsUtils(self.ohlcv)
    
    def set_params(self, params: StrategyParams):
        pass
  
    def set_candles(self, ohlcv):
        self.ohlcv = ohlcv

    def set_higher_timeframe_candles(self, ohlcv_higher: OhlcvWrapper):
        self.ohlcv_higher = ohlcv_higher

    async def get_signal(self) -> SignalResult:

        last_closed_candle = self.ohlcv.get_last_closed_candle()
        supertrend, trend, upperband, lowerband, supertrend_smooth,_,_ = self.indicators.supertrend()
        ema_cross_signal = CrossEmaStrategy.build_signal(self.indicators, self.ohlcv)

        signal = ema_cross_signal[-2]
        close = last_closed_candle.close
        closes = self.ohlcv.closes

        lookback = 10
        if signal == Signal.BUY:
            sl = min(lowerband[-lookback:])  # SL no ponto mais baixo da banda
            tp = max(upperband[-lookback:]) + (max(upperband[-lookback:]) - sl) * 0.5

        elif signal == Signal.SELL:
            #sl = upperband[-2]
            sl = max(upperband[-lookback:])  # SL no ponto mais alto da banda
            tp = min(lowerband[-lookback:]) - (sl - min(lowerband[-lookback:])) * 0.5

        else:
            return SignalResult(signal, None, None)
        
        # valida relação risco/benefício
        risk = abs(close - sl)
        reward = abs(tp - close)

        if (signal == Signal.BUY or signal == Signal.SELL) and  reward < risk:
            # ajusta SL e TP dinamicamente
            sl_adjusted = close - (risk * 0.5) if signal == Signal.BUY else close + (risk * 0.5)
            tp_adjusted = close + (reward * 1.5) if signal == Signal.BUY else close - (reward * 1.5)

            return SignalResult(signal, sl, tp_adjusted)

        return SignalResult(signal, sl, tp)
    
    @staticmethod
    def build_signal(indicators: IndicatorsUtils, ohlcv: OhlcvWrapper, trailing_n = 3):
        closes = ohlcv.closes
        n = len(closes)

        trend_signal = [Signal.HOLD] * n

        last_signal = None
        active_trend = None
        active_fast_trend = None
        active_macro_trend = None

        ema200 = indicators.ema(200)
        ema50 = indicators.ema(50)
        ema21 = indicators.ema(21)
        ema9 = indicators.ema(9)
        psar = indicators.psar()
        atr = indicators.atr()
        lateral = indicators.detect_low_volatility()
        rsi = indicators.rsi()
        stoch_k, stoch_d = indicators.stochastic()

        entry_price = 0
        profits = []
        min_profit_threshold = 0.001
        cross_index = None
        current_profit_pct = None
        cross_age = 0
        cross_age_fast = 0
        cross_index_fast = None
        cross_age_macro = 0
        cross_index_macro = None

        for i in range(1, n):

            current_signal = None

            # Calcula lucro atual
            if last_signal == Signal.BUY:
                current_profit_pct = (closes[i] - entry_price) / entry_price
                profits.append(current_profit_pct)
                
            elif last_signal == Signal.SELL:  # SELL
                current_profit_pct = (entry_price - closes[i]) / entry_price
                profits.append(current_profit_pct)
            
            # ---------------------------------------------------------
            # → NOVA CHAMADA AO MÉTODO DE EXIT LOGIC
            # ---------------------------------------------------------
            current_signal = CrossEmaStrategy.check_exit_signal(
                last_signal=last_signal,
                profits=profits,
                current_profit_pct=current_profit_pct,
                psar_value=psar[i],
                close_value=closes[i],
                trailing_n=trailing_n,
                min_profit_threshold=min_profit_threshold
            )
            
            # --- Detecção de tendência via EMA ---
            active_trend, cross_index, cross_age, mid_signal = CrossEmaStrategy.analyze_ema_trend(
                indicators,
                [ema21, ema50, ema200],
                psar,
                closes,
                atr,
                active_trend,
                cross_index,
                cross_age,
                i
            )

            active_fast_trend, cross_index_fast, cross_age_fast, fast_signal = CrossEmaStrategy.analyze_ema_trend(
                indicators,
                [ema9, ema21, ema50, ema200],
                psar,
                closes,
                atr,
                active_fast_trend,
                cross_index_fast,
                cross_age_fast,
                i
            )

            active_macro_trend, cross_index_macro, cross_age_macro, macro_signal = CrossEmaStrategy.analyze_ema_trend(
                indicators,
                [ema50, ema200],
                psar,
                closes,
                atr,
                active_macro_trend,
                cross_index_macro,
                cross_age_macro,
                i
            )
            
            if fast_signal:
                current_signal = fast_signal
            if mid_signal:
                current_signal = mid_signal
            elif macro_signal:
                current_signal = macro_signal


            if lateral[i] and current_signal != Signal.CLOSE:
                current_signal = None

            if current_signal is not None and current_signal != last_signal:

                if trend_signal[i-1] == Signal.CLOSE:
                    last_signal = None

                trend_signal[i] = current_signal
                last_signal = current_signal
                entry_price = closes[i]
                active_trend = None
                profits = []

        return trend_signal
    
    @staticmethod
    def analyze_ema_trend(
        indicators: IndicatorsUtils,
        emas: list[list[float]],
        psar: list[float],
        closes: list[float],
        atr: list[float],
        active_trend: Signal | None,
        cross_index: int | None,
        cross_age: int,
        i: int,
        mode: ModeEnum = ModeEnum.CONSERVATIVE
    ) -> tuple[Signal | None, int | None, int, Signal | None]:
        
        fast = emas[0]
        slow = emas[1]

        # 1. Detectar cruzamento
        prev_diff = fast[i-1] - slow[i-1]
        curr_diff = fast[i] - slow[i]

        if prev_diff <= 0 and curr_diff > 0:
            active_trend = Signal.BUY
            cross_index = i
            cross_age = 0
        elif prev_diff >= 0 and curr_diff < 0:
            active_trend = Signal.SELL
            cross_index = i
            cross_age = 0

        # 2. Atualizar idade do cruzamento
        if cross_index is not None:
            cross_age += 1

        # 3. Calcular spread normalizado
        spread = abs(fast[i] - slow[i])
        spread_pct = spread / closes[i]

        # 4. Volatilidade mínima necessária
        _, _, ema_spread = indicators.get_volatility_profile(atr)

        spread_pct_aux = spread_pct > ema_spread
        if mode == ModeEnum.AGGRESSIVE:
            spread_pct_aux = True

        # 5. Criar sinal
        buy_signal = (
            active_trend == Signal.BUY
            and spread_pct_aux
            and CrossEmaStrategy.are_emas_ordered(emas, i, Signal.BUY)
            and closes[i] > psar[i]
            and cross_age < 10
        )

        sell_signal = (
            active_trend == Signal.SELL
            and spread_pct_aux
            and CrossEmaStrategy.are_emas_ordered(emas, i, Signal.SELL)
            and closes[i] < psar[i]
            and cross_age < 10
        )

        if buy_signal:
            final_signal = Signal.BUY
        elif sell_signal:
            final_signal = Signal.SELL
        else:
            final_signal = None

        return active_trend, cross_index, cross_age, final_signal
    
    @staticmethod
    def are_emas_ordered(emas: list[list[float]], i: int, signal: Signal, min_diff_pct=0.0001):
        """
        direction = "up"  → ema1 < ema2 < ema3 ...
        direction = "down" → ema1 > ema2 > ema3 ...
        """
        for a, b in zip(emas, emas[1:]):
            pa = a[i]
            pb = b[i]

            # Evita EMAs encostadas (mercado enrolado)
            if abs(pa - pb) < pa * min_diff_pct:
                return False

            if signal == Signal.BUY and not (pa > pb):
                return False

            if signal == Signal.SELL and not (pa < pb):
                return False

        return True

    
    @staticmethod
    def check_exit_signal(
        last_signal: Signal | None,
        profits: list[float],
        current_profit_pct: float| None,
        psar_value: float,
        close_value: float,
        trailing_n: int,
        min_profit_threshold: float
    ):
        """
        Avalia se deve sair da posição com base nas condições de exit logic.
        Retorna Signal.CLOSE ou None.
        """

        # Se não há posição aberta → nada a fazer
        if last_signal not in (Signal.BUY, Signal.SELL) or current_profit_pct is None:
            return None

        # --------------------------
        # 1. EXIT: Trailing profit descendente + PSAR
        # --------------------------
        if len(profits) >= trailing_n and current_profit_pct > min_profit_threshold:
            # últimos N profits estão sempre a descer
            if all(profits[-k] < profits[-(k+1)] for k in range(1, trailing_n)):
                # validação pelo PSAR
                if (psar_value < close_value and last_signal == Signal.SELL) or \
                (psar_value > close_value and last_signal == Signal.BUY):
                    return Signal.CLOSE

        # --------------------------
        # 2. EXIT: Mercado sem direção
        # --------------------------
        profit_pos = sum(1 for x in profits if x > 0)
        profit_neg = sum(1 for x in profits if x < 0)

        if len(profits) >= trailing_n and profit_neg >= profit_pos and current_profit_pct > min_profit_threshold:
            return Signal.CLOSE

        return None
    
    @staticmethod
    def get_rsi_stoch_signal(rsi: list[float], stoch_k: list[float], stoch_d: list[float], i: int, stoch_oversold=20, stoch_overbought=80, rsi_oversold=30, rsi_overbought=70):
        """
        Gera sinal baseado na lógica:
        - VENDA: RSI > 50 e Stoch K cruza para baixo D
        - COMPRA: RSI < 50 e Stoch K cruza para cima D
        
        Os arrays devem ter o mesmo tamanho.
        Retorna Signal.BUY, Signal.SELL ou None.
        """

        # Últimos valores
        rsi_now = rsi[i]
        k_now = stoch_k[i]
        d_now = stoch_d[i]

        # Valores anteriores (para detectar cruzamento)
        k_prev = stoch_k[i - 1]
        d_prev = stoch_d[i - 1]
        rsi_prev = rsi[i - 1]

        # ----------------------------
        #  COMPRA
        # ----------------------------
        # RSI < 50 → mercado já está fraco/baixo
        # K cruza para cima de D → momentum a inverter para cima
        if k_prev < d_prev and k_now > d_now:
            if k_now < stoch_oversold and rsi_now < rsi_oversold:
                return Signal.BUY

        # ----------------------------
        #  VENDA
        # ----------------------------
        # RSI > 50 → mercado esticado para cima
        # K cruza para baixo de D → momentum a inverter para baixo
        if k_prev > d_prev and k_now < d_now:
            if k_now > stoch_overbought and rsi_now > rsi_overbought:
                return Signal.SELL

        return None