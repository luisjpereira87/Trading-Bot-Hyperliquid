import numpy as np

from commons.enums.mode_enum import ModeEnum
from commons.enums.signal_enum import Signal
from commons.models.signal_result_dclass import SignalResult
from commons.models.strategy_base_dclass import StrategyBase
from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from trading_bot.exchange_base import ExchangeBase
from trading_bot.exchange_client import ExchangeClient


class DoubleBollingerBandsStrategy(StrategyBase):

    def __init__(self, exchange: ExchangeBase):
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
        supertrend, trend, upperband, lowerband, supertrend_smooth, _, _ = self.indicators.supertrend()
        ema_cross_signal = DoubleBollingerBandsStrategy.build_signal(self.indicators, self.ohlcv)

        signal = ema_cross_signal[-2]
        close = last_closed_candle.close
        closes = self.ohlcv.closes

        lookback = 10
        if signal == Signal.BUY:
            sl = min(lowerband[-lookback:])
            tp = max(upperband[-lookback:]) + (max(upperband[-lookback:]) - sl) * 0.5
        elif signal == Signal.SELL:
            sl = max(upperband[-lookback:])
            tp = min(lowerband[-lookback:]) - (sl - min(lowerband[-lookback:])) * 0.5
        else:
            return SignalResult(signal, None, None)

        risk = abs(close - sl)
        reward = abs(tp - close)

        if (signal == Signal.BUY or signal == Signal.SELL) and reward < risk:
            sl_adjusted = close - (risk * 0.5) if signal == Signal.BUY else close + (risk * 0.5)
            tp_adjusted = close + (reward * 1.5) if signal == Signal.BUY else close - (reward * 1.5)
            return SignalResult(signal, sl, tp_adjusted)

        return SignalResult(signal, sl, tp)

    @staticmethod
    def build_signal(indicators: IndicatorsUtils, ohlcv: OhlcvWrapper, trailing_n=3):
        closes = ohlcv.closes
        lows = ohlcv.lows
        highs= ohlcv.highs
        n = len(closes)

        # Variáveis de controlo da máquina de estados
        waiting_buy = False
        buy_timer = 0

        waiting_sell = False
        sell_timer = 0

        MAX_WINDOW = 20  # Janela máxima de 3 a 4 candles para o RSI alinhar

        trend_signal = [Signal.HOLD] * n
        double_bb = indicators.double_bb_rsi_logic()
        rsi, rsi_ema = indicators.rsi()
        bb20_up = double_bb['bbshort_up']
        bb20_low = double_bb['bbshort_low']
        bb20_mid = double_bb['bbshort_mid']
        bb80_up = double_bb['bblong_up']
        bb80_low = double_bb['bblong_low']
        bb80_mid = double_bb['bblong_mid']
        signals = double_bb['signals']
        atr = indicators.atr()

        atr_pct = indicators.get_normalized_atr_series(atr)
        window = max(3, int(MAX_WINDOW * atr_pct[-1]))
        for i in range(1, n):
            current_signal = Signal.HOLD

            # 2. Condições de Momentum do RSI
            rsi_bullish = (rsi[i] > rsi_ema[i]) and (rsi[i] > 55)
            rsi_bearish = (rsi[i] < rsi_ema[i]) and (rsi[i] < 45)

            # ==========================================
            # 🟢 MÁQUINA DE ESTADOS: COMPRA (LONG)
            # ==========================================
            if not waiting_buy and not waiting_sell:
                # Passo 1: O preço furou a banda inferior do timeframe maior
                if signals[i] == 1:
                    waiting_buy = True
                    buy_timer = 0

            elif waiting_buy:
                buy_timer += 1

                # Passo 2: O RSI posicionou-se a favor dentro da janela de 3-4 candles?
                if rsi_bullish:
                    current_signal = Signal.BUY
                    waiting_buy = False
                    buy_timer = 0

                # Passo 3: Se passaram mais de 4 candles e o RSI não confirmou, expira (ficou lateral)
                elif buy_timer >= window:
                    waiting_buy = False
                    buy_timer = 0

            # ==========================================
            # 🔴 MÁQUINA DE ESTADOS: VENDA (SHORT)
            # ==========================================
            if not waiting_sell and not waiting_buy:
                # Passo 1: O preço furou a banda superior do timeframe maior
                if signals[i] == -1:
                    waiting_sell = True
                    sell_timer = 0

            elif waiting_sell:
                sell_timer += 1

                # Passo 2: O RSI posicionou-se a favor dentro da janela
                if rsi_bearish:
                    current_signal = Signal.SELL
                    waiting_sell = False
                    sell_timer = 0

                # Passo 3: Expiração por tempo
                elif sell_timer >= window:
                    waiting_sell = False
                    sell_timer = 0

            trend_signal[i] = current_signal

        return trend_signal
