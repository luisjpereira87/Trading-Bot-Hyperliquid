from commons.enums.signal_enum import Signal
from commons.models.signal_result_dclass import SignalResult
from commons.models.strategy_base_dclass import StrategyBase
from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from trading_bot.exchange_base import ExchangeBase


class DonchianStrategy(StrategyBase):

    def __init__(self, exchange: ExchangeBase):
        super().__init__()

        self.exchange = exchange
        self.ohlcv: OhlcvWrapper
        self.ohlcv_higher: OhlcvWrapper
        self.symbol = None
        self.indicators = None

    def required_init(self, ohlcv: OhlcvWrapper, ohlcv_higher: OhlcvWrapper, symbol: str, price_ref: float):
        self.ohlcv = ohlcv
        self.ohlcv_higher = ohlcv_higher
        self.symbol = symbol
        self.price_ref = price_ref
        self.indicators = IndicatorsUtils(ohlcv)

    def set_params(self, params: StrategyParams):
        pass

    def set_candles(self, ohlcv):
        self.ohlcv = ohlcv

    def set_higher_timeframe_candles(self, ohlcv_higher: OhlcvWrapper):
        self.ohlcv_higher = ohlcv_higher

    async def get_signal(self) -> SignalResult:

        if self.symbol is None or self.indicators is None:
            return SignalResult(Signal.HOLD, None, None)

        last_closed_candle = self.ohlcv.get_last_closed_candle()
        supertrend, trend, upperband, lowerband, supertrend_smooth, _, _ = self.indicators.supertrend()
        signal_val = DonchianStrategy.build_signal(self.indicators, self.ohlcv)

        signal = signal_val[-2]
        close = last_closed_candle.close
        closes = self.ohlcv.closes

        lookback = 1
        MAX_SL_DIST_PCT = 1.5

        if signal == Signal.BUY:
            # sl = close - (close * 0.005)
            # tp = close + ((close * 0.005) * 2.5)
            sl = min(lowerband[-lookback:])  # SL no ponto mais baixo da banda
            tp = max(upperband[-lookback:]) + (max(upperband[-lookback:]) - sl) * 0.5


        elif signal == Signal.SELL:
            # sl = close + (close * 0.005)
            # tp = close - ((close * 0.005) * 2.5)
            sl = max(upperband[-lookback:])  # SL no ponto mais alto da banda
            tp = min(lowerband[-lookback:]) - (sl - min(lowerband[-lookback:])) * 0.5
        else:
            return SignalResult(signal, None, None, None, 0, signal_val[-2])

        # =========================================================================
        # 🎯 NOVA VALIDAÇÃO: TRAVA DE DISTÂNCIA MÁXIMA DO SL
        # =========================================================================
        sl_distance_pct = (abs(close - sl) / close) * 100

        if sl_distance_pct > MAX_SL_DIST_PCT:
            # Se o stop estiver demasiado longe, cancelamos a entrada para evitar a perda de $70-$100
            # print(f"⚠️ {self.symbol} - Sinal {signal.name} abortado! SL muito largo: {sl_distance_pct:.2f}%")
            return SignalResult(Signal.HOLD, None, None, None, 0, signal_val[-2])
        # =========================================================================

        # valida relação risco/benefício
        risk = abs(close - sl)
        reward = abs(tp - close)

        if (signal == Signal.BUY or signal == Signal.SELL) and reward < risk:
            # ajusta SL e TP dinamicamente
            sl_adjusted = close - (risk * 0.5) if signal == Signal.BUY else close + (risk * 0.5)
            tp_adjusted = close + (reward * 1.5) if signal == Signal.BUY else close - (reward * 1.5)

            # return SignalResult(Signal.HOLD, sl, tp_adjusted, None, 0, signal_val[-2])
            return SignalResult(Signal.HOLD, None, None, None, 0, signal_val[-2])

        return SignalResult(signal, sl, tp, None, 0, signal_val[-2])

    @staticmethod
    def build_signal(indicators: IndicatorsUtils, ohlcv: OhlcvWrapper, trailing_n=3):
        closes = ohlcv.closes
        lows = ohlcv.lows
        highs = ohlcv.highs
        n = len(closes)
        trend_signal = [Signal.HOLD] * n

        """
        ema21 = indicators.ema(21)
        ema50 = indicators.ema(50)
        ema200 = indicators.ema(200)
        adx = indicators.adx()
        rsi, rsi_ema = indicators.rsi()

        donchian = indicators.donchian_channels(52)
        dc_upper = donchian['dc_upper']
        dc_lower = donchian['dc_lower']
        dc_mid = donchian['dc_mid']
        distancia_minima_pct = 3.0

        classify_candles = indicators.classify_candles()
        atr = indicators.atr()
        """
        cooldown_n = 10

        # direction = indicators.market_structure_rsi()

        ema5_h = indicators.ema_list(highs, 5)
        ema5_l = indicators.ema_list(lows, 5)
        ema100_h = indicators.ema_list(highs, 100)
        ema100_l = indicators.ema_list(lows, 100)
        ema200 = indicators.ema_list(closes, 200)
        # rsi, rsi_ema = indicators.rsi()

        # Listas para Sinais
        buy_idx, buy_val = [], []
        sell_idx, sell_val = [], []

        # Variáveis de Estado (Memória do Pullback)
        waiting_buy = False
        waiting_sell = False

        macro_trend = 0
        for i in range(200, n):
            current_signal = None

            slope_200 = ema200[i] - ema200[i - 5]
            min_slope = ema200[i] * 0.0001  # Filtro de inclinação mínima (ajustável)

            # 2. LÓGICA DE COMPRA (LONG)
            if closes[i] > ema200[i] and slope_200 > min_slope:
                # Verificamos se o RSI está acima de 50 (Confirmação de força)
                # if rsi[i] > 30 and rsi[i] > rsi_ema[i]:
                if lows[i] <= ema100_h[i]:  # Toque no canal de Pullback
                    waiting_buy = True

                if waiting_buy and closes[i] > ema5_h[i]:
                    # Filtro extra: O candle de sinal deve ter volume acima da média
                    # if volumes[i] > np.mean(volumes[i - 20:i]):
                    buy_idx.append(i)
                    buy_val.append(lows[i])
                    current_signal = Signal.BUY
                    waiting_buy = False

            # 3. LÓGICA DE VENDA (SHORT)
            elif closes[i] < ema200[i] and slope_200 < -min_slope:
                # if rsi[i] < 70 and rsi[i] < rsi_ema[i]:
                if highs[i] >= ema100_l[i]:
                    waiting_sell = True

                if waiting_sell and closes[i] < ema5_l[i]:
                    # if volumes[i] > np.mean(volumes[i - 20:i]):
                    sell_idx.append(i)
                    sell_val.append(highs[i])
                    current_signal = Signal.SELL
                    waiting_sell = False

            # 4. INVALIDAÇÃO AUTOMÁTICA
            # Se o preço cruzar a EMA 200 antes do sinal de entrada, o pullback faliu
            if waiting_buy and closes[i] < ema200[i]: waiting_buy = False
            if waiting_sell and closes[i] > ema200[i]: waiting_sell = False

            """
            dc_mid_ascending = dc_lower[i] > dc_mid[i - 3]
            dc_mid_descending = dc_mid[i] < dc_mid[i - 3]

            if dc_mid_ascending and closes[i] > dc_mid[i]:
                macro_trend = 1  # Tendência de Alta oficializada! 📈
            elif dc_mid_descending and closes[i] < dc_mid[i]:
                macro_trend = -1  # Tendência de Baixa oficializada! 📉
            else:
                macro_trend = 0  # Neutro / Lateralização agressiva
            """
            """
            canal_range = dc_upper[i] - dc_lower[i]
            bandwidth_pct = (canal_range / closes[i]) * 100

            spread = abs(ema50[i] - ema200[i])
            spread_pct = spread / closes[i]

            # _, _, ema_spread = indicators.get_volatility_profile(atr)
            # print("AQUIII", i, spread_pct, ema_spread)

            if closes[i] > ema200[i]:
                macro_trend = 1
            else:
                macro_trend = -1
            """
            """
            if spread_pct < 0.010:
                continue

            if bandwidth_pct < distancia_minima_pct:
                continue
            """
            """
            # Verificar se pelo menos uma das bandas se moveu nas últimas 3 velas
            banda_sup_a_subir = dc_upper[i] > dc_upper[i - 3]
            banda_inf_a_subir = dc_lower[i] > dc_lower[i - 3]

            banda_sup_a_descer = dc_upper[i] < dc_upper[i - 3]
            banda_inf_a_descer = dc_lower[i] < dc_lower[i - 3]

            # 🟢 TENDÊNCIA DE ALTA (OR): Pelo menos uma linha empurra para cima
            # E o preço confirma negociando acima do meio do canal
            if (banda_sup_a_subir or banda_inf_a_subir) and closes[i] > dc_mid[i]:
                macro_trend = 1

            # 🔴 TENDÊNCIA DE BAIXA (OR): Pelo menos uma linha empurra para baixo
            # E o preço confirma negociando abaixo do meio do canal
            elif (banda_sup_a_descer or banda_inf_a_descer) and closes[i] < dc_mid[i]:
                macro_trend = -1

            else:
                macro_trend = 0
            """

            """
            bull_ema = ema21[i] > ema50[i] > ema200[i]
            bear_ema = ema21[i] < ema50[i] < ema200[i]

            if direction[i] == 1:
                current_signal = Signal.BUY
            elif direction[i] == -1:
                current_signal = Signal.SELL
            """
            """
            # print("AQUIII", adx[i], macro_trend, closes[i] > dc_upper[i], closes[i] < dc_lower[i])
            if adx[i] > 20 and spread_pct >= 0.010 and bandwidth_pct >= distancia_minima_pct:
                if macro_trend == 1:
                    retest_ema21_bull = lows[i - 1] <= ema21[i - 1] and closes[i] > ema21[
                        i] and closes[i] > closes[i - 1]

                    retest_ema50_bull = lows[i - 1] <= ema50[i - 1] and closes[i] > ema50[
                        i] and closes[i] > closes[i - 1]

                    if retest_ema21_bull or retest_ema50_bull:
                        current_signal = Signal.BUY

                elif macro_trend == -1:
                    retest_ema21_bear = highs[i - 1] >= ema21[i - 1] and closes[i] < ema21[
                        i] and closes[i] < closes[i - 1]

                    retest_ema50_bear = highs[i - 1] >= ema50[i - 1] and closes[i] < ema50[
                        i] and closes[i] < closes[i - 1]

                    if retest_ema21_bear or retest_ema50_bear:
                        current_signal = Signal.SELL
            """

            if current_signal is not None:

                if all(trend_signal[i - k] == Signal.HOLD for k in range(1, cooldown_n + 1)):
                    # Histórico limpo! Pode abrir o trade.
                    trend_signal[i] = current_signal
                else:
                    # Se pelo menos uma não foi HOLD, significa que há um sinal recente. Trava!
                    trend_signal[i] = Signal.HOLD

                # trend_signal[i] = current_signal
                last_signal = current_signal
                entry_price = closes[i]
                profits = []

        return trend_signal

    @staticmethod
    def check_exit_signal(
            last_signal: Signal | None,
            profits: list[float],
            current_profit_pct: float | None,
            trailing_n: int,
            min_profit_threshold: float,
            signal_indicator: int
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
            if all(profits[-k] < profits[-(k + 1)] for k in range(1, trailing_n)):
                return Signal.CLOSE

        # --------------------------
        # 2. EXIT: Cruzamento contrário regression_slope_oscillator
        # --------------------------
        if current_profit_pct != None and (last_signal == Signal.SELL and signal_indicator < 0 or \
                                           last_signal == Signal.BUY and signal_indicator > 0) and current_profit_pct > min_profit_threshold:
            return Signal.CLOSE

        # if current_profit_pct != None and not gap_is_accelerating and current_profit_pct > min_profit_threshold:
        #    return Signal.CLOSE

        """
        if last_signal == Signal.BUY:
            # Reversão no BUY: O preço bateu no topo e foi rejeitado ou virou Bear
            reversal_types = [
                CandleType.TOP_EXHAUSTION, 
                CandleType.WEAK_BULL,  # Subiu mas deixou muito pavio superior
                CandleType.STRONG_BEAR, 
                CandleType.BEAR
            ]
            if classify_candle in reversal_types and current_profit_pct > min_profit_threshold:
                return Signal.CLOSE

        elif last_signal == Signal.SELL:
            # Reversão no SELL: O preço bateu no fundo e foi rejeitado ou virou Bull
            reversal_types = [
                CandleType.BOTTOM_EXHAUSTION, 
                CandleType.WEAK_BEAR, # Desceu mas deixou muito pavio inferior
                CandleType.STRONG_BULL, 
                CandleType.BULL
            ]
            if classify_candle in reversal_types and current_profit_pct > min_profit_threshold:
                return Signal.CLOSE
            """

        return None
