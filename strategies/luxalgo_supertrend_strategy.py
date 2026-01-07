from commons.enums.signal_enum import Signal
from commons.models.signal_result_dclass import SignalResult
from commons.models.strategy_base_dclass import StrategyBase
from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from trading_bot.exchange_client import ExchangeClient


class LuxAlgoSupertrendStrategy(StrategyBase):

    def __init__(self, exchange: ExchangeClient):
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
        supertrend, trend, upperband, lowerband, supertrend_smooth,_,_ = self.indicators.supertrend()
        ema_cross_signal, ts = LuxAlgoSupertrendStrategy.build_signal(self.indicators, self.ohlcv)

        signal = ema_cross_signal[-2]
        close = last_closed_candle.close
        closes = self.ohlcv.closes

        lookback = 10
        if signal == Signal.BUY:
            #sl = close - (close * 0.005)
            sl = min(lowerband[-lookback:])  # SL no ponto mais baixo da banda
            tp = max(upperband[-lookback:]) + (max(upperband[-lookback:]) - sl) * 0.5
            #tp = close + ((close * 0.005) * 2.5)

        elif signal == Signal.SELL:
            #sl = upperband[-2]
            #sl = close + (close * 0.005)
            sl = max(upperband[-lookback:])  # SL no ponto mais alto da banda
            tp = min(lowerband[-lookback:]) - (sl - min(lowerband[-lookback:])) * 0.5
            #tp = close - ((close * 0.005) * 2.5)

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
        highs = ohlcv.highs
        lows = ohlcv.lows
        opens = ohlcv.opens
        n = len(closes)
        trend_signal = [Signal.HOLD] * n
        last_signal = None
        last_supertrend_value = None
        psar = indicators.psar()
        ema50 = indicators.ema(50)
        ema200 = indicators.ema(200)
        #res = indicators.luxalgo_supertrend_ai(symbol)
        supertrend = indicators.supertrend_ai()
        #two_p, two_pp, buy, sell, direction1 = indicators.two_pole_oscillator()
        #res = indicators.volumatic_vidya()
        psi = indicators.squeeze_index()
        #vidya = res.vidya
        #upper = res.upper_band
        #lower = res.lower_band
        #smoothed = res.smoothed
        #pivot_high = res.pivot_high
        #pivot_low = res.pivot_low
        #delta_vol = res.delta_volume_pct
        #is_trend_up = res.is_trend_up
        #retest = res.retest

        #lateral = indicators.detect_low_volatility(slope_threshold=0.008)
        ts = supertrend.ts
        direction = supertrend.direction  # 1 bullish, 0 bearish
        perf_score = supertrend.score
        delta_vol = supertrend.delta_vol
        retest = supertrend.retest

        entry_price = 0
        profits = []
        min_profit_threshold = 0.001
        current_profit_pct = None

        
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
            
            current_signal = LuxAlgoSupertrendStrategy.check_exit_signal(
                last_signal=last_signal,
                profits=profits,
                current_profit_pct=current_profit_pct,
                psar_value=psar[i],
                close_value=closes[i],
                trailing_n=trailing_n,
                min_profit_threshold=min_profit_threshold,
                supertrend_value=last_supertrend_value
            )
            
            #if last_signal == Signal.BUY and sell[i] or last_signal == Signal.SELL and buy[i]:
            #    current_signal = Signal.CLOSE

            #print(f"AQUI i={i}, {upper[i]}, {lower[i]}, is_trend_up={is_trend_up[i]}, delta_vol={delta_vol[i]}, direction={direction1[i]}, is_ranging={psi[i]}, retest={retest[i]}")
            
            #print(f"AQUII, {i}, {ts[i]}")
            #print("AQUII", i, direction[i], retest[i])
            # --- Detecção de tendência via EMA ---
            """
            if is_trend_up[i] and retest[i] == 1:
                current_signal = Signal.BUY

            elif not is_trend_up[i] and retest[i] == -1:
                current_signal = Signal.SELL
            """
            if direction[i-1] == -1 and direction[i] == 1 and psi[i] < 80 and ema50[i] > ema200[i]:
                current_signal = Signal.BUY

            elif direction[i-1] == 1 and direction[i] == -1 and psi[i] < 80 and ema50[i] < ema200[i]:
                current_signal = Signal.SELL

            #if lateral[i] and current_signal != Signal.CLOSE:
            #    current_signal = None
            
            if current_signal is not None and current_signal != last_signal:

                if trend_signal[i-1] == Signal.CLOSE:
                    last_signal = None

                trend_signal[i] = current_signal
                last_signal = current_signal
                #last_supertrend_value = ts[i-1]
                entry_price = closes[i]
                active_trend = None
                profits = []

        return trend_signal, ts
    
    
    @staticmethod
    def check_exit_signal(
        last_signal: Signal | None,
        profits: list[float],
        current_profit_pct: float| None,
        psar_value: float,
        close_value: float,
        trailing_n: int,
        min_profit_threshold: float,
        supertrend_value: float | None
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
                #if (psar_value < close_value and last_signal == Signal.SELL) or \
                #(psar_value > close_value and last_signal == Signal.BUY):
                return Signal.CLOSE

        # --------------------------
        # 2. EXIT: Mercado sem direção
        # --------------------------
        profit_pos = sum(1 for x in profits if x > 0)
        profit_neg = sum(1 for x in profits if x < 0)

        #if len(profits) >= trailing_n and profit_neg >= profit_pos and current_profit_pct > min_profit_threshold:
        #    return Signal.CLOSE
        
        # --------------------------
        # 3. EXIT: Falha de rompimento da linha ATR (SuperTrend)
        # --------------------------

        """
        # BUY → perdeu a linha que tinha rompido
        if supertrend_value is not None:
            if last_signal == Signal.BUY:
                if close_value < supertrend_value:
                    return Signal.CLOSE

            # SELL → perdeu a linha que tinha rompido
            if last_signal == Signal.SELL:
                print("AQUIIII", last_signal, close_value, supertrend_value, close_value > supertrend_value)
                if close_value > supertrend_value:
                    return Signal.CLOSE
        """
        return None
