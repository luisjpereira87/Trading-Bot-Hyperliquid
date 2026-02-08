from commons.enums.candle_type_enum import CandleType
from commons.enums.signal_enum import Signal
from commons.models.signal_result_dclass import SignalResult
from commons.models.strategy_base_dclass import StrategyBase
from commons.models.strategy_params_dclass import StrategyParams
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.indicators.tv_indicators_utils import TvIndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from trading_bot.exchange_base import ExchangeBase
from trading_bot.exchange_client import ExchangeClient


class CrossEmaLinearRegressionStrategy(StrategyBase):

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
        self.indicators = TvIndicatorsUtils(ohlcv)

        
    
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
        signal = CrossEmaLinearRegressionStrategy.build_signal(self.indicators, self.ohlcv)

        signal = signal[-2]
        close = last_closed_candle.close
        closes = self.ohlcv.closes

        lookback = 1
        if signal == Signal.BUY:
            #sl = close - (close * 0.005)
            #tp = close + ((close * 0.005) * 2.5)
            sl = min(lowerband[-lookback:])  # SL no ponto mais baixo da banda
            tp = max(upperband[-lookback:]) + (max(upperband[-lookback:]) - sl) * 0.5
            

        elif signal == Signal.SELL:
            #sl = upperband[-2]
            #sl = close + (close * 0.005)
            #tp = close - ((close * 0.005) * 2.5)
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
    def build_signal(indicators: TvIndicatorsUtils, ohlcv: OhlcvWrapper, trailing_n = 3, gap_pct = 20):
        closes = ohlcv.closes
        n = len(closes)
        trend_signal = [Signal.HOLD] * n
        last_signal = None

        sma5 = indicators.sma(5)
        sma13 = indicators.sma(13)
        oscillator_values, signal_line, signals, gap_index = indicators.regression_slope_oscillator()
        classify_candle = indicators.classify_candles()

        entry_price = 0
        profits = []
        min_profit_threshold = 0.001
        current_profit_pct = None
        sl = None

        
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
            current_signal = CrossEmaLinearRegressionStrategy.check_exit_signal(
                last_signal=last_signal,
                profits=profits,
                current_profit_pct=current_profit_pct,
                trailing_n=trailing_n,
                min_profit_threshold=min_profit_threshold,
                signal_indicator=signals[i]
            )
            
            # Cruzamento das sma no candle anterior
            sma_cross_up = sma5[i-2] <= sma13[i-2] and sma5[i-1] > sma13[i-1]
            sma_cross_down = sma5[i-2] >= sma13[i-2] and sma5[i-1] < sma13[i-1]

            # Validar se existiu cruzamento no regression_slope_oscillator nos 2 candles anteriores
            start = max(0, i - 2)
            window = signals[start : i + 1]
            osc_confirmed_up = any(s > 1 for s in window)
            osc_confirmed_down = any(s < -1 for s in window)


            #print(f"index={i} gap_index={gap_index[i]} classify_candle={classify_candle[i]} osc_confirmed_down={osc_confirmed_down} sma_cross_down={sma_cross_down}")
            if sma_cross_up and osc_confirmed_up and closes[i] > closes[i-1] and gap_index[i] > gap_pct and not classify_candle[i] in (CandleType.WEAK_BULL, CandleType.TOP_EXHAUSTION):
                current_signal = Signal.BUY

            elif sma_cross_down and osc_confirmed_down and closes[i] < closes[i-1] and gap_index[i] > gap_pct and not classify_candle[i] in (CandleType.WEAK_BEAR, CandleType.BOTTOM_EXHAUSTION):
                current_signal = Signal.SELL

            if current_signal is not None and current_signal != last_signal:
                trend_signal[i] = current_signal
                last_signal = current_signal
                entry_price = closes[i]
                profits = []
                
        return trend_signal
    
    
    @staticmethod
    def check_exit_signal(
        last_signal: Signal | None,
        profits: list[float],
        current_profit_pct: float| None,
        trailing_n: int,
        min_profit_threshold: float,
        signal_indicator: int,
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
                return Signal.CLOSE
            
        # --------------------------
        # 2. EXIT: Cruzamento contrário regression_slope_oscillator
        # --------------------------
        if current_profit_pct != None and (last_signal == Signal.SELL and signal_indicator < 0 or \
            last_signal == Signal.BUY and signal_indicator > 0) and current_profit_pct > min_profit_threshold:
            return Signal.CLOSE

        return None
