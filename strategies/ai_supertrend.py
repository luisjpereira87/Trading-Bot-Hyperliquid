import logging
from collections import deque
from typing import Optional

import numpy as np

from commons.enums.mode_enum import ModeEnum
from commons.enums.signal_enum import Signal
from commons.models.signal_result_dclass import SignalResult
from commons.models.strategy_base_dclass import StrategyBase
from commons.models.strategy_params_dclass import StrategyParams
from commons.models.trade_snapashot_dclass import TradeSnapshot
from commons.utils.indicators.indicators_utils import IndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from commons.utils.strategies.price_action_utils import PriceActionUtils
from commons.utils.strategies.sl_tp_utils import SlTpUtils
from commons.utils.strategies.trend_utils import TrendUtils
from strategies.generic_strategy import SignalStrategy
from trading_bot.exchange_client import ExchangeClient


class AISuperTrend(StrategyBase):
    def __init__(self, exchange: ExchangeClient):
        super().__init__()
    
        self.exchange = exchange
        self.ohlcv: OhlcvWrapper
        self.ohlcv_higher: OhlcvWrapper
        self.symbol = None
        self.mode = ModeEnum.CONSERVATIVE
        self.multiplier = 0.9
        self.adx_threshold = 15
        self.rsi_buy_threshold = 40
        self.rsi_sell_threshold = 60
        self.price_ref: float = 0.0

        # Variáveis de instância para parâmetros configuráveis (antes constantes)
        self.sl_multiplier_aggressive = 1.5
        self.tp_multiplier_aggressive = 3.0
        self.sl_multiplier_conservative = 2.0
        self.tp_multiplier_conservative = 4.0

        self.volume_threshold_ratio = 0.6
        self.atr_threshold_ratio = 0.6

        self.block_lateral_market = True

        self.penalty_exhaustion = 0
        self.penalty_factor = 0
        self.penalty_manipulation = 0.5
        self.penalty_confirmation_candle = 0.5

        #self.weights = StrategyUtils.equal_weights(["trend", "momentum", "oscillators", "structure"])
        #self.penalties = StrategyUtils.equal_weights(["exhaustion", "factor", "manipulation", "abnormal_volume"])

   
        self.weights = {
            "trend": 0.125, 
            "momentum": 0,
            "oscillators": 0.125,
            "structure": 0.125, 
            "early_signal": 0.625
        }
        """
        self.penalties = {
            "exhaustion": 1.0,         
            "factor": 0.8,
            "manipulation": 1.2,
            "confirmation_candle": 0.5, 
            "abnormal_volume": 0.5
        }
        """

        self.score_history_buy = deque(maxlen=100)   # buffer circular para últimos 100 scores BUY
        self.score_history_sell = deque(maxlen=100)  # idem para SELL
        self.dynamic_threshold_percentile = 60  # percentile para threshold dinâmico (ex: 60%)

        # threshold mínimo fixo para evitar ser demasiado baixo (fallback)
        self.min_score_threshold = 0.55

    def required_init(self, ohlcv: OhlcvWrapper, ohlcv_higher: OhlcvWrapper, symbol: str, price_ref: float):
        self.ohlcv = ohlcv
        self.symbol = symbol
        self.ohlcv_higher = ohlcv_higher
        self.price_ref = price_ref
    
    def set_params(self, params: StrategyParams):
        self.mode = params.mode
        self.multiplier = params.multiplier
        self.adx_threshold = params.adx_threshold
        self.rsi_buy_threshold = params.rsi_buy_threshold
        self.rsi_sell_threshold = params.rsi_sell_threshold

        # Atualizar variáveis de instância dos parâmetros que antes eram constantes
        #self.sl_multiplier_aggressive = params.sl_multiplier_aggressive
        #self.tp_multiplier_aggressive = params.tp_multiplier_aggressive
        #self.sl_multiplier_conservative = params.sl_multiplier_conservative
        #self.tp_multiplier_conservative = params.tp_multiplier_conservative

        self.volume_threshold_ratio = params.volume_threshold_ratio
        self.atr_threshold_ratio = params.atr_threshold_ratio

        self.block_lateral_market = params.block_lateral_market

        """
        self.penalty_exhaustion = params.penalty_exhaustion
        self.penalty_factor = params.penalty_factor
        self.penalty_manipulation = params.penalty_manipulation
        self.penalty_confirmation_candle = params.penalty_confirmation_candle
        
        self.weights = {
            "trend": params.weights_trend, 
            "momentum": params.weights_momentum,
            "oscillators": params.weights_oscillators,
            "structure": params.weights_structure,
            "early_signal": params.weights_early_signal
        }

        """
        self.weights = {
            "trend": 0, 
            "momentum": 0,
            "oscillators": 0,
            "structure": 0, 
            "early_signal": 1
        }
  

    def set_candles(self, ohlcv):
        self.ohlcv = ohlcv

    def set_higher_timeframe_candles(self, ohlcv_higher: OhlcvWrapper):
        self.ohlcv_higher = ohlcv_higher

    async def get_signal(self) -> SignalResult:

        if len(self.ohlcv) == 0 or self.symbol is None:
            logging.error("Tem que executar em primeiro lugar o método required_init")
            return SignalResult(Signal.HOLD, None, None)

        if not self.has_enough_candles():
            logging.info(f"{self.symbol} - Dados insuficientes para cálculo.")
            return SignalResult(Signal.HOLD, None, None)

        last_closed = self.ohlcv.get_last_closed_candle()

        #print(f"Last closed candle - ts: {last_closed.timestamp}, open: {last_closed.open}, high: {last_closed.high}, low: {last_closed.low}, close: {last_closed.close}, volume: {last_closed.volume}")

        is_bearish_reversal, is_bullish_reversal = PriceActionUtils.detect_reversal_pattern(self.ohlcv)



        """
        if is_bearish_reversal:
            logging.info(f"{self.symbol} - Reversão de topo detetada: HOLD")
            return SignalResult(Signal.HOLD, None, None)

        if is_bullish_reversal:
            logging.info(f"{self.symbol} - Reversão de fundo detetada: HOLD")
            return SignalResult(Signal.HOLD, None, None)

        if not StrategyUtils.calculate_higher_tf_trend(self.ohlcv_higher, self.adx_threshold):
            logging.info(f"{self.symbol} - Sinal rejeitado por tendência contrária no timeframe maior.: HOLD")
            return SignalResult(Signal.HOLD, None, None)
        """

        """
        if self.mode == ModeEnum.CONSERVATIVE and StrategyUtils.is_market_manipulation(self.ohlcv):
            logging.info(f"{self.symbol} - Sinal rejeitado por deteção de manipulação de mercado: HOLD")
            return SignalResult(Signal.HOLD, None, None)
        """

        logging.info(f"{self.symbol} - Modo selecionado: {self.mode}")
        score = self.calculate_score()

        # Confiança normalizada
        buy_score = round(score["buy"], 3)
        sell_score = round(score["sell"], 3)
        hold_score = round(score["hold"], 3)
        raw_score = max(score["buy"], score["sell"])

        # Atualiza históricos dos scores
        self.score_history_buy.append(buy_score)
        self.score_history_sell.append(sell_score)

        # Calcula thresholds dinâmicos pelos percentis dos históricos
        threshold_buy = max(self.min_score_threshold,
                            float(np.percentile(self.score_history_buy, self.dynamic_threshold_percentile)))
        threshold_sell = max(self.min_score_threshold,
                            float(np.percentile(self.score_history_sell, self.dynamic_threshold_percentile)))

        # Logging para debug
        logging.info(f"{self.symbol} - Score BUY: {buy_score}, SELL: {sell_score}, HOLD: {hold_score}")
        logging.info(f"{self.symbol} - Threshold dinâmico BUY: {threshold_buy:.3f}, SELL: {threshold_sell:.3f}")

        if buy_score > sell_score and buy_score >= threshold_buy:
            signal = Signal.BUY
        elif sell_score > buy_score and sell_score >= threshold_sell:
            signal = Signal.SELL
        else:
            return SignalResult(Signal.HOLD, None, None, hold_score, max(buy_score, sell_score), buy_score, sell_score, hold_score, None)

        try:
            sl, tp = SlTpUtils.calculate_sl_tp_simple(
                        self.ohlcv,
                        self.price_ref,
                        signal
                    )
            """
            sl, tp = StrategyUtils.calculate_sl_tp(
                        self.ohlcv,
                        self.price_ref, 
                        signal,
                        self.mode,
                        self.sl_multiplier_aggressive,
                        self.tp_multiplier_aggressive,
                        self.sl_multiplier_conservative,
                        self.tp_multiplier_conservative
                    )
            """
        except Exception as e:
            logging.warning(f"{self.symbol} - Erro ao calcular SL/TP: {e}")
            return SignalResult(Signal.HOLD, None, None, None, 0)

        return SignalResult(signal, sl, tp, hold_score, max(buy_score, sell_score), buy_score, sell_score, hold_score,  self.get_trade_snapshot(signal, sl, tp))
    
    def get_trade_snapshot(self, signal: Signal, sl: float, tp: float) -> Optional[TradeSnapshot]:
        if self.symbol == None:
            return None


        indicators = IndicatorsUtils(self.ohlcv)
        rsi = indicators.rsi()[-1]
        stoch_k, stoch_d = indicators.stochastic()
        adx = indicators.adx()[-1]
        macd_line, macd_signal_line = indicators.macd()
        macd = macd_line[-1] - macd_signal_line[-1]
        cci = indicators.cci(20)[-1]
        candle_type = PriceActionUtils.get_candle_type(self.ohlcv.get_last_closed_candle())
        timestamp = self.ohlcv.get_last_closed_candle().timestamp
        volume_ratio = TrendUtils.calculate_volume_ratio(self.ohlcv)
        atr_ratio = TrendUtils.calculate_atr_ratio(self.ohlcv)

        return TradeSnapshot(
            symbol=self.symbol,
            entry_price=self.price_ref,
            sl=sl,
            tp=tp,
            signal=signal,
            size=0.0,
            candle_type=candle_type,
            rsi=rsi,
            stochastic=stoch_k[-1],  # ou uma média k+d
            adx=adx,
            macd=macd,
            cci=cci,
            weights_trend=0,
            weights_momentum=0,
            weights_divergence=0,
            weights_oscillators=0,
            weights_price_action=0,
            weights_price_levels=0,
            weights_channel_position=0, 
            penalty_exhaustion=self.penalty_exhaustion,
            penalty_factor=self.penalty_factor,
            penalty_manipulation=self.penalty_manipulation,
            penalty_confirmation_candle=self.penalty_confirmation_candle,
            volume_ratio=volume_ratio,
            atr_ratio=atr_ratio,
            timestamp=timestamp
        )

    def calculate_score(self):
        score = {"buy": 0.0, "sell": 0.0, "hold": 0.0}

        if self.symbol is None:
            return score

        #max_score = sum(weights.values())
        score["buy"], score["sell"] = SignalStrategy.get_signal(self.ohlcv)
        score["buy"] = max(score["buy"], 0)
        score["sell"] = max(score["sell"], 0)
        score["hold"] = max(1 - (score["buy"] + score["sell"]), 0)

        return  score 
    
    









