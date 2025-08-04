import logging
from collections import deque
from typing import Optional, Tuple

import numpy as np

from commons.enums.mode_enum import ModeEnum
from commons.enums.signal_enum import Signal
from commons.models.signal_result_dclass import SignalResult
from commons.models.strategy_base_dclass import StrategyBase
from commons.models.strategy_params_dclass import StrategyParams
from commons.models.trade_snapashot_dclass import TradeSnapshot
from commons.utils.ohlcv_wrapper import OhlcvWrapper
from strategies.strategy_utils import StrategyUtils
from trading_bot.exchange_client import ExchangeClient

from .indicators import Indicators


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

        self.weights = {
            "trend": 1.0,         # EMA
            "momentum": 0.8,
            "oscillators": 1.2,
            "price_action": 0.5,  # candle, setup 123, breakout, bandas
            "price_levels": 0.5,
            "divergence": 0.5
        }

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
        self.sl_multiplier_aggressive = params.sl_multiplier_aggressive
        self.tp_multiplier_aggressive = params.tp_multiplier_aggressive
        self.sl_multiplier_conservative = params.sl_multiplier_conservative
        self.tp_multiplier_conservative = params.tp_multiplier_conservative

        self.volume_threshold_ratio = params.volume_threshold_ratio
        self.atr_threshold_ratio = params.atr_threshold_ratio

        self.block_lateral_market = params.block_lateral_market

        self.penalty_exhaustion = params.penalty_exhaustion
        self.penalty_factor = params.penalty_factor
        self.penalty_manipulation = params.penalty_manipulation
        self.penalty_confirmation_candle = params.penalty_confirmation_candle

        self.weights = {
            "trend": params.weights_trend, 
            "momentum": params.weights_momentum,
            "oscillators": params.weights_oscillators,
            "price_action": params.weights_price_action,  
            "price_levels" : params.weights_price_levels,
            "divergence": params.weights_divergence,
            "channel_position": params.weights_channel_position,
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

        is_bearish_reversal, is_bullish_reversal = StrategyUtils.detect_reversal_pattern(self.ohlcv)

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
        except Exception as e:
            logging.warning(f"{self.symbol} - Erro ao calcular SL/TP: {e}")
            return SignalResult(Signal.HOLD, None, None, None, 0)

        return SignalResult(signal, sl, tp, hold_score, max(buy_score, sell_score), buy_score, sell_score, hold_score,  self.get_trade_snapshot(signal, sl, tp))
    
    def get_trade_snapshot(self, signal: Signal, sl: float, tp: float) -> Optional[TradeSnapshot]:
        if self.symbol == None:
            return None


        indicators = Indicators(self.ohlcv)
        rsi = indicators.rsi()[-1]
        stoch_k, stoch_d = indicators.stochastic()
        adx = indicators.adx()[-1]
        macd_line, macd_signal_line = indicators.macd()
        macd = macd_line[-1] - macd_signal_line[-1]
        cci = indicators.cci(20)[-1]
        trend = self._score_trend()
        momentum = self._score_momentum()
        divergence = self._score_divergence()
        oscillators = self._score_oscillators()
        price_action = self._score_price_action()
        price_levels = self._score_price_levels()
        channel_position = self._score_channel_position()
        candle_type = StrategyUtils.get_candle_type(self.ohlcv.get_last_closed_candle())
        timestamp = self.ohlcv.get_last_closed_candle().timestamp
        volume_ratio = StrategyUtils.calculate_volume_ratio(self.ohlcv)
        atr_ratio = StrategyUtils.calculate_atr_ratio(self.ohlcv)

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
            weights_trend=trend,
            weights_momentum=momentum,
            weights_divergence=divergence,
            weights_oscillators=oscillators,
            weights_price_action=price_action,
            weights_price_levels=price_levels,
            weights_channel_position=channel_position, 
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

        score["buy"] += self._score_trend() * self.weights.get("trend", 0)
        score["sell"] += self._score_trend(sell=True) * self.weights.get("trend", 0)

        score["buy"] += self._score_momentum() * self.weights.get("momentum", 0)
        score["sell"] += self._score_momentum(sell=True) * self.weights.get("momentum", 0)

        score["buy"] += self._score_oscillators() * self.weights.get("oscillators", 0)
        score["sell"] += self._score_oscillators(sell=True) * self.weights.get("oscillators", 0)

        score["buy"] += self._score_price_action() * self.weights.get("price_action", 0)
        score["sell"] += self._score_price_action(sell=True) * self.weights.get("price_action", 0)

        score["buy"] += self._score_price_levels() * self.weights.get("price_levels", 0)
        score["sell"] += self._score_price_levels(sell=True) * self.weights.get("price_levels", 0)

        # Detectar lateralidade e breakout (multiplica score)
        multiplier = self._score_lateral_and_breakout()
        score["buy"] *= multiplier
        score["sell"] *= multiplier

        # Divergências RSI/Estocástico (isolado)
        score["buy"] += self._score_divergence() * self.weights.get("divergence", 0)
        score["sell"] += self._score_divergence(sell=True) * self.weights.get("divergence", 0)


        # 🔽 NOVO: aplicar penalização por volume
        buy_penalty, sell_penalty = StrategyUtils.calculate_volume_penalty(self.ohlcv)
        score["buy"] *= buy_penalty
        score["sell"] *= sell_penalty

        """
        # Penalização por candle em formação fraco
        candle_buy_penalty, candle_sell_penalty = self._penalty_incomplete_candle()
        score["buy"] *= candle_buy_penalty
        score["sell"] *= candle_sell_penalty
        """

        # Canal de tendencia
        score["buy"] += self._score_channel_position(sell=False) * self.weights.get("channel_position", 0)
        score["sell"] += self._score_channel_position(sell=True) * self.weights.get("channel_position", 0)

        # Normalização final
        max_score = sum(self.weights.values())
        score["buy"] = max(score["buy"], 0)
        score["sell"] = max(score["sell"], 0)
        if max_score > 0:
            score["buy"] /= max_score
            score["sell"] /= max_score
            score["hold"] = max(1 - (score["buy"] + score["sell"]), 0)
        else:
            score["hold"] = 1

        return score

    def _score_trend(self, sell=False):
        trend_signal = StrategyUtils.trend_signal_with_adx(self.ohlcv, self.symbol, self.adx_threshold)
        if sell:
            return 1 if trend_signal == -1 else 0
        else:
            return 1 if trend_signal == 1 else 0

    def _score_momentum(self, sell=False):
        indicators = Indicators(self.ohlcv)
        rsi = indicators.rsi()[-1]
        stochastic_signal = StrategyUtils.stochastic(self.ohlcv)

        buy_points = 0
        sell_points = 0
        if rsi < self.rsi_buy_threshold:
            buy_points += 1
        elif rsi > self.rsi_sell_threshold:
            sell_points += 1

        if stochastic_signal == Signal.BUY:
            buy_points += 1
        elif stochastic_signal == Signal.SELL:
            sell_points += 1
    

        buy_points /= 2
        sell_points /= 2

    
        # ⚠️ Penalização por volume anormal
        if StrategyUtils.is_abnormal_volume(self.ohlcv):
            penalty = self.weights.get("abnormal_volume_penalty", 0.5)  # Configurável no JSON
            if sell:
                sell_points *= penalty
            else:
                buy_points *= penalty


        return sell_points if sell else buy_points

    def _score_oscillators(self, sell=False):
        indicators = Indicators(self.ohlcv)
        macd_line, macd_signal_line = indicators.macd()
        cci_list = indicators.cci(20)
        macd_val = macd_line[-1]
        macd_signal_val = macd_signal_line[-1]
        cci_val = cci_list[-1]

        buy_points = 0
        sell_points = 0

        if macd_val > macd_signal_val:
            buy_points += 1
        elif macd_val < macd_signal_val:
            sell_points += 1

        if cci_val < -100:
            buy_points += 1
        elif cci_val > 100:
            sell_points += 1

        buy_points /= 2
        sell_points /= 2

        return sell_points if sell else buy_points

    def _score_price_action(self, sell=False):
        pa_signal = StrategyUtils.check_price_action_signals(self.ohlcv)
        buy_points = 1 if pa_signal == 'buy' else 0
        sell_points = 1 if pa_signal == 'sell' else 0

        # Penaliza sinais fracos
        candle = self.ohlcv.get_last_closed_candle()
        if StrategyUtils.is_weak_confirmation_candle(candle):
            penalty = self.penalty_confirmation_candle
            if not sell:
                buy_points *= penalty
            else:
                sell_points *= penalty

        # Penaliza manipulação de mercado
        manip_score = StrategyUtils.get_market_manipulation_score(self.ohlcv)
        penalty = self.penalty_manipulation
        buy_points *= (1 - manip_score * penalty)
        sell_points *= (1 - manip_score * penalty)

        return sell_points if sell else buy_points

    def _score_price_levels(self, sell=False):
        upper_band, lower_band = StrategyUtils.calculate_bands(self.ohlcv, multiplier=self.multiplier)
        price = self.ohlcv.get_last_closed_candle().close
        dist_to_upper = abs(price - upper_band[-1])
        dist_to_lower = abs(price - lower_band[-1])
        band_range = upper_band[-1] - lower_band[-1]

        proximity_buy = 0
        proximity_sell = 0
        if band_range > 0:
            if dist_to_lower / band_range < 0.1:
                proximity_buy += 1
            if dist_to_upper / band_range < 0.1:
                proximity_sell += 1

        #support, resistance = StrategyUtils.detect_support_resistance(self.ohlcv)
        dist_to_res, dist_to_sup = StrategyUtils.get_distance_to_levels(self.ohlcv, price, lookback=50)
        penalty_buy = min(1.0, max(0.0, 1 - (dist_to_res / (price * 0.01))))
        penalty_sell = min(1.0, max(0.0, 1 - (dist_to_sup / (price * 0.01))))

        proximity_buy *= (1 - penalty_buy * self.penalty_factor)
        proximity_sell *= (1 - penalty_sell * self.penalty_factor)

        # Exaustão
        is_top, is_bottom = StrategyUtils.is_exhaustion_candle(self.ohlcv)
        if is_top:
            proximity_buy -= self.penalty_exhaustion
        if is_bottom:
            proximity_sell -= self.penalty_exhaustion

        return proximity_sell if sell else proximity_buy

    def _score_lateral_and_breakout(self):
        adx = Indicators(self.ohlcv).adx()
        is_lateral = StrategyUtils.detect_lateral_market(self.ohlcv, self.symbol, self.adx_threshold)
        breakout = StrategyUtils.is_breakout_candle(self.ohlcv, -1)

        if self.block_lateral_market and is_lateral and not breakout:
            adx_now = adx[-1]
            multiplier = adx_now / (self.adx_threshold + 1e-8)
            logging.info(f"{self.symbol} - Mercado lateral → Penalização aplicada ao score: x{multiplier:.2f}")
            return multiplier
        elif breakout:
            logging.info(f"{self.symbol} - Breakout detetado → Bónus aplicado")
            return 1.2  # bónus
        else:
            return 1.0

    def _score_divergence(self, sell=False):
        divergence_signal = StrategyUtils.detect_divergence(self.ohlcv)
        if sell:
            return 1 if divergence_signal == -1 else 0
        else:
            return 1 if divergence_signal == 1 else 0
    
    
    def _penalty_incomplete_candle(self) -> tuple[float, float]:
        """Aplica penalização ao score de BUY e SELL com base no candle em formação."""
        candle = self.ohlcv.get_current_candle()  # candle em formação
        open = candle.open
        high = candle.high
        low = candle.low
        close = candle.close
        #_, open_, high, low, close, volume = candle
        body = abs(close - open)
        total_range = high - low
        wick_top = high - max(open, close)
        wick_bottom = min(open, close) - low

        # Evitar divisão por zero
        if total_range == 0:
            return 1.0, 1.0

        # Parâmetros ajustáveis (podem ser otimizados com Optuna)
        wick_ratio_threshold = 0.6
        body_ratio_threshold = 0.3
        penalty_value = 0.8  # Penalização moderada (multiplica por 0.8)

        buy_penalty = 1.0
        sell_penalty = 1.0

        # BUY: wick superior longo e corpo pequeno => fraqueza
        if wick_top / total_range > wick_ratio_threshold and body / total_range < body_ratio_threshold:
            buy_penalty = penalty_value

        # SELL: wick inferior longo e corpo pequeno => fraqueza na venda
        if wick_bottom / total_range > wick_ratio_threshold and body / total_range < body_ratio_threshold:
            sell_penalty = penalty_value

        return buy_penalty, sell_penalty
    
    def _score_channel_position(self, sell=False) -> float:
        upper_band, lower_band = StrategyUtils.calculate_bands(self.ohlcv, multiplier=self.multiplier)
        close = self.ohlcv.get_last_closed_candle().close

        # Validar que as bandas estão corretamente calculadas
        if not upper_band or not lower_band or len(upper_band) < 1:
            return 0.0

        top = upper_band[-1]
        bottom = lower_band[-1]
        band_range = top - bottom

        if band_range == 0:
            return 0.0

        # Distância relativa ao canal
        dist_from_bottom = (close - bottom) / band_range
        score = 1 - 2 * dist_from_bottom  # vai de +1 (na base) até -1 (no topo)

        # Inverter se for SELL
        if sell:
            score *= -1

        # Reforçar score se canal está inclinado na direção do sinal
        recent_top_slope = StrategyUtils.linear_slope(upper_band[-5:])
        recent_bottom_slope = StrategyUtils.linear_slope(lower_band[-5:])
        avg_slope = (recent_top_slope + recent_bottom_slope) / 2

        # Ajustar score com base na inclinação
        if not sell and avg_slope > 0:
            score += 0.2  # reforça BUY
        elif sell and avg_slope < 0:
            score += 0.2  # reforça SELL

        return max(min(score, 1), -1)  # limitar entre -1 e +1
        









