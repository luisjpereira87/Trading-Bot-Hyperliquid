import logging
from collections import deque

import numpy as np

from commons.enums.mode_enum import ModeEnum
from commons.enums.signal_enum import Signal
from commons.models.signal_result import SignalResult
from commons.models.strategy_base import StrategyBase
from commons.models.strategy_params import StrategyParams
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
        #self.rsi:float
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

        """
        self.weights = {
            "trend": 1.0,         # EMA
            "rsi": 0.8,
            "stochastic": 1.2,
            "price_action": 0.5,  # candle, setup 123, breakout, bandas
            "proximity_to_bands" :0.5,
            "exhaustion": 0.8,
            "penalty_factor": 0.7,
            'macd': 0.1,    # exemplo de peso
            'cci': 0.05,
            'confirmation_candle_penalty': 0.5,
            'divergence': 0.7
        }
        """
        self.weights = {
            "trend": 1.0,         # EMA
            "momentum": 0.8,
            "oscillators": 1.2,
            "price_action": 0.5,  # candle, setup 123, breakout, bandas
            "price_levels" :0.5
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

        """
        self.weights = {
            "trend": params.weights_trend, 
            "rsi": params.weights_rsi,
            "stochastic": params.weights_stochastic,
            "price_action": params.weights_price_action,  
            "proximity_to_bands" : params.weights_proximity_to_bands,
            "exhaustion" : params.weights_exhaustion,
            "penalty_factor": params.weights_penalty_factor,
            'macd': params.weights_macd,    # exemplo de peso
            'cci': params.weights_cci,
            'confirmation_candle_penalty': params.weights_confirmation_candle_penalty,
            'divergence': params.weights_divergence
        }
        """

        self.weights = {
            "trend": params.weights_trend, 
            "momentum": params.weights_momentum,
            "oscillators": params.weights_oscillators,
            "price_action": params.weights_price_action,  
            "price_levels" : params.weights_price_levels,
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
    

       
        #self.indicators = Indicators(self.ohlcv)
        #self.extract_data()
        StrategyUtils.detect_lateral_market(self.ohlcv, self.symbol, self.adx_threshold)

        last_closed = self.ohlcv.get_last_closed_candle()

        print(f"Last closed candle - ts: {last_closed.timestamp}, open: {last_closed.open}, high: {last_closed.high}, low: {last_closed.low}, close: {last_closed.close}, volume: {last_closed.volume}")

 
        is_bearish_reversal, is_bullish_reversal = StrategyUtils.detect_reversal_pattern(self.ohlcv)

        if is_bearish_reversal:
            logging.info(f"{self.symbol} - Reversão de topo detetada: HOLD")
            return SignalResult(Signal.HOLD, None, None)

        if is_bullish_reversal:
            logging.info(f"{self.symbol} - Reversão de fundo detetada: HOLD")
            return SignalResult(Signal.HOLD, None, None)
  
        """
        if StrategyUtils.is_flat_candle(self.ohlcv):
            logging.info(f"{self.symbol} - Candle sem corpo")
            return SignalResult(Signal.HOLD, None, None)
        """
        
        if not StrategyUtils.passes_volume_volatility_filter(self.ohlcv, self.symbol, self.volume_threshold_ratio, self.atr_threshold_ratio):
            logging.info(f"{self.symbol} - Filtro de volume/volatilidade não passou: HOLD")
            return SignalResult(Signal.HOLD, None, None)
        
        if not StrategyUtils.calculate_higher_tf_trend(self.ohlcv_higher, self.adx_threshold):
            logging.info(f"{self.symbol} - Sinal rejeitado por tendência contrária no timeframe maior.: HOLD")
            return SignalResult(Signal.HOLD, None, None)
        
        """
        # ⛔️ Bloqueio por lateralização com ADX
        if self.block_lateral_market:
            adx_threshold = self.adx_threshold
            if StrategyUtils.detect_lateral_market(self.ohlcv, self.symbol, adx_threshold):
                return SignalResult(Signal.HOLD, None, None)  # mercado lateral, ignora sinal
        """
    
        
        """
        is_top, is_bottom = self.is_exhaustion_candle(self.ohlcv)

        if is_top or is_bottom:
            logging.info(f"{self.symbol} - Sinal rejeitado por mercado estar no topo ou resistencia.: HOLD")
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
            return SignalResult(Signal.HOLD, None, None, hold_score, max(buy_score, sell_score))

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

        return SignalResult(signal, sl, tp, hold_score, max(buy_score, sell_score))

    """
    def extract_data(self):
        self.closes = self.indicators.closes
        self.highs = self.indicators.highs
        self.lows = self.indicators.lows
        self.opens = self.indicators.opens

        self.price = self.opens[0]
        self.atr = self.indicators.atr()

        self.ema = self.indicators.ema()[-1]
        self.prev_ema = self.indicators.ema()[-2]
        self.rsi = self.indicators.rsi()[-1]

        stoch_k, stoch_d = self.indicators.stochastic()
        self.k_now, self.d_now = stoch_k[-1], stoch_d[-1]
        self.k_prev, self.d_prev = stoch_k[-2], stoch_d[-2]

        self.open_now = self.opens[-1]
        self.close_now = self.closes[-1]
        self.high_now = self.highs[-1]
        self.low_now = self.lows[-1]

        self.open_prev = self.opens[-2]
        self.close_prev = self.closes[-2]
        self.high_prev = self.highs[-2]
        self.low_prev = self.lows[-2]
    """


        
    """
    def calculate_score(self):
        indicators =Indicators(self.ohlcv)
        opens = indicators.opens
        adx = indicators.adx()
        self.price = opens[0]
        rsi = indicators.rsi()[-1]
        macd_line, macd_signal_line = indicators.macd()
        cci_list = indicators.cci(20)

        macd_val = macd_line[-1]
        macd_signal_val = macd_signal_line[-1]
        cci_val = cci_list[-1]

        score = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        #weights_sum = sum(self.weights.values())

        # 1. Tendência
        trend_signal = StrategyUtils.trend_signal_with_adx(self.ohlcv, self.symbol, self.adx_threshold)
        if trend_signal == 1:
            score["buy"] += 1 * self.weights["trend"]
        elif trend_signal == -1:
            score["sell"] += 1 * self.weights["trend"]

        # 2. RSI
        if rsi < self.rsi_buy_threshold:
            score["buy"] += 1 * self.weights["rsi"]
        elif rsi > self.rsi_sell_threshold:
            score["sell"] += 1 * self.weights["rsi"]

        # 3. Estocástico
        stochastic_signal = StrategyUtils.stochastic(self.ohlcv)
        if stochastic_signal == Signal.BUY:
            score["buy"] += 1 * self.weights["stochastic"]
        elif stochastic_signal == Signal.SELL:
            score["sell"] += 1 * self.weights["stochastic"]

        # 4. Price action
        price_action = StrategyUtils.check_price_action_signals(self.ohlcv)
        if price_action == 'buy':
            score["buy"] += 1 * self.weights["price_action"]
        elif price_action == 'sell':
            score["sell"] += 1 * self.weights["price_action"]

        # 5. Proximidade às bandas
        upper_band, lower_band = StrategyUtils.calculate_bands(self.ohlcv, multiplier=self.multiplier)
        dist_to_upper = abs(self.price - upper_band[-1])
        dist_to_lower = abs(self.price - lower_band[-1])
        band_range = upper_band[-1] - lower_band[-1]

        if band_range > 0:
            if dist_to_lower / band_range < 0.1:
                score["buy"] += 1 * self.weights["proximity_to_bands"]
            elif dist_to_upper / band_range < 0.1:
                score["sell"] += 1 * self.weights["proximity_to_bands"]

        # 6. Verifica lateralidade com método existente
        is_lateral = StrategyUtils.detect_lateral_market(self.ohlcv, self.symbol, self.adx_threshold)
        breakout = StrategyUtils.is_breakout_candle(self.ohlcv, -1)

        if self.block_lateral_market and is_lateral and not breakout:
            self.adx_now = adx[-1]
            multiplier = self.adx_now / (self.adx_threshold + 1e-8)
            score["buy"] *= multiplier
            score["sell"] *= multiplier
            logging.info(f"{self.symbol} - Mercado lateral → Penalização aplicada ao score: x{multiplier:.2f}")
        elif breakout:
            score["buy"] += 0.2
            score["sell"] += 0.2
            logging.info(f"{self.symbol} - Breakout detetado → Bónus aplicado")

        # 7. Peso de exaustão
        is_top, is_bottom = StrategyUtils.is_exhaustion_candle(self.ohlcv)
        if is_top:
            score["buy"] -= self.weights["exhaustion"]
        if is_bottom:
            score["sell"] -= self.weights["exhaustion"]

 
        # 8. Deteta se o candle está em zona de suporte ou resistencia
        support, resistance = StrategyUtils.detect_support_resistance(self.ohlcv)
        price = self.price_ref  # último preço de fecho

        dist_to_res, dist_to_sup = StrategyUtils.get_distance_to_levels(self.ohlcv, self.price_ref, lookback=50)
        penalty_buy = min(1.0, max(0.0, 1 - (dist_to_res / (price * 0.01))))
        penalty_sell = min(1.0, max(0.0, 1 - (dist_to_sup / (price * 0.01))))

        # Penalização: reduz score proporcionalmente (podes ajustar a força multiplicando por um fator)
        score["buy"] *= (1 - penalty_buy * self.weights["penalty_factor"])
        score["sell"] *= (1 - penalty_sell * self.weights["penalty_factor"])


        # --- NOVO: MACD ---
        # 9. Cruzamento MACD simples: se MACD > signal, compra; se MACD < signal, venda
        if macd_val > macd_signal_val:
            score["buy"] += 1 * self.weights.get("macd", 0)
        elif macd_val < macd_signal_val:
            score["sell"] += 1 * self.weights.get("macd", 0)

        # --- NOVO: CCI ---
        # 10. Usar limiares clássicos CCI: < -100 compra, > +100 venda
        if cci_val < -100:
            score["buy"] += 1 * self.weights.get("cci", 0)
        elif cci_val > 100:
            score["sell"] += 1 * self.weights.get("cci", 0)

        # 11. Análise do candle de confirmação, confirmação de sinal forte
        if StrategyUtils.is_weak_confirmation_candle(self.ohlcv.get_last_closed_candle()):
            penalty = self.weights.get("confirmation_candle_penalty", 0.5)
            score["buy"] *= penalty
            score["sell"] *= penalty
            logging.info(f"{self.symbol} - Candle fraco → Penalização do score com fator {penalty}")

        # 12. Deteta divergencias entre RSI e Estocástico
        divergence_signal = StrategyUtils.detect_divergence(self.ohlcv)
        if divergence_signal == 1:
            score["buy"] += 1 * self.weights.get("divergence", 0.7)  # peso configurável
        elif divergence_signal == -1:
            score["sell"] += 1 * self.weights.get("divergence", 0.7)

   
        max_score = sum(self.weights.values())
        
        # Garante que score não seja negativo antes da normalização
        score["buy"] = max(score["buy"], 0)
        score["sell"] = max(score["sell"], 0)

        # Normalização final pelo total dos pesos para garantir escala 0-1
        if max_score > 0:
            score["buy"] /= max_score
            score["sell"] /= max_score
            score["hold"] = 1 - (score["buy"] + score["sell"])
            score["hold"] = max(score["hold"], 0)
        else:
            score["hold"] = 1  # fallback, nenhum peso definido
        return score
    """

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
            penalty = self.weights.get("confirmation_candle_penalty", 0.5)
            if not sell:
                buy_points *= penalty
            else:
                sell_points *= penalty

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

        support, resistance = StrategyUtils.detect_support_resistance(self.ohlcv)
        dist_to_res, dist_to_sup = StrategyUtils.get_distance_to_levels(self.ohlcv, price, lookback=50)
        penalty_buy = min(1.0, max(0.0, 1 - (dist_to_res / (price * 0.01))))
        penalty_sell = min(1.0, max(0.0, 1 - (dist_to_sup / (price * 0.01))))

        proximity_buy *= (1 - penalty_buy * self.weights.get("penalty_factor", 0))
        proximity_sell *= (1 - penalty_sell * self.weights.get("penalty_factor", 0))

        # Exaustão
        is_top, is_bottom = StrategyUtils.is_exhaustion_candle(self.ohlcv)
        if is_top:
            proximity_buy -= self.weights.get("exhaustion", 0)
        if is_bottom:
            proximity_sell -= self.weights.get("exhaustion", 0)

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









