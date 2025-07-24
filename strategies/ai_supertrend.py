import logging

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
        self.rsi:float
        self.rsi_buy_threshold = 40
        self.rsi_sell_threshold = 60
        self.signals = []
        self.price_ref: float = 0.0

        # Variáveis de instância para parâmetros configuráveis (antes constantes)
        self.sl_multiplier_aggressive = 1.5
        self.tp_multiplier_aggressive = 3.0
        self.sl_multiplier_conservative = 2.0
        self.tp_multiplier_conservative = 4.0

        self.volume_threshold_ratio = 0.6
        self.atr_threshold_ratio = 0.6

        self.block_lateral_market = True

        self.weights = {
            "trend": 1.0,         # EMA
            "rsi": 0.8,
            "stochastic": 1.2,
            "price_action": 0.5,  # candle, setup 123, breakout, bandas
            "proximity_to_bands" :0.5,
            "exhaustion": 0.8,
            "penalty_factor": 0.7
        }

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

        self.weights = {
            "trend": params.weights_trend, 
            "rsi": params.weights_rsi,
            "stochastic": params.weights_stochastic,
            "price_action": params.weights_price_action,  
            "proximity_to_bands" : params.weights_proximity_to_bands,
            "exhaustion" : params.weights_exhaustion,
            "penalty_factor": params.weights_penalty_factor
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
    

       
        self.indicators = Indicators(self.ohlcv)
        self.extract_data()
        StrategyUtils.detect_lateral_market(self.ohlcv, self.symbol, self.adx_threshold)

 
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
        score["buy"] = round(score["buy"], 2)
        score["sell"] = round(score["sell"], 2)
        score["hold"] = round(score["hold"], 2)
        raw_score = max(score["buy"], score["sell"])
        #confidence = raw_score / max_possible_score if max_possible_score > 0 else 0.0

        logging.info(f"{self.symbol} - Score BUY: {score['buy']}, SELL: {score['sell']}, HOLD: {score['hold']}")

        if score["buy"] > score["sell"] and score["buy"] >= 0.55:
            signal = Signal.BUY
        elif score["sell"] > score["buy"] and score["sell"] >= 0.55:
            signal = Signal.SELL
        else:
            return SignalResult(Signal.HOLD, None, None, score["hold"], raw_score)
    
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

        return SignalResult(signal, sl, tp, score["hold"], raw_score)

    
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



        
    
    def calculate_score(self):
        score = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        weights_sum = sum(self.weights.values())

        # 1. Tendência
        trend_signal = StrategyUtils.trend_signal_with_adx(self.ohlcv, self.symbol, self.adx_threshold, self.ema, self.prev_ema)
        if trend_signal == 1:
            score["buy"] += 1 * self.weights["trend"]
        elif trend_signal == -1:
            score["sell"] += 1 * self.weights["trend"]

        # 2. RSI
        if self.rsi < self.rsi_buy_threshold:
            score["buy"] += 1 * self.weights["rsi"]
        elif self.rsi > self.rsi_sell_threshold:
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
            adx = self.indicators.adx()
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
    









