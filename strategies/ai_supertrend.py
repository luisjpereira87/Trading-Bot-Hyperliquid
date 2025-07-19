import logging

from commons.enums.mode_enum import ModeEnum
from commons.enums.signal_enum import Signal
from commons.models.signal_result import SignalResult
from commons.models.strategy_base import StrategyBase
from commons.models.strategy_params import StrategyParams
from commons.utils.ohlcv_wrapper import OhlcvWrapper
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
        self.signals = []
        self.price_ref: float = 0.0

        # Variáveis de instância para parâmetros configuráveis (antes constantes)
        self.sl_multiplier_aggressive = 1.5
        self.tp_multiplier_aggressive = 3.0
        self.sl_multiplier_conservative = 2.0
        self.tp_multiplier_conservative = 4.0

        self.volume_threshold_ratio = 0.6
        self.atr_threshold_ratio = 0.6

        self.weights = {
            "trend": 1.0,         # EMA
            "rsi": 0.8,
            "stochastic": 1.2,
            "price_action": 0.5,  # candle, setup 123, breakout, bandas
            "proximity_to_bands" :0.5,
            "exhaustion": 0.8
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

        self.weights = {
            "trend": params.weights_trend, 
            "rsi": params.weights_rsi,
            "stochastic": params.weights_stochastic,
            "price_action": params.weights_price_action,  
            "proximity_to_bands" : params.weights_proximity_to_bands,
            "exhaustion" : params.weights_exhaustion
        }
        
    def set_candles(self, ohlcv):
        self.ohlcv = ohlcv

    def set_higher_timeframe_candles(self, ohlcv_higher: OhlcvWrapper):
        self.ohlcv_higher = ohlcv_higher

    def calculate_higher_tf_trend(self):
        if not hasattr(self, 'ohlcv_higher') or len(self.ohlcv_higher) < 21:
            # Sem dados suficientes para timeframe maior → assume neutro
            return 0

        indicators_htf = Indicators(self.ohlcv_higher)
        closes_htf = indicators_htf.closes
        atr_htf = indicators_htf.atr()
        ema_htf = indicators_htf.ema()
        adx_htf = indicators_htf.adx()

        ema_now = ema_htf[-1]
        ema_prev = ema_htf[-2]
        adx_now = adx_htf[-1]

        lateral = adx_now < self.adx_threshold

        if lateral:
            if ema_now > ema_prev:
                return 1
            elif ema_now < ema_prev:
                return -1
            else:
                return 0
        else:
            # Se mercado não lateral, confia na tendência EMA simples
            if ema_now > ema_prev:
                return 1
            elif ema_now < ema_prev:
                return -1
            else:
                return 0

    async def get_signal(self) -> SignalResult:

        if len(self.ohlcv) == 0 or self.symbol is None:
            logging.error("Tem que executar em primeiro lugar o método required_init")
            return SignalResult(Signal.HOLD, None, None)

        if not self.has_enough_candles():
            logging.info(f"{self.symbol} - Dados insuficientes para cálculo.")
            return SignalResult(Signal.HOLD, None, None)

       
        self.indicators = Indicators(self.ohlcv)
        self.extract_data()
        self.calculate_bands(multiplier=self.multiplier)
        self.detect_lateral_market(adx_threshold=self.adx_threshold)
        
        if not self.passes_volume_volatility_filter():
            logging.info(f"{self.symbol} - Filtro de volume/volatilidade não passou: HOLD")
            return SignalResult(Signal.HOLD, None, None)
        
        if not self.calculate_higher_tf_trend():
            logging.info(f"{self.symbol} - Sinal rejeitado por tendência contrária no timeframe maior.: HOLD")
            return SignalResult(Signal.HOLD, None, None)
        
        is_top, is_bottom = self.is_exhaustion_candle(self.ohlcv)

        if is_top or is_bottom:
            logging.info(f"{self.symbol} - Sinal rejeitado por mercado estar no topo ou resistencia.: HOLD")
            return SignalResult(Signal.HOLD, None, None)

        logging.info(f"{self.symbol} - Modo selecionado: {self.mode}")
        score = self.calculate_score()

        # Confiança normalizada
        score["buy"] = round(score["buy"], 2)
        score["sell"] = round(score["sell"], 2)
        raw_score = max(score["buy"], score["sell"])
        #confidence = raw_score / max_possible_score if max_possible_score > 0 else 0.0

        logging.info(f"{self.symbol} - Score BUY: {score['buy']}, SELL: {score['sell']}, HOLD: {score['hold']}")

        if score["buy"] > score["sell"] and score["buy"] >= 0.55:
            signal = Signal.BUY
        elif score["sell"] > score["buy"] and score["sell"] > 0.55:
            signal = Signal.SELL
        else:
            return SignalResult(Signal.HOLD, None, None, None, raw_score)
    
        try:
            sl, tp = self.calculate_sl_tp(
                        self.price_ref, 
                        signal,
                        atr_value=self.atr[-1],
                        mode=self.mode
                    )
        except Exception as e:
            logging.warning(f"{self.symbol} - Erro ao calcular SL/TP: {e}")
            return SignalResult(Signal.HOLD, None, None, None, 0)

        return SignalResult(signal, sl, tp, None, raw_score)


    def passes_volume_volatility_filter(self):
        volumes = getattr(self.indicators, 'volumes', None)
        if volumes is None or len(volumes) < 20:
            return True  # Sem dados suficientes

        avg_volume = sum(volumes[-20:]) / 20
        current_volume = volumes[-1]

        if current_volume < self.volume_threshold_ratio * avg_volume:
            logging.info(f"{self.symbol} - Volume baixo: {current_volume:.2f} < {self.volume_threshold_ratio*100:.0f}% da média ({avg_volume:.2f})")
            return False

        avg_atr = sum(self.atr[-20:]) / 20
        current_atr = self.atr[-1]

        if current_atr < self.atr_threshold_ratio * avg_atr:
            logging.info(f"{self.symbol} - ATR baixo: {current_atr:.4f} < {self.atr_threshold_ratio*100:.0f}% da média ({avg_atr:.4f})")
            return False

        return True

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

    def calculate_bands(self, multiplier):
        self.upper_band = [self.closes[i] + multiplier * self.atr[i] for i in range(len(self.atr))]
        self.lower_band = [self.closes[i] - multiplier * self.atr[i] for i in range(len(self.atr))]

    def detect_lateral_market(self, adx_threshold):
        adx = self.indicators.adx()
        self.adx_now = adx[-1]
        self.lateral_market = self.adx_now < adx_threshold
        logging.info(f"{self.symbol} - ADX: {self.adx_now:.2f} → Lateral: {self.lateral_market}")
        return self.lateral_market

    def detect_setup_123(self):
        if len(self.closes) < 5:
            return False, False

        h = self.highs[-5:]
        l = self.lows[-5:]

        # Buy Setup 123
        buy_123 = (
            l[2] < l[1] and l[2] < l[3] and  # ponto 2 é mínimo local
            h[3] > h[2] and  # ponto 3 alta local
            l[4] > l[2] and  # ponto 4 confirma subida
            self.price > h[3]  # preço acima de ponto 3
        )

        # Sell Setup 123
        sell_123 = (
            h[2] > h[1] and h[2] > h[3] and  # ponto 2 é máximo local
            l[3] < l[2] and  # ponto 3 baixa local
            h[4] < h[2] and  # ponto 4 confirma queda
            self.price < l[3]  # preço abaixo do ponto 3
        )

        return buy_123, sell_123
    
    def is_breakout_candle(self, idx: int, multiplier: float = 2.0, window: int = 20) -> bool:
        if idx < window:
            return False

        candle_bodies = [abs(self.closes[i] - self.opens[i]) for i in range(idx - window, idx)]
        avg_body = sum(candle_bodies) / window
        current_body = abs(self.closes[idx] - self.opens[idx])

        return current_body > multiplier * avg_body
    
    def check_price_action_signals(self):
       # Prioridade 1: Breakout candle (mais forte)
        if self.is_breakout_candle(-1):
            return "buy"  # ou True, se quiseres só booleano

        # Prioridade 2: Setup 123
        buy_123, sell_123 = self.detect_setup_123()
        if buy_123:
            return "buy"
        if sell_123:
            return "sell"

        # Prioridade 3: Cor da vela atual
        if self.close_now > self.open_now:
            return "buy"
        elif self.close_now < self.open_now:
            return "sell"

        # Se nenhuma condição for satisfeita
        return None
    
    def trend_signal_with_adx(self,ema_now, ema_prev):
        if self.detect_lateral_market(self.adx_threshold):
            if ema_now > ema_prev:
                return 1  # buy
            elif ema_now < ema_prev:
                return -1  # sell
        return 0  # sem sinal
    
    def is_exhaustion_candle(self, candles: OhlcvWrapper, lookback: int = 20, threshold: float = 0.95) -> tuple[bool, bool]:
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
        
    
    def calculate_score(self):
        score = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        weights_sum = sum(self.weights.values())

        # 1. Tendência
        trend_signal = self.trend_signal_with_adx(self.ema, self.prev_ema)
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
        if self.k_now > self.d_now and self.k_prev <= self.d_prev:
            score["buy"] += 1 * self.weights["stochastic"]
        elif self.k_now < self.d_now and self.k_prev >= self.d_prev:
            score["sell"] += 1 * self.weights["stochastic"]

        # 4. Price action
        price_action = self.check_price_action_signals()
        if price_action == 'buy':
            score["buy"] += 1 * self.weights["price_action"]
        elif price_action == 'sell':
            score["sell"] += 1 * self.weights["price_action"]

        # 5. Proximidade às bandas
        dist_to_upper = abs(self.price - self.upper_band[-1])
        dist_to_lower = abs(self.price - self.lower_band[-1])
        band_range = self.upper_band[-1] - self.lower_band[-1]

        if band_range > 0:
            if dist_to_lower / band_range < 0.1:
                score["buy"] += 1 * self.weights["proximity_to_bands"]
            elif dist_to_upper / band_range < 0.1:
                score["sell"] += 1 * self.weights["proximity_to_bands"]

        # 6. ADX filtro de lateralidade
        #if self.lateral_market:
        #    score["buy"] -= 1  * self.weights["adx"]
        #    score["sell"] -= 1 * self.weights["adx"]

        # Peso de exaustão (apenas penaliza o BUY score)
        is_top, is_bottom = self.is_exhaustion_candle(self.ohlcv)
        if is_top:
            score["buy"] -= self.weights["exhaustion"]
        if is_bottom:
            score["sell"] -= self.weights["exhaustion"]

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
    
    def calculate_sl_tp(self, price_ref: float, side: Signal, atr_value, mode: ModeEnum):
        
        atr_avg = atr_value  # já está suavizado
        
        if mode == ModeEnum.AGGRESSIVE:
            sl_dist = self.sl_multiplier_aggressive * atr_avg
            tp_dist = self.tp_multiplier_aggressive * atr_avg
        else:
            sl_dist = self.sl_multiplier_conservative * atr_avg
            tp_dist = self.tp_multiplier_conservative * atr_avg

        if side == Signal.BUY:
            sl = price_ref - sl_dist
            tp = price_ref + tp_dist
        else:
            sl = price_ref + sl_dist
            tp = price_ref - tp_dist

        #print(f"entry_price: {price_ref} SL: {sl} TP:{tp}")
        return sl, tp








