import logging

from commons.enums.signal_enum import Signal
from commons.models.signal_result import SignalResult
from commons.models.strategy_base import StrategyBase

from .indicators import Indicators


class AISuperTrend(StrategyBase):
    def __init__(self, exchange, symbol, timeframe, mode='conservative', multiplier=1.2, adx_threshold=20, rsi_buy_threshold=40, rsi_sell_threshold=60):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.mode = mode
        self.multiplier = multiplier
        self.adx_threshold = adx_threshold
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold

    async def get_signal(self) -> SignalResult:
        ohlcv = await self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe)
        if len(ohlcv) < 21:
            logging.info(f"{self.symbol} - Dados insuficientes para cálculo.")
            return SignalResult(Signal.HOLD, None, None)

        self.indicators = Indicators(ohlcv)
        self.extract_data()
        self.calculate_bands(multiplier=self.multiplier)
        self.detect_lateral_market(adx_threshold=self.adx_threshold)

        logging.info(f"{self.symbol} - Modo selecionado: {self.mode}")
        signal = self.decide_signal()

        if signal in [Signal.BUY, Signal.SELL]:
            try:
                sl, tp = self.calculate_sl_tp(
                    entry_price=self.price,
                    side=signal,
                    atr_now=self.atr[-1],
                    mode=self.mode
                )
                return SignalResult(signal, sl, tp)
            except Exception as e:
                logging.warning(f"{self.symbol} - Erro ao calcular SL/TP: {e}")
                return SignalResult(Signal.HOLD, None, None)

        return SignalResult(Signal.HOLD, None, None)

    def extract_data(self):
        self.closes = self.indicators.closes
        self.highs = self.indicators.highs
        self.lows = self.indicators.lows
        self.opens = self.indicators.opens

        self.price = self.closes[-1]
        self.ema = self.indicators.ema()[-1]
        self.rsi = self.indicators.rsi()[-1]
        self.atr = self.indicators.atr()

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

    def detect_setup_123(self):
        if len(self.closes) < 5:
            return False, False

        h = self.highs
        l = self.lows

        h0, h1, h2, h3, h4 = h[-5:]
        l0, l1, l2, l3, l4 = l[-5:]

        buy_123 = (
            l2 < l1 and l2 < l3 and
            h3 > h2 and
            l4 > l2 and
            self.price > h3
        )

        sell_123 = (
            h2 > h1 and h2 > h3 and
            l3 < l2 and
            h4 < h2 and
            self.price < l3
        )

        return buy_123, sell_123

    def decide_signal(self):
        near_lower = self.price < self.lower_band[-1] * 1.01
        near_upper = self.price > self.upper_band[-1] * 0.99

        buy = (
            self.price > self.ema and
            self.rsi > self.rsi_buy_threshold and
            (
                (self.k_prev < self.d_prev and self.k_now > self.d_now and self.k_now < 50) or
                (self.k_now > 80)
            )
        )

        sell = (
            self.price < self.ema and
            self.rsi < self.rsi_sell_threshold and
            (
                (self.k_prev > self.d_prev and self.k_now < self.d_now and self.k_now > 50) or
                (self.k_now < 20)
            )
        )

        bullish_candle = (self.close_now > self.open_now)
        bearish_candle = (self.close_now < self.open_now)

        buy_123, sell_123 = self.detect_setup_123()

        buy = buy and bullish_candle and buy_123
        sell = sell and bearish_candle and sell_123

        logging.info(
            f"{self.symbol} - Indicadores:"
            f"\n🟢 Price: {self.price}"
            f"\n📈 EMA21: {self.ema}"
            f"\n📊 RSI: {self.rsi}"
            f"\n📉 Stoch K: {self.k_now} | D: {self.d_now} (prev K: {self.k_prev}, D: {self.d_prev})"
            f"\n🟩 Lower Band: {self.lower_band[-1]} | Upper Band: {self.upper_band[-1]}"
            f"\n✅ Near Lower Band: {near_lower}, Near Upper Band: {near_upper}"
            f"\n🕯️ Bullish Candle: {bullish_candle}, Bearish Candle: {bearish_candle}"
            f"\n🔁 Setup 123 Buy: {buy_123}, Sell: {sell_123}"
            f"\n💡 Buy Cond (final): {buy}, Sell Cond (final): {sell}"
        )

        if self.mode == 'aggressive':
            return self.aggressive_signal(buy, sell, near_lower, near_upper)
        else:
            return self.conservative_signal(buy, sell, near_lower, near_upper)

    def aggressive_signal(self, buy, sell, near_lower, near_upper, band_threshold=0.02):
        band_range = self.upper_band[-1] - self.lower_band[-1]
        relative_band = band_range / self.price

        if self.lateral_market and relative_band > band_threshold:
            logging.info(f"{self.symbol} - Mercado lateral com bandas largas (rel: {relative_band:.4f}), evitando entrada agressiva.")
            return 'hold'

        if buy and near_lower:
            logging.info(f"{self.symbol} - 🎯 Modo Agressivo: BUY")
            return 'buy'
        if sell and near_upper:
            logging.info(f"{self.symbol} - 🎯 Modo Agressivo: SELL")
            return 'sell'
        return 'hold'

    def conservative_signal(self, buy, sell, near_lower, near_upper):
        if self.lateral_market:
            logging.info(f"{self.symbol} - Mercado lateral e modo conservador: HOLD")
            return 'hold'

        if buy and near_lower:
            logging.info(f"{self.symbol} - 🎯 Modo Conservador: BUY")
            return 'buy'
        if sell and near_upper:
            logging.info(f"{self.symbol} - 🎯 Modo Conservador: SELL")
            return 'sell'
        return 'hold'

    def calculate_sl_tp(self, entry_price, side, atr_now, mode="normal"):
        """
        Calcula SL e TP dinâmicos com base no ATR, percentual e valor absoluto.
        Ajusta parâmetros dinamicamente para diferentes faixas de preço e modos.
        """
        if entry_price < 50:
            sl_pct_base = 0.01
            tp_factor_base = 2.0
        elif entry_price < 500:
            sl_pct_base = 0.008
            tp_factor_base = 2.2
        elif entry_price < 5000:
            sl_pct_base = 0.006
            tp_factor_base = 2.5
        else:
            sl_pct_base = 0.004
            tp_factor_base = 3.0

        if mode == "aggressive":
            sl_pct = sl_pct_base * 0.8
            tp_factor = tp_factor_base * 1.2
        elif mode == "conservative":
            sl_pct = sl_pct_base * 1.5
            tp_factor = tp_factor_base * 0.9
        else:
            sl_pct = sl_pct_base
            tp_factor = tp_factor_base

        if sl_pct > 1:
            logging.warning(f"⚠️ sl_pct parece estar em valor absoluto ({sl_pct}). Esperado valor entre 0 e 1.")

        sl_min_dist = sl_pct * entry_price
        atr_cap = entry_price * 0.015
        atr_now = min(atr_now, atr_cap)

        sl_distance = max(sl_min_dist, atr_now)

        logging.info(f"🔎 Cálculo TP/SL ({mode}):")
        logging.info(f"📈 Preço de entrada: {entry_price}")
        logging.info(f"📊 ATR atual (limitado): {atr_now:.4f}")
        logging.info(f"🧮 Distância mínima SL (%): {sl_min_dist:.4f}")
        logging.info(f"🧮 Distância usada SL: {sl_distance:.4f}")

        if side == "buy":
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_factor * sl_distance
        else:
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_factor * sl_distance

        sl_pct_off = abs((sl_price - entry_price) / entry_price)
        tp_pct_off = abs((tp_price - entry_price) / entry_price)

        if sl_pct_off > 0.10 or tp_pct_off > 0.25:
            logging.warning(f"🚫 SL ou TP fora de range aceitável. SL: {sl_price}, TP: {tp_price}")
            raise ValueError("SL ou TP calculado está fora do intervalo aceitável.")

        logging.info(f"✅ SL final: {sl_price:.2f} ({sl_pct_off*100:.2f}%)")
        logging.info(f"✅ TP final: {tp_price:.2f} ({tp_pct_off*100:.2f}%)")

        return round(sl_price, 2), round(tp_price, 2)




