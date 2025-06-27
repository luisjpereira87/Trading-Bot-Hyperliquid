import logging
from .indicators import Indicators
import numpy as np

class AISuperTrend:
    def __init__(self, exchange, symbol, timeframe, mode='conservative'):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.mode = mode

    async def get_signal(self):
        ohlcv = await self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe)
        if len(ohlcv) < 21:
            logging.info(f"{self.symbol} - Dados insuficientes para cálculo.")
            return 'hold'

        self.indicators = Indicators(ohlcv)
        self.extract_data()
        self.calculate_bands()
        self.detect_lateral_market()

        return self.decide_signal()

    def extract_data(self):
        self.closes = self.indicators.closes
        self.highs = self.indicators.highs
        self.lows = self.indicators.lows

        self.price = self.closes[-1]
        self.ema = self.indicators.ema()[-1]
        self.rsi = self.indicators.rsi()[-1]
        self.atr = self.indicators.atr()

        stoch_k, stoch_d = self.indicators.stochastic()
        self.k_now, self.d_now = stoch_k[-1], stoch_d[-1]
        self.k_prev, self.d_prev = stoch_k[-2], stoch_d[-2]

    def calculate_bands(self, multiplier=1.5):
        self.upper_band = [self.closes[i] + multiplier * self.atr[i] for i in range(len(self.atr))]
        self.lower_band = [self.closes[i] - multiplier * self.atr[i] for i in range(len(self.atr))]

    def detect_lateral_market(self, adx_threshold=20):
        adx = self.indicators.adx()
        self.adx_now = adx[-1]
        self.lateral_market = self.adx_now < adx_threshold
        logging.info(f"{self.symbol} - ADX: {self.adx_now:.2f} → Lateral: {self.lateral_market}")

    def decide_signal(self):
        near_lower = self.price < self.lower_band[-1] * 1.01
        near_upper = self.price > self.upper_band[-1] * 0.99

        buy = (
            self.price > self.ema and
            self.rsi > 45 and
            self.k_prev < self.d_prev and self.k_now > self.d_now and self.k_now < 50
        )

        sell = (
            self.price < self.ema and
            self.rsi < 55 and
            self.k_prev > self.d_prev and self.k_now < self.d_now and self.k_now > 50
        )

        logging.info(
            f"{self.symbol} - Indicadores:"
            f"\n🟢 Price: {self.price}"
            f"\n📈 EMA21: {self.ema}"
            f"\n📊 RSI: {self.rsi}"
            f"\n📉 Stoch K: {self.k_now} | D: {self.d_now} (prev K: {self.k_prev}, D: {self.d_prev})"
            f"\n🟩 Lower Band: {self.lower_band[-1]} | Upper Band: {self.upper_band[-1]}"
            f"\n✅ Near Lower Band: {near_lower}, Near Upper Band: {near_upper}"
            f"\n💡 Buy Cond: {buy}, Sell Cond: {sell}"
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

