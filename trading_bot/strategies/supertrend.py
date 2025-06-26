import numpy as np

class SuperTrend:
    def __init__(self, exchange, symbol, timeframe, period=10, multiplier=3):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.period = period
        self.multiplier = multiplier

    async def get_signal(self):
        ohlcv = await self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe)
        closes = [c[4] for c in ohlcv]
        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]

        if len(closes) < self.period + 1:
            return 'hold'

        atr = self._calculate_atr(highs, lows, closes)
        hl2 = [(h + l) / 2 for h, l in zip(highs, lows)]
        upperband = [hl2[i] + self.multiplier * atr[i] for i in range(len(atr))]
        lowerband = [hl2[i] - self.multiplier * atr[i] for i in range(len(atr))]

        # Simples lógica de cruzamento
        if closes[-2] < upperband[-2] and closes[-1] > upperband[-1]:
            print("\n📊 Sinal: BUY (SuperTrend)")
            return 'buy'

        if closes[-2] > lowerband[-2] and closes[-1] < lowerband[-1]:
            print("\n📊 Sinal: SELL (SuperTrend)")
            return 'sell'

        print("\n📊 Sinal: HOLD (SuperTrend)")
        return 'hold'

    def _calculate_atr(self, highs, lows, closes):
        trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
               for i in range(1, len(highs))]
        atr = [np.mean(trs[i - self.period:i]) if i >= self.period else 0 for i in range(len(trs))]
        atr.insert(0, 0)
        return atr
