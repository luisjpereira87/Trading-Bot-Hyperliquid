import numpy as np

class AISuperTrend:
    def __init__(self, exchange, symbol, timeframe):
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe

    async def get_signal(self):
        ohlcv = await self.exchange.fetch_ohlcv(self.symbol, timeframe=self.timeframe)
        closes = [c[4] for c in ohlcv]
        highs = [c[2] for c in ohlcv]
        lows = [c[3] for c in ohlcv]

        if len(closes) < 20:
            return 'hold'

        atr = self._calculate_atr(highs, lows, closes, period=10)
        multiplier = 1.5

        upper_band = [closes[i] + multiplier * atr[i] for i in range(len(atr))]
        lower_band = [closes[i] - multiplier * atr[i] for i in range(len(atr))]

        if closes[-2] < lower_band[-2] and closes[-1] > lower_band[-1]:
            print("\n📊 Sinal: BUY (AI SuperTrend)")
            return 'buy'

        if closes[-2] > upper_band[-2] and closes[-1] < upper_band[-1]:
            print("\n📊 Sinal: SELL (AI SuperTrend)")
            return 'sell'

        print("\n📊 Sinal: HOLD (AI SuperTrend)")
        return 'hold'

    def _calculate_atr(self, highs, lows, closes, period=10):
        trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
               for i in range(1, len(highs))]
        atr = [np.mean(trs[i - period:i]) if i >= period else 0 for i in range(len(trs))]
        atr.insert(0, 0)
        return atr
